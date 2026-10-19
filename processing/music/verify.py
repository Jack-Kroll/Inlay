"""A learned note verifier over Basic Pitch's posteriorgram.

Basic Pitch decides a note exists by thresholding one number per pitch per
frame. Measured against GuitarSet, that is where inlay was losing notes: of the
notes it missed, 80% had an onset activation above .30 and 22% were above the
.60 threshold the decoder was using. Almost none -- 0.1% -- had no activation
at all. The evidence was in the posteriorgram; the decoder was discarding it.

A threshold cannot do better, because the number alone is ambiguous: a quiet
fretted note and the octave of a loud one look the same at that pitch and that
instant. What separates them is the surrounding harmonic and temporal context.
Sweeping a hand-written onset-first decoder over that posteriorgram confirmed
it -- its best setting tied the shipped baseline, trading precision for recall
one for one.

So this reads a patch of the posteriorgram around each candidate attack (+/- 10
frames, +/- 2 octaves, over the note, onset and contour heads) and predicts:

  exist  -- is a real note attacking here?
  delta  -- how far the true onset is from this frame, in frames
  string -- which string played it

It reuses the forward pass inlay already runs, so it adds no inference over
the audio. Measured on GuitarSet's held-out test player, against the shipped
thresholds: note F1 .821 -> .896, string accuracy .653 -> .863, and the two
together .536 -> .773. See docs/pitch-tuning.md and docs/verification.md.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# Basic Pitch's note and onset heads cover 88 semitones from A0.
LOWEST_BIN = 21
FPS = 22050/256              # AUDIO_SAMPLE_RATE/FFT_HOP.
FRAME_RADIUS = 10            # Frames either side of the candidate.
PITCH_RADIUS = 24            # Semitones either side: two octaves of harmonics.
FRAME_SPAN = 2*FRAME_RADIUS + 1
PITCH_SPAN = 2*PITCH_RADIUS + 1
CHANNELS = 5                 # note, onset, and the contour head's three sub-bins.
CQT_CHANNELS = 3             # The CQT's three sub-bins per semitone, when used.
PAD = ((FRAME_RADIUS, FRAME_RADIUS), (PITCH_RADIUS, PITCH_RADIUS))

# Candidates come from permissive peak-picking: at this threshold 98.7% of
# GuitarSet's notes have a candidate inside the 50 ms onset tolerance, at 27
# candidates per real note. Raising it throws away notes the verifier could
# have recovered; lowering it to .10 buys 0.6 points of ceiling for seven
# times the candidates.
PEAK_THRESHOLD = .15
PEAK_WIDTH = 1
# GuitarSet's F1 peaks at .85 and is flat up to .93, but F1 is not what a tab
# reader notices. On real clips .85 was dropping notes a human could hear:
# Blackbird reported 25 where Basic Pitch's thresholds found 42, and
# Thunderstruck 93 against 95. At .70 those become 37 and 102; .75 sits just
# above, and the overlays either side of it were judged better by eye than .85.
#
# On GuitarSet's test player the move from .85 down to .70 costs 1.3 points of
# note F1 (.902 -> .889) and 5.2 of precision (.909 -> .857), and buys 2.7 of
# recall (.896 -> .923); .75 is partway along that trade. Nothing here has ground truth for a phone recording of an
# electric guitar, so this is a judgement that a missed note is worse than a
# spurious one for a transcript a person is going to correct by hand -- the
# opposite of the trade the Basic Pitch thresholds were tuned for, where a
# spurious pitch could take a string from a real note. Whole-onset assignment
# weakened that coupling. Pass --verifier-threshold .85 for the F1-optimal
# setting. See docs/note-verifier.md.
SCORE_THRESHOLD = .75
# Notes closer together than the onset tolerance cannot both be real attacks
# on one string, so the weaker is suppressed.
ONSET_TOLERANCE = .05
OFFSET_GATE = .30
OFFSET_RELATIVE = .45


def _sliding_max(column, width, side):
    """Max over the ``width`` frames before (-1) or after (+1) each frame."""
    frames = len(column)
    out = np.full(frames, -np.inf, dtype=np.float32)
    for shift in range(1, width + 1):
        rolled = np.full(frames, -np.inf, dtype=np.float32)
        if side < 0:
            rolled[shift:] = column[:frames - shift]
        else:
            rolled[:frames - shift] = column[shift:]
        out = np.maximum(out, rolled)
    return out


def peaks(onset, threshold=PEAK_THRESHOLD, width=PEAK_WIDTH):
    """Boolean (frames, pitches): where the onset head has a real attack.

    A local maximum, not a threshold crossing: a slow rise or a plateau would
    otherwise yield either nothing or one candidate per frame above the line.
    """
    found = np.zeros_like(onset, dtype=bool)
    for column in range(onset.shape[1]):
        curve = onset[:, column]
        left = _sliding_max(curve, width, -1)
        right = _sliding_max(curve, width, +1)
        # Strict on the left, loose on the right: a plateau yields its first frame.
        found[:, column] = (curve >= threshold) & (curve > left) & (curve >= right)
    return found


def _fold(array, semitones):
    """(frames, 3*semitones) -> (frames, semitones, 3), padded or trimmed to fit."""
    frames = array.shape[0]
    folded = array[:, :(array.shape[1]//3)*3].reshape(frames, -1, 3)
    folded = folded[:, :semitones, :]
    if folded.shape[1] < semitones:
        folded = np.pad(folded, ((0, 0), (0, semitones - folded.shape[1]), (0, 0)))
    return folded


SAMPLE_RATE = 22050
HOP = 256
BINS_PER_OCTAVE = 36
SEMITONES = 88


def spectrum(audio, sample_rate=SAMPLE_RATE):
    """A CQT of one recording, three bins per semitone from A0.

    ``audio`` is a path to read or a mono waveform already at ``sample_rate``;
    training augmentation hands over a waveform it has just filtered, and
    going back through a file would be the only difference between the two.

    Basic Pitch's heads describe pitch, not tone: two strings sounding the same
    pitch differ in brightness and inharmonicity, and that lives in the
    spectrum the network already reduced away. Stacking this on as extra patch
    channels raised string accuracy on GuitarSet's test player from .863 to
    .887 and note F1 from .896 to .898.

    The grid matches the contour head's layout so one patch geometry serves
    both. Normalising by a high percentile rather than the peak means a phone
    recording's gain does not change what a patch looks like.
    """
    import librosa
    wave = (np.asarray(audio, np.float32) if isinstance(audio, np.ndarray)
            else librosa.load(str(audio), sr=sample_rate, mono=True)[0])
    magnitude = np.abs(librosa.cqt(
        wave, sr=sample_rate, hop_length=HOP,
        fmin=float(librosa.midi_to_hz(LOWEST_BIN)),
        n_bins=SEMITONES*3, bins_per_octave=BINS_PER_OCTAVE))
    scaled = np.log1p(magnitude*20.).T.astype(np.float32)
    return scaled/max(1e-6, float(np.percentile(scaled, 99)))


def align_cqt(spectrum, times, semitones):
    """Resample a hop-grid CQT onto Basic Pitch's own frame times.

    Basic Pitch corrects a window offset once every 172 frames, about 10 ms
    each time, so by the end of a thirty-second excerpt its frames sit ten
    frames away from a plain hop grid. Indexing by time rather than by frame
    number is the difference between a patch that lines up and one that does not.
    """
    rows = np.clip(np.round(times*FPS).astype(np.int64), 0, len(spectrum) - 1)
    return _fold(spectrum[rows], semitones)


def prepare(note, onset, contour, cqt=None):
    """Pad one excerpt's heads so a patch is a plain slice, no bounds checks."""
    folded = _fold(contour, note.shape[1])[:note.shape[0]]
    prepared = [np.pad(note, PAD).astype(np.float16),
                np.pad(onset, PAD).astype(np.float16),
                np.pad(folded, PAD + ((0, 0),)).astype(np.float16)]
    if cqt is not None:
        prepared.append(np.pad(cqt[:note.shape[0]], PAD + ((0, 0),)).astype(np.float16))
    return prepared


def fill(destination, prepared, row, column):
    """Copy one patch out of ``prepared`` into ``destination``, channel by channel."""
    rows = slice(row, row + FRAME_SPAN)
    columns = slice(column, column + PITCH_SPAN)
    destination[0] = prepared[0][rows, columns]
    destination[1] = prepared[1][rows, columns]
    destination[2:CHANNELS] = prepared[2][rows, columns].transpose(2, 0, 1)
    if len(prepared) > 3:
        destination[CHANNELS:] = prepared[3][rows, columns].transpose(2, 0, 1)


def cut(prepared, frame, pitch, channels):
    """(n, channels, FRAME_SPAN, PITCH_SPAN) patches from one excerpt."""
    out = np.empty((len(frame), channels, FRAME_SPAN, PITCH_SPAN), np.float32)
    for position, (row, column) in enumerate(zip(frame, pitch)):
        fill(out[position], prepared, int(row), int(column))
    return out


def build(width=48, channels=CHANNELS):
    """The verifier network. Imported lazily so numpy-only callers skip torch."""
    import torch.nn as nn

    class Verifier(nn.Module):
        """A small CNN over the patch, plus the candidate's absolute pitch."""

        def __init__(self):
            super().__init__()
            def block(inputs, outputs, stride=1):
                return nn.Sequential(
                    nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))
            self.stem = nn.Sequential(
                block(channels, width), block(width, width),
                block(width, 2*width, stride=2), block(2*width, 2*width),
                block(2*width, 4*width, stride=2), block(4*width, 4*width))
            self.pool = nn.AdaptiveAvgPool2d(1)
            # Absolute pitch matters: the harmonic series below a low E is
            # empty, so the same patch means different things at different
            # pitches.
            self.trunk = nn.Sequential(nn.Linear(4*width + 2, 128),
                                       nn.ReLU(inplace=True), nn.Dropout(.1))
            self.exist = nn.Linear(128, 1)
            self.delta = nn.Linear(128, 1)
            self.string = nn.Linear(128, 6)

        def forward(self, patch, pitch):
            import torch
            features = self.pool(self.stem(patch)).flatten(1)
            # Pitch as a smooth pair rather than a bare index.
            extra = torch.stack([(pitch - 50.)/25., ((pitch - 50.)/25.)**2], dim=1)
            hidden = self.trunk(torch.cat([features, extra], dim=1))
            return (self.exist(hidden).squeeze(1), self.delta(hidden).squeeze(1),
                    self.string(hidden))

    return Verifier()


def load(checkpoint, device=None):
    """Load a trained verifier. Returns (model, channels, cqt cache or None)."""
    import torch
    checkpoint = Path(checkpoint)
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    channels = int(stored.get('channels', CHANNELS))
    model = build(int(stored.get('width', 48)), channels)
    model.load_state_dict(stored['model'])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, channels, (stored.get('cqt') or None)


def suppress(frame, pitch, score, window):
    """Keep the best candidate per pitch within ``window`` frames of it."""
    kept, claimed = [], {}
    for index in np.argsort(-score):
        column, row = int(pitch[index]), int(frame[index])
        if any(abs(row - other) <= window for other in claimed.get(column, ())):
            continue
        claimed.setdefault(column, []).append(row)
        kept.append(index)
    return np.array(kept, dtype=np.int64)


def score(model, prepared, frame, pitch, channels, device=None, batch=1024):
    """Run the verifier over every candidate. Returns (exist, delta, string)."""
    import torch
    exists = np.empty(len(frame), np.float32)
    deltas = np.empty(len(frame), np.float32)
    strings = np.empty((len(frame), 6), np.float32)
    with torch.no_grad():
        for start in range(0, len(frame), batch):
            chunk = slice(start, start + batch)
            patch = torch.from_numpy(cut(prepared, frame[chunk], pitch[chunk], channels))
            column = torch.from_numpy(pitch[chunk].astype(np.float32))
            if device is not None:
                patch, column = patch.to(device), column.to(device)
            exist, delta, string = model(patch, column)
            exists[chunk] = torch.sigmoid(exist).float().cpu().numpy()
            deltas[chunk] = delta.float().cpu().numpy()
            # The head predicts GuitarSet's low-E-first order; inlay numbers
            # strings from the high E.
            strings[chunk] = torch.softmax(string, 1).float().cpu().numpy()[:, ::-1]
    return exists, deltas, strings


def decode(note, onset, times, exists, deltas, strings, frame, pitch,
           threshold=SCORE_THRESHOLD, refine=True):
    """Verifier scores -> (start, end, midi, confidence, string posterior).

    Offsets still come from the frame head: the verifier answers whether a note
    started, not how long it rang.
    """
    frames = min(note.shape[0], len(times))
    keep = np.flatnonzero(exists >= threshold)
    if not len(keep):
        return []
    kept = keep[suppress(frame[keep], pitch[keep], exists[keep],
                         int(round(ONSET_TOLERANCE*FPS)))]
    # Sort by pitch then frame so each note's successor on its own pitch is the
    # next row: that is what ends a note when the string is plucked again.
    kept = kept[np.lexsort((frame[kept], pitch[kept]))]
    out = []
    for position, index in enumerate(kept):
        row, column = int(frame[index]), int(pitch[index])
        limit = frames
        if position + 1 < len(kept) and int(pitch[kept[position + 1]]) == column:
            limit = int(frame[kept[position + 1]])
        support = float(note[row:min(frames, row + 3), column].max())
        floor = max(OFFSET_GATE, OFFSET_RELATIVE*support)
        end = row + 1
        while end < limit and note[end, column] >= floor:
            end += 1
        start = float(times[row])
        if refine:
            start = max(0., start + float(deltas[index])/FPS)
        finish = max(start + 1/FPS, float(times[min(end, frames - 1)]))
        out.append((start, finish, column + LOWEST_BIN, float(exists[index]),
                    strings[index].copy()))
    out.sort(key=lambda row: (row[0], row[2]))
    return out
