"""Training windows for the pitch front end: audio and Basic Pitch's targets.

    uv run python -m processing.training.frontend_data --prepare

The front end (``processing.music.frontend``) is trained the way Basic Pitch
was: two-second windows of audio against three frame-level targets -- which
pitches sound (``note``), which attack (``onset``) and, where an f0 track
exists, the pitch contour at three bins per semitone (``contour``).

Sources, all licensed for training and shipping weights:

- **GuitarSet**, players 00-03: the microphone, the pickup mix of the same
  takes, and the three augmented renders of the microphone the verifier
  trains on (``aug-v1``..``aug-v3``, the same samples, from the same seeds).
  GuitarSet is the one source with a real f0 track, so it alone trains the
  contour head against labels.
- **Guitar-TECHS**, players P1-P2, all four chains. The direct input and amp
  microphone train every head; the two room microphones, whose MIDI agrees
  with their attacks only to a 33 ms spread, train ``note`` alone -- the same
  split the verifier makes. Guitar-TECHS has no f0 track, so its contour
  target is the released model's own contour: distillation, so that the head
  the verifier's string cues live in is anchored rather than left to drift.
- **EGFxSet** (Zenodo 7044411, CC BY 4.0): every string and fret 0-22 of one
  Stratocaster, five pickup positions, clean and through three hardware
  distortion pedals. Single notes, so its labels are exact; half the windows
  mix two to four of them into a chord. Nothing else here is distorted by real
  hardware, and nothing else has open strings on an electric in quantity.

Held out exactly as the verifier holds them out: GuitarSet 04 and Guitar-TECHS
P3 validate, GuitarSet 05 is the test player, and IDMT-SMT-GUITAR is never
read here at all.

``--prepare`` renders every source once to 22,050 Hz float16 arrays under
``data/frontend`` so a training window is a memory-mapped slice, and writes
each item's labels beside it.
"""
from __future__ import annotations
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import torch

from processing.music import frontend
from processing.tools import guitarset, guitar_techs

ROOT = Path('data/frontend')
EGFXSET = Path('data/egfxset')
LOWEST_PITCH = 21
TRAIN_PLAYERS = ('00', '01', '02', '03', 'P1', 'P2')
GUITARSET_DOMAINS = ('mic', 'pickup', 'aug-v1', 'aug-v2', 'aug-v3')
TECHS_CHAINS = ('di', 'amp', 'ego', 'exo')
# Guitar-TECHS chains whose labels say what sounded but not when it started.
LOOSE_CHAINS = ('ego', 'exo')
# Real hardware distortion only: EGFxSet's delays would put audible repeats
# into a window that labels one attack, and its reverbs and modulation blur
# the offsets this module estimates from the clean decay.
EGFX_EFFECTS = ('Clean', 'BluesDriver', 'TubeScreamer', 'RAT')
EGFX_NAME = re.compile(r'^(?P<string>[1-6])-(?P<fret>\d+)$')
# EGFxSet numbers strings the guitarist's way, 1 the high E.
EGFX_OPEN = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

# Share of each batch per group. GuitarSet and Guitar-TECHS are close to even,
# so the drills -- 29% of their notes above fret 12 -- cannot dominate, and
# the single notes stay a seasoning.
WEIGHTS = {'guitarset': .45, 'techs': .25, 'techs-loose': .15, 'egfx': .15}


@dataclass(frozen=True)
class Item:
    """One recording to draw windows from."""
    name: str           # domain/name, unique
    group: str          # a key of WEIGHTS
    player: str
    source: Path        # the original audio
    audio: Path         # rendered float16 at 22,050 Hz
    labels: Path        # notes and contour, see ``write_labels``
    onsets: bool = True     # are its attack times trustworthy?
    contours: bool = False  # does it carry an f0 track?
    domain: str = 'mic'
    render: str = 'plain'   # how ``_prepare`` makes the audio
    annotation: Path | None = None


def guitarset_items(players=None, domains=GUITARSET_DOMAINS, root=ROOT):
    found, _ = guitarset.excerpts(Path('data/guitarset'))
    pickups = {e.name: e for e in guitarset.excerpts(Path('data/guitarset'),
                                                     'audio_mono-pickup', '_mix')[0]}
    items = []
    for excerpt in sorted(found, key=lambda excerpt: excerpt.name):
        if players is not None and excerpt.player not in players:
            continue
        for domain in domains:
            source = pickups[excerpt.name].audio if domain == 'pickup' else excerpt.audio
            stem = f'gs-{domain}/{excerpt.name}'
            items.append(Item(name=stem, group='guitarset', player=excerpt.player,
                              source=source, audio=root/'audio'/f'{stem}.npy',
                              labels=root/'labels'/f'gs/{excerpt.name}.npz',
                              contours=True, domain=domain,
                              render='augment' if domain.startswith('aug') else 'plain',
                              annotation=excerpt.annotation))
    return items


def techs_items(players=None, chains=TECHS_CHAINS, root=ROOT):
    items = []
    for excerpt in sorted(guitar_techs.excerpts(), key=lambda excerpt: excerpt.name):
        chain = excerpt.key
        if chain not in chains or (players is not None and excerpt.player not in players):
            continue
        loose = chain in LOOSE_CHAINS
        stem = f'gt-{chain}/{excerpt.name}'
        items.append(Item(name=stem, group='techs-loose' if loose else 'techs',
                          player=excerpt.player, source=excerpt.audio,
                          audio=root/'audio'/f'{stem}.npy',
                          labels=root/'labels'/f'{stem}.npz', onsets=not loose,
                          domain=f'gt-{chain}', annotation=excerpt.annotation))
    return items


def egfx_items(effects=EGFX_EFFECTS, root=ROOT, data=EGFXSET):
    items = []
    for effect in effects:
        for path in sorted((data/'extracted'/effect).rglob('*.wav')):
            fields = EGFX_NAME.match(path.stem)
            if not fields or path.name.startswith('._'):
                continue
            pickup = path.parent.name
            # Labels are read off the clean take of the same note. One pedal
            # file has none -- TubeScreamer/Bridge-Middle/1-23, where every
            # other set has a 1-17 -- so its label cannot be trusted either.
            if not (data/'extracted'/'Clean'/pickup/path.name).is_file():
                continue
            stem = f'egfx/{effect}/{pickup}/{path.stem}'
            items.append(Item(name=stem, group='egfx', player='egfx', source=path,
                              audio=root/'audio'/f'{stem}.npy',
                              labels=root/'labels'/f'{stem}.npz',
                              domain=f'egfx-{effect}', render='single'))
    return items


def training_items(groups=tuple(WEIGHTS)):
    items = []
    if 'guitarset' in groups:
        items += guitarset_items(TRAIN_PLAYERS)
    if 'techs' in groups or 'techs-loose' in groups:
        items += [item for item in techs_items(TRAIN_PLAYERS) if item.group in groups]
    if 'egfx' in groups:
        items += egfx_items()
    return items


def validation_items():
    """The three sets a front-end checkpoint is chosen on: GuitarSet 04 through
    its microphone and its pickup, and Guitar-TECHS P3 through the amp mic."""
    return {'gs04-mic': guitarset_items(('04',), ('mic',)),
            'gs04-pickup': guitarset_items(('04',), ('pickup',)),
            'p3-amp': techs_items(('P3',), ('amp',))}


def contour_points(annotation):
    """GuitarSet's voiced f0 points, all strings: (n, 2) of (seconds, MIDI)."""
    document = json.loads(Path(annotation).read_text())
    points = []
    for block in document.get('annotations', ()):
        if block.get('namespace') != 'pitch_contour':
            continue
        data = block['data']
        for time, value in zip(data['time'], data['value']):
            frequency = float(value.get('frequency', 0.))
            if value.get('voiced', True) and frequency > 0:
                points.append((float(time), 69. + 12.*np.log2(frequency/440.)))
    return np.asarray(points, np.float32).reshape(-1, 2)


def single_note(wave, string, fret, sample_rate=frontend.SAMPLE_RATE):
    """(start, end, pitch) for one EGFxSet tone, from its own envelope.

    The attack is the first 5 ms block within 20 dB of the loudest, less one
    block; the note ends when the envelope falls 40 dB below that peak or the
    file runs out. Each file is one pluck left to ring, so this is all the
    annotation it needs."""
    block = int(sample_rate*.005)
    frames = len(wave)//block
    energy = np.sqrt(np.mean(wave[:frames*block].reshape(frames, block)**2, axis=1) + 1e-12)
    level = 20*np.log10(energy/energy.max())
    loud = np.flatnonzero(level > -20)
    start = max(int(loud[0]) - 1, 0) if len(loud) else 0
    after = np.flatnonzero(level[start:] > -40)
    end = start + int(after[-1]) + 1 if len(after) else frames
    return start*block/sample_rate, end*block/sample_rate, float(EGFX_OPEN[string] + fret)


def write_labels(path, notes, contour=None, string=None):
    """Written to a temporary name and renamed, because the five GuitarSet
    domains of one excerpt share one label file and render in parallel."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = np.asarray(notes, np.float32).reshape(-1, 3)
    contour = np.zeros((0, 2), np.float32) if contour is None else contour
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(temporary, notes=notes, contour=contour,
             strings=np.asarray(string if string is not None else [], np.int8))
    os.replace(temporary, path)


def _prepare(item):
    import librosa
    if item.audio.is_file() and item.labels.is_file():
        return item.name, 'cached'
    if item.render == 'augment':
        from processing.tools import variants
        wave, _, _ = variants.render_wave(variants.Job(
            domain=item.domain, name=item.name.split('/', 1)[1], audio=item.source,
            root=Path('data/guitarset'), augmented=True))
    else:
        wave = librosa.load(str(item.source), sr=frontend.SAMPLE_RATE, mono=True)[0]
    wave = np.asarray(wave, np.float32)
    item.audio.parent.mkdir(parents=True, exist_ok=True)
    np.save(item.audio, wave.astype(np.float16))
    if not item.labels.is_file():
        if item.render == 'single':
            fields = EGFX_NAME.match(item.source.stem)
            assert fields
            string, fret = int(fields['string']), int(fields['fret'])
            # Labels come from the clean take, whose decay the pedals stretch.
            clean = EGFXSET/'extracted'/'Clean'/item.source.parent.name/item.source.name
            reference = (wave if 'Clean' in item.name else np.asarray(
                librosa.load(str(clean), sr=frontend.SAMPLE_RATE, mono=True)[0], np.float32))
            start, end, pitch = single_note(reference, string, fret)
            # The pedals sustain past the clean decay; the onset is theirs.
            if 'Clean' not in item.name:
                start = single_note(wave, string, fret)[0]
                end = max(end, single_note(wave, string, fret)[1])
            write_labels(item.labels, [(start, end, pitch)],
                         string=[6 - string])
        else:
            notes = guitarset.load_notes(item.annotation)
            contour = contour_points(item.annotation) if item.contours else None
            write_labels(item.labels, [(n.start, n.end, n.pitch) for n in notes], contour,
                         [n.string for n in notes])
    return item.name, f'{len(wave)/frontend.SAMPLE_RATE:.1f}s'


def prepare(items, workers=6):
    todo = [item for item in items if not (item.audio.is_file() and item.labels.is_file())]
    print(f'{len(items)} items, {len(todo)} to render', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number, (name, note) in enumerate(pool.map(_prepare, todo, chunksize=4), 1):
            if number % 200 == 0 or number == len(todo):
                print(f'  {number}/{len(todo)} {name}: {note}', flush=True)


def load_labels(item):
    with np.load(item.labels) as stored:
        return stored['notes'], stored['contour']


def _segment(audio, start, length=frontend.WINDOW):
    """``length`` samples from ``start``, zero-padded past either end."""
    out = np.zeros(length, np.float32)
    lo, hi = max(start, 0), min(start + length, len(audio))
    if hi > lo:
        out[lo - start:hi - start] = audio[lo:hi]
    return out


def targets(notes, contour, start, onsets=True, contours=True):
    """Basic Pitch's three frame targets for the window starting at sample
    ``start``: frame k is the CQT centred on sample ``start + k*HOP``."""
    frames, rate, hop = frontend.FRAMES, frontend.SAMPLE_RATE, frontend.HOP
    note = np.zeros((frames, frontend.NOTE_BINS), np.float32)
    onset = np.zeros_like(note)
    for begin, end, pitch in notes:
        column = int(round(pitch)) - LOWEST_PITCH
        if not 0 <= column < frontend.NOTE_BINS:
            continue
        first = int(round((begin*rate - start)/hop))
        last = max(int(round((end*rate - start)/hop)), first + 1)
        if last <= 0 or first >= frames:
            continue
        note[max(first, 0):min(last, frames), column] = 1
        if onsets and 0 <= first < frames:
            onset[first, column] = 1
    shape = (frames, frontend.CONTOUR_BINS)
    target = np.zeros(shape, np.float32)
    if contours and len(contour):
        rows = np.round((contour[:, 0]*rate - start)/hop).astype(np.int64)
        # The released model puts a pitch on the middle of its three sub-bins,
        # one above the bin A0 would round to: measured, its contour lines up
        # with labels shifted one bin up at .45 and unshifted at .17.
        columns = (np.round((contour[:, 1] - LOWEST_PITCH)*frontend.BINS_PER_SEMITONE)
                   .astype(np.int64) + 1)
        keep = (rows >= 0) & (rows < frames) & (columns >= 0) & (columns < shape[1])
        target[rows[keep], columns[keep]] = 1
    return note, onset, target


class Windows(torch.utils.data.Dataset):
    """Random training windows, drawn by group weight then uniformly.

    Indexable for ``torch.utils.data.DataLoader``: item ``i`` of epoch ``e``
    is a fixed draw, so a run is reproducible whatever the worker count."""

    def __init__(self, items, length, weights=WEIGHTS, seed=0):
        self.items = items
        self.length = length
        self.seed = seed
        groups = {}
        for index, item in enumerate(items):
            groups.setdefault(item.group, []).append(index)
        self.groups = sorted(groups)
        self.members = [np.asarray(groups[g]) for g in self.groups]
        total = sum(weights.get(g, 0.) for g in self.groups)
        self.weights = np.asarray([weights.get(g, 0.)/total for g in self.groups])
        # GuitarSet's five domains of one excerpt share one label file.
        shared = {}
        for item in items:
            if item.labels not in shared:
                shared[item.labels] = load_labels(item)
        self.labels = [shared[item.labels] for item in items]
        # Single notes that may be mixed into a chord, grouped by effect and
        # pickup so a mixture is one guitar through one chain.
        self.singles = {}
        for index, item in enumerate(items):
            if item.render == 'single':
                self.singles.setdefault(item.source.parent.parent.name + item.source.parent.name,
                                        []).append(index)
        self._audio = {}

    def __len__(self):
        return self.length

    def audio(self, index):
        if index not in self._audio:
            self._audio[index] = np.load(self.items[index].audio, mmap_mode='r')
        return self._audio[index]

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        group = rng.choice(len(self.groups), p=self.weights)
        index = int(rng.choice(self.members[group]))
        item = self.items[index]
        if item.render == 'single':
            wave, notes = self._chord(index, rng)
            contour = np.zeros((0, 2), np.float32)
            start = 0
        else:
            audio = self.audio(index)
            start = int(rng.integers(0, max(len(audio) - frontend.WINDOW, 1)))
            wave = _segment(audio, start)
            notes, contour = self.labels[index]
        note, onset, target = targets(notes, contour, start, item.onsets, item.contours)
        return {'audio': wave, 'note': note, 'onset': onset, 'contour': target,
                'onset_weight': np.float32(item.onsets),
                'contour_weight': np.float32(item.contours)}

    def _chord(self, index, rng):
        """One single note, or with even odds a chord of two to four from the
        same pickup and pedal, each attacking somewhere in the window."""
        item = self.items[index]
        pool = self.singles[item.source.parent.parent.name + item.source.parent.name]
        count = 1 if rng.random() < .5 else int(rng.integers(2, 5))
        chosen = [index] + [int(j) for j in rng.choice(pool, count - 1, replace=False)]
        wave = np.zeros(frontend.WINDOW, np.float32)
        notes = []
        for j in chosen:
            (begin, end, pitch), = self.labels[j][0]
            # The attack lands between 50 ms and 1.6 s into the window.
            at = rng.uniform(.05, 1.6)
            offset = int(round((begin - at)*frontend.SAMPLE_RATE))
            gain = rng.uniform(.4, 1.) if count > 1 else 1.
            wave += gain*_segment(np.asarray(self.audio(j), np.float32), offset)
            shift = offset/frontend.SAMPLE_RATE
            notes.append((begin - shift, end - shift, pitch))
        peak = np.abs(wave).max()
        if peak > .99:
            wave *= .99/peak
        return wave, np.asarray(notes, np.float32)


def peak_scores(onset, notes, times, tolerance=.05):
    """Every onset-head peak's value and whether it is a real attack, and how
    many real attacks have a peak at or above the candidate threshold.

    This is the space the verifier works in: it only ever sees peaks, so a
    front end is better exactly when its peaks rank real attacks above the
    rest. Returns (scores, labels, found, referenced)."""
    from processing.music import verify
    found = verify.peaks(onset, threshold=.02)
    found[len(times):] = False
    rows, columns = np.nonzero(found)
    by_pitch = {}
    for begin, _, pitch in notes:
        by_pitch.setdefault(int(round(pitch)) - LOWEST_PITCH, []).append(begin)
    label = np.zeros(len(rows), bool)
    for position, (row, column) in enumerate(zip(rows, columns)):
        label[position] = any(abs(times[row] - begin) <= tolerance
                              for begin in by_pitch.get(int(column), ()))
    hit = 0
    strong = onset >= verify.PEAK_THRESHOLD
    for begin, _, pitch in notes:
        column = int(round(pitch)) - LOWEST_PITCH
        if not 0 <= column < onset.shape[1]:
            continue
        near = np.flatnonzero(np.abs(times - begin) <= tolerance)
        if len(near) and (found[near, column] & strong[near, column]).any():
            hit += 1
    return onset[rows, columns], label, hit, len(notes)


def evaluate(model, sets, device, limit=None):
    """Peak AP, candidate recall at the verifier's threshold, and candidates
    per real note, per validation set."""
    from basic_pitch.note_creation import model_frames_to_time
    from sklearn.metrics import average_precision_score
    results = {}
    for name, items in sets.items():
        scores, labels, hits, total, peaks = [], [], 0, 0, 0
        for item in items[:limit]:
            wave = np.load(item.audio).astype(np.float32)
            heads = frontend.infer(model, wave, device)
            times = model_frames_to_time(heads['contour'].shape[0])[:heads['onset'].shape[0]]
            notes, _ = load_labels(item)
            s, l, h, t = peak_scores(heads['onset'], notes, times)
            scores.append(s)
            labels.append(l)
            hits += h
            total += t
            peaks += int((s >= .15).sum())
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        results[name] = {'ap': float(average_precision_score(labels, scores)),
                         'recall': hits/max(total, 1),
                         'candidates': peaks/max(total, 1)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.prepare:
        items = training_items()
        for found in validation_items().values():
            items += found
        prepare(items, args.workers)


if __name__ == '__main__':
    main()
