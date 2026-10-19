"""Polyphonic pitch detection with Spotify Basic Pitch.

Basic Pitch consumes a whole audio file, so this stage is offline by design.
It reports sounding pitch only: it does not know which string produced a note.
"""
from __future__ import annotations
import contextlib
from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
import shutil
import subprocess
import numpy as np

SAMPLE_RATE = 22050
# Guitar in standard tuning spans E2 (82 Hz) to roughly D6 at fret 22. The
# margins keep handling noise and string squeak out of the note list.
MIN_FREQUENCY = 70.
MAX_FREQUENCY = 1400.


@dataclass(frozen=True)
class NoteEvent:
    """One detected note: times in seconds, MIDI pitch, and loudness 0..1.

    ``string_posterior`` is what the verifier's string head thought of each
    string for this note, highest string first, or None when no verifier ran.
    The tab stage mixes it with the camera's view of the neck; see
    ``processing.music.fusion.AUDIO_STRING_WEIGHT``.
    """
    start: float
    end: float
    pitch: int
    amplitude: float
    string_posterior: np.ndarray | None = field(default=None, compare=False)

    def active(self, timestamp, lead=0.):
        return self.start - lead <= timestamp < self.end


# Basic Pitch's note and onset heads cover 88 semitones from A0.
LOWEST_BIN = 21


@dataclass(frozen=True)
class Evidence:
    """What the network says about one pitch over one span of time."""
    time: float    # Where the onset activation peaked, in seconds.
    onset: float   # Peak onset activation over the window.
    frame: float   # Peak note activation over the window.


@dataclass(frozen=True)
class Posteriorgram:
    """Per-frame activation for every pitch, as the network produced it.

    ``detect_notes`` reports only what survives the global thresholds, but the
    thresholds are applied *after* the network: everything the model saw is in
    these arrays. A stage that already knows which pitch to look for -- a fret
    and string read off the camera, say -- can ask what the audio says at that
    pitch without lowering the thresholds for the whole recording, and without
    a second pass over the waveform.

    ``note`` and ``onset`` are (frames, 88), ``contour`` is (frames, 264) at
    three bins per semitone, and ``times`` is the onset time of each frame in
    seconds.
    """
    note: np.ndarray
    onset: np.ndarray
    times: np.ndarray
    contour: np.ndarray | None = None

    def evidence(self, pitch, start, end):
        """What the network says about ``pitch`` over ``[start, end]`` seconds.

        Returns None when the pitch or the window falls outside what the model
        reported, so a caller cannot mistake "no evidence" for "off the end".
        The reported time is where the *onset* head peaked: a caller that knows
        which pitch to look for gets its attack from the onset head, which is
        the head trained to localise one.
        """
        column = int(round(pitch)) - LOWEST_BIN
        if not 0 <= column < self.note.shape[1]:
            return None
        frames = min(len(self.times), self.note.shape[0])
        if frames <= 0 or end < start:
            return None
        # Convert the time window to a frame range (at least one frame long).
        first = min(max(int(np.searchsorted(self.times, start)), 0), frames - 1)
        last = min(max(int(np.searchsorted(self.times, end)), first + 1), frames)
        if last <= first:
            return None
        onsets = self.onset[first:last, column]
        peak = first + int(np.argmax(onsets))
        return Evidence(time=float(self.times[peak]), onset=float(onsets.max()),
                        frame=float(self.note[first:last, column].max()))


def extract_audio(media, destination, sample_rate=SAMPLE_RATE):
    """Decode the media soundtrack to mono WAV with ffmpeg."""
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required to read audio from video. Install it '
                           '(brew install ffmpeg) or pass --audio with a WAV file.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(media),
         '-vn', '-ac', '1', '-ar', str(sample_rate), str(destination)],
        capture_output=True, text=True)
    if result.returncode or not destination.is_file() or not destination.stat().st_size:
        raise RuntimeError(f'ffmpeg could not extract audio from {media}: '
                           f'{result.stderr.strip() or "no audio stream"}')
    return destination


DEFAULT_VERIFIER = Path('runs/verifier/v2-cqt/best.pt')


def default_verifier(path=DEFAULT_VERIFIER):
    """The shipped verifier checkpoint, or None when it was never trained.

    Model weights are not in the repository, so a fresh clone falls back to
    Basic Pitch's thresholds rather than failing.
    """
    path = Path(path)
    return path if path.is_file() else None


def raw_activations(audio, verbose=False):
    """The posteriorgram as the network produced it, with nothing masked out.

    ``detect_notes`` cannot supply this. Basic Pitch's ``predict`` applies the
    frequency limits to its own arrays *in place* before creating notes, which
    leaves 47 of the 88 pitch columns hard zero. The verifier reads two octaves
    either side of each candidate and was trained on unmasked output, so those
    zeros are input it has never seen -- on a test clip they cost two thirds of
    the notes. Reading the model's output directly keeps the limits where they
    belong: on which candidates the guitar could have played, applied below.
    """
    from basic_pitch.inference import Model, run_inference
    from basic_pitch.note_creation import model_frames_to_time
    from basic_pitch import ICASSP_2022_MODEL_PATH
    audio = Path(audio)
    if not audio.is_file():
        raise FileNotFoundError(audio)
    level = logging.root.manager.disable
    sink = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    if not verbose:
        logging.disable(logging.WARNING)
    try:
        with sink:
            output = run_inference(str(audio), Model(ICASSP_2022_MODEL_PATH))
    finally:
        logging.disable(level)
    note = np.asarray(output['note'], dtype=np.float32)
    return Posteriorgram(
        note=note, onset=np.asarray(output['onset'], dtype=np.float32),
        times=np.asarray(model_frames_to_time(note.shape[0]), dtype=np.float64),
        contour=np.asarray(output['contour'], dtype=np.float32))


def verified_notes(activations, checkpoint, device=None, cqt=None, audio=None,
                   threshold=None, minimum_pitch=None, maximum_pitch=None,
                   verbose=False):
    """Re-decode a posteriorgram with the learned verifier.

    The thresholds in ``detect_notes`` throw away notes the network did see:
    of the notes they missed on GuitarSet, 80% had onset activation above .30.
    This asks a small model trained on GuitarSet about each candidate attack
    instead, and returns notes carrying a per-string posterior for the tab
    stage. See ``processing.music.verify``.
    """
    from . import verify
    if activations.contour is None:
        raise ValueError('the verifier needs the contour head; re-run detect_notes')
    model, channels, _ = verify.load(checkpoint, device)
    spectrum = None
    if channels > verify.CHANNELS:
        if cqt is None:
            if audio is None:
                raise ValueError(f'{checkpoint} was trained with a CQT; pass audio '
                                 'or a precomputed cqt')
            cqt = verify.spectrum(audio)
        spectrum = verify.align_cqt(cqt, activations.times[:activations.note.shape[0]],
                                    activations.note.shape[1])
    prepared = verify.prepare(activations.note, activations.onset,
                              activations.contour, spectrum)
    frames = min(activations.note.shape[0], len(activations.times))
    found = verify.peaks(activations.onset)
    found[frames:] = False
    # The frequency limits belong here rather than on the posteriorgram: reject
    # candidates the guitar could not have played, without blanking the
    # harmonic context the verifier reads around the ones it could.
    if minimum_pitch is not None:
        found[:, :max(0, int(round(minimum_pitch)) - verify.LOWEST_BIN)] = False
    if maximum_pitch is not None:
        found[:, int(round(maximum_pitch)) - verify.LOWEST_BIN + 1:] = False
    rows, columns = np.nonzero(found)
    if not len(rows):
        return []
    rows, columns = rows.astype(np.int64), columns.astype(np.int64)
    exists, deltas, strings = verify.score(model, prepared, rows, columns,
                                           channels, device)
    decoded = verify.decode(activations.note, activations.onset, activations.times,
                            exists, deltas, strings, rows, columns,
                            verify.SCORE_THRESHOLD if threshold is None else threshold)
    if verbose:
        print(f'verifier: {len(rows)} candidates -> {len(decoded)} notes')
    # The verifier's own confidence stands in for loudness: the tab stage uses
    # amplitude only to break ties between notes competing for a string, and a
    # confidence is a better tie-breaker than an amplitude this path never
    # estimated.
    return [NoteEvent(start, end, pitch, confidence, posterior)
            for start, end, pitch, confidence, posterior in decoded]


def detect_notes(audio, onset_threshold=.6, frame_threshold=.45, minimum_note_length=60.,
                 minimum_frequency=MIN_FREQUENCY, maximum_frequency=MAX_FREQUENCY,
                 verbose=False):
    """Return note events sorted by onset, and the activations behind them.

    ``minimum_note_length`` is milliseconds; the default is shorter than Basic
    Pitch's so that quick fretted passages survive.

    These thresholds were measured against GuitarSet rather than guessed; see
    docs/pitch-tuning.md. Basic Pitch's own defaults (.5/.3) report far too
    many notes on guitar, and a note inlay invents is worse than one it misses
    because a spurious pitch takes a string away from a real note.

    The returned `Posteriorgram` is what the network produced before any of
    these thresholds were applied. It costs nothing extra -- the same forward
    pass produced it -- and it is the only way a later stage can ask about a
    pitch this one decided against.
    """
    from basic_pitch.inference import predict
    from basic_pitch.note_creation import model_frames_to_time
    from basic_pitch import ICASSP_2022_MODEL_PATH
    audio = Path(audio)
    if not audio.is_file():
        raise FileNotFoundError(audio)
    level = logging.root.manager.disable
    # The Core ML path prints tensor diagnostics straight to stdout.
    sink = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    if not verbose:
        logging.disable(logging.WARNING)
    try:
        with sink:
            output, _, events = predict(str(audio), ICASSP_2022_MODEL_PATH,
                                        onset_threshold=onset_threshold,
                                        frame_threshold=frame_threshold,
                                        minimum_note_length=minimum_note_length,
                                        minimum_frequency=minimum_frequency,
                                        maximum_frequency=maximum_frequency)
    finally:
        logging.disable(level)
    # Each event is (start, end, pitch, amplitude, pitch_bends).
    notes = [NoteEvent(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in events]
    activations = Posteriorgram(
        note=np.asarray(output['note'], dtype=np.float32),
        onset=np.asarray(output['onset'], dtype=np.float32),
        times=np.asarray(model_frames_to_time(output['note'].shape[0]), dtype=np.float64),
        contour=np.asarray(output['contour'], dtype=np.float32))
    return sorted(notes, key=lambda note: (note.start, note.pitch)), activations


def _pitch_of(frequency):
    """Hz -> MIDI pitch, or None. Basic Pitch takes a cutoff in Hz; the
    verifier works in the posteriorgram's semitone columns."""
    if frequency is None or frequency <= 0:
        return None
    return int(round(69 + 12*np.log2(float(frequency)/440.)))


def notes_from_media(media, audio=None, workdir=None, verifier=None, device=None,
                     **kwargs):
    """Detect notes in a video (audio extracted) or an existing audio file.

    With ``verifier`` set to a checkpoint, the notes are decoded by the learned
    verifier instead of by Basic Pitch's thresholds, and carry a per-string
    posterior. The forward pass is the same either way.

    Returns the notes, the activations they were decoded from, and the audio
    the soundtrack was read out of.
    """
    media = Path(media)
    if audio is None and media.suffix.lower() in ('.wav', '.flac', '.mp3', '.m4a', '.ogg'):
        audio = media
    if audio is None:
        workdir = Path(workdir) if workdir else media.parent
        audio = extract_audio(media, workdir/f'{media.stem}.inlay.wav')
    if verifier is None:
        notes, activations = detect_notes(audio, **kwargs)
        return notes, activations, Path(audio)
    # The verifier replaces the thresholds, so the threshold decoder never runs.
    verbose = kwargs.get('verbose', False)
    activations = raw_activations(audio, verbose)
    notes = verified_notes(
        activations, verifier, device, audio=audio, verbose=verbose,
        minimum_pitch=_pitch_of(kwargs.get('minimum_frequency')),
        maximum_pitch=_pitch_of(kwargs.get('maximum_frequency')))
    return notes, activations, Path(audio)
