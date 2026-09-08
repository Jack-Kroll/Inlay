"""Polyphonic pitch detection with Spotify Basic Pitch.

Basic Pitch consumes a whole audio file, so this stage is offline by design.
It reports sounding pitch only: it does not know which string produced a note.
"""
from __future__ import annotations
import contextlib
from dataclasses import dataclass
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
    start: float
    end: float
    pitch: int
    amplitude: float

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

    ``note`` and ``onset`` are (frames, 88), ``times`` is the onset time of each
    frame in seconds.
    """
    note: np.ndarray
    onset: np.ndarray
    times: np.ndarray

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
    notes = [NoteEvent(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in events]
    activations = Posteriorgram(
        note=np.asarray(output['note'], dtype=np.float32),
        onset=np.asarray(output['onset'], dtype=np.float32),
        times=np.asarray(model_frames_to_time(output['note'].shape[0]), dtype=np.float64))
    return sorted(notes, key=lambda note: (note.start, note.pitch)), activations


def notes_from_media(media, audio=None, workdir=None, **kwargs):
    """Detect notes in a video (audio extracted) or an existing audio file.

    Returns the notes, the activations they were decoded from, and the audio
    the soundtrack was read out of.
    """
    media = Path(media)
    if audio is None and media.suffix.lower() in ('.wav', '.flac', '.mp3', '.m4a', '.ogg'):
        audio = media
    if audio is None:
        workdir = Path(workdir) if workdir else media.parent
        audio = extract_audio(media, workdir/f'{media.stem}.inlay.wav')
    notes, activations = detect_notes(audio, **kwargs)
    return notes, activations, Path(audio)
