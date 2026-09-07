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


def detect_notes(audio, onset_threshold=.5, frame_threshold=.3, minimum_note_length=70.,
                 minimum_frequency=MIN_FREQUENCY, maximum_frequency=MAX_FREQUENCY,
                 verbose=False):
    """Return note events sorted by onset.

    ``minimum_note_length`` is milliseconds; the default is shorter than Basic
    Pitch's so that quick fretted passages survive.
    """
    from basic_pitch.inference import predict
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
            _, _, events = predict(str(audio), ICASSP_2022_MODEL_PATH,
                                   onset_threshold=onset_threshold,
                                   frame_threshold=frame_threshold,
                                   minimum_note_length=minimum_note_length,
                                   minimum_frequency=minimum_frequency,
                                   maximum_frequency=maximum_frequency)
    finally:
        logging.disable(level)
    notes = [NoteEvent(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in events]
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def notes_from_media(media, audio=None, workdir=None, **kwargs):
    """Detect notes in a video (audio extracted) or an existing audio file."""
    media = Path(media)
    if audio is None and media.suffix.lower() in ('.wav', '.flac', '.mp3', '.m4a', '.ogg'):
        audio = media
    if audio is None:
        workdir = Path(workdir) if workdir else media.parent
        audio = extract_audio(media, workdir/f'{media.stem}.inlay.wav')
    return detect_notes(audio, **kwargs), Path(audio)
