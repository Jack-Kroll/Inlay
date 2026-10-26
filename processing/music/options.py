"""What a transcription can be asked to do differently, without the pipeline.

Kept apart from ``transcribe`` so that something which only needs to check a
request -- the API process, deciding whether to queue a job -- can do so
without importing the models.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from processing.vision import DEFAULT_MODEL
from .fretboard import DEFAULT_STRING_INSET
from .pitch import default_verifier
from .tuning import STANDARD_TUNING, validate_tuning


@dataclass(frozen=True)
class Options:
    """Everything a transcription can be asked to do differently.

    The CLI builds one from its flags and the service from a job's options;
    both then call ``transcribe_video``. Defaults are the shipped settings."""
    model: Path = DEFAULT_MODEL
    device: str = 'mps'
    imgsz: int | None = None
    threshold: float = .5
    max_gap: float = .5
    max_fret: int = 22
    inset: float = DEFAULT_STRING_INSET
    string_order: str = 'auto'
    tuning: tuple[int, ...] = tuple(STANDARD_TUNING)
    hands: int = 2
    hand_model: Path | None = None
    # None with use_verifier means the shipped checkpoint, when it exists.
    verifier: Path | None = None
    use_verifier: bool = True
    verifier_threshold: float | None = None
    audio_string_weight: float | None = None
    visual_priority: bool = False
    recover_strong_attacks: bool = False
    open_audio_neutral: float | None = None
    # The fallback decoder's thresholds, when no verifier is used; tuned on
    # GuitarSet, see docs/pitch-tuning.md.
    onset_threshold: float = .6
    frame_threshold: float = .45
    min_note_ms: float = 60.
    mirror: bool = False
    show_strings: bool = True
    write_video: bool = True
    max_frames: int | None = None
    progress: bool = False

    def resolved_verifier(self):
        """The verifier checkpoint this run will use, or None for thresholds."""
        if not self.use_verifier:
            return None
        return self.verifier if self.verifier is not None else default_verifier()

    def problems(self):
        """Why these options cannot run, as messages; empty when they can."""
        found = []
        unit = lambda value: value is None or 0 <= value <= 1
        if not 0 < self.threshold < 1 or not 0 <= self.inset < .5 or self.max_fret < 1:
            found.append('Invalid threshold, inset or max-fret')
        if self.hands < 1 or (self.max_frames is not None and self.max_frames < 1):
            found.append('Invalid hands or max-frames')
        if not unit(self.verifier_threshold):
            found.append('Verifier threshold must be between 0 and 1')
        if not unit(self.audio_string_weight) or not unit(self.open_audio_neutral):
            found.append('Fusion weights must be between 0 and 1')
        if (not np.isfinite(self.max_gap) or self.max_gap <= 0
                or not 0 <= self.onset_threshold <= 1 or not 0 <= self.frame_threshold <= 1
                or not np.isfinite(self.min_note_ms) or self.min_note_ms <= 0
                or (self.imgsz is not None and (self.imgsz < 64 or self.imgsz % 32))):
            found.append('Invalid max-gap, pitch thresholds, min-note-ms or imgsz')
        if self.string_order not in ('auto', 'normal', 'flipped'):
            found.append(f'Unknown string order {self.string_order!r}')
        try:
            validate_tuning(list(self.tuning))
        except ValueError as error:
            found.append(str(error))
        if self.verifier is not None and not Path(self.verifier).is_file():
            found.append(f'No verifier checkpoint at {self.verifier}')
        if self.recover_strong_attacks and self.resolved_verifier() is None:
            found.append('--recover-strong-attacks requires a verifier checkpoint')
        return found
