"""Tuning and pitch/tab conversion. String 1 is the highest string, per packages/music."""
from __future__ import annotations

# E4 B3 G3 D3 A2 E2 as MIDI sounding pitches, string 1 first.
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)
NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def note_name(pitch):
    return f'{NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}'


def validate_tuning(tuning):
    tuning = tuple(int(p) for p in tuning)
    if not 1 <= len(tuning) <= 12 or any(not 0 <= p <= 127 for p in tuning):
        raise ValueError('Tuning must be 1..12 MIDI pitches in 0..127')
    if list(tuning) != sorted(tuning, reverse=True):
        # Reentrant tunings exist, but the string-order convention would silently
        # invert the tab, so require an explicit descending order for now.
        raise ValueError('Tuning must be descending: string 1 is the highest string')
    return tuning


def candidates(pitch, tuning=STANDARD_TUNING, max_fret=22):
    """Every (string, fret) that produces this pitch, lowest fret first.

    Fret 0 is the open string. Ordering is a presentation default only; it is
    not a playability model and must not stand in for positional evidence.
    """
    if max_fret < 0:
        raise ValueError('max_fret must be non-negative')
    found = [(index + 1, int(pitch) - open_pitch)
             for index, open_pitch in enumerate(tuning)
             if 0 <= int(pitch) - open_pitch <= max_fret]
    return sorted(found, key=lambda item: (item[1], item[0]))


def pitch_at(string, fret, tuning=STANDARD_TUNING):
    if not 1 <= string <= len(tuning) or fret < 0:
        raise ValueError('Invalid string or fret')
    return tuning[string - 1] + fret
