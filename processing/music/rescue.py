"""Experimental weak-audio rescue with stable, pitch-matched finger evidence.

Activations are model scores, not calibrated probabilities. A visible finger
is not proof of pressure/plucking; this gate must be judged on annotated clips.
"""
from collections import Counter
from bisect import bisect_left, bisect_right
import numpy as np
from .pitch import LOWEST_BIN
from .fretboard import board_transform
from .fusion import TabNote, finger_contacts
from .hands import Hand


def weak_candidates(activations, duration, onset_threshold=.45, frame_threshold=.35,
                    minimum_duration=.06):
    """Yield actual onset peaks with contiguous post-attack pitch support.

    Uses lower thresholds than normal note detection, so it finds quiet notes
    the main pass dropped. Yields (start, end, pitch, mean frame score, onset score).
    """
    times = activations.times
    if len(times) < 3:
        return
    step = float(np.median(np.diff(times)))
    if not np.isfinite(step) or step <= 0:
        return
    for column in range(activations.note.shape[1]):
        onsets = activations.onset[:, column]
        frames = activations.note[:, column]
        # Local maxima of the onset score that clear the threshold.
        peaks = np.flatnonzero((onsets[1:-1] >= onset_threshold)
                               & (onsets[1:-1] > onsets[:-2])
                               & (onsets[1:-1] >= onsets[2:])) + 1
        for peak in peaks:
            start = float(times[peak])
            if not 0 <= start < duration:
                continue
            # Require energy at the attack; distant sustain cannot confirm it.
            # The note lasts as long as the frame score stays above threshold.
            end = int(peak)
            while end < len(times) and frames[end] >= frame_threshold:
                end += 1
            finish = min(duration, float(times[end]) if end < len(times)
                         else float(times[-1] + step))
            if finish - start < minimum_duration:
                continue
            yield (start, finish, column + LOWEST_BIN,
                   float(np.mean(frames[peak:end])), float(onsets[peak]))


def rescue_notes(activations, records, existing, tuning, max_fret, inset, flipped,
                 duration, onset_threshold=.45, frame_threshold=.35):
    """Add only notes corroborated by >=3 fresh frames and no string conflict.

    Existing notes keep their placement and duration. Candidates must match one
    unique cell/finger in >=2/3 of nearby frames spanning >=40 ms; absent or
    tracked-only frames count against agreement. No open-string inference.
    """
    times = [r.timestamp for r in records]
    contacts_cache = {}
    added = []
    report = {'audio_candidates': 0, 'duplicates': 0, 'visual_rejected': 0,
              'string_conflicts': 0, 'accepted': 0}
    for start, end, pitch, amplitude, onset in sorted(
            weak_candidates(activations, duration, onset_threshold, frame_threshold)):
        report['audio_candidates'] += 1
        # Skip pitches the main pass already reported around this time.
        if any(n.pitch == pitch and (abs(n.start-start) <= .08 or
                                     n.start < end and start < n.end)
               for n in list(existing) + added):
            report['duplicates'] += 1
            continue
        # Look at video frames within 60 ms of the onset and count, per
        # (string, fret, finger), how many show a fingertip squarely on a cell
        # that would produce exactly this pitch.
        low, high = bisect_left(times, start-.06), bisect_right(times, start+.06)
        matches = Counter()
        observed_times = {}
        for index in range(low, high):
            if index not in contacts_cache:
                record = records[index]
                contacts = []
                if (record.state == 'detected' and record.observation is not None
                        and record.hand_points is not None):
                    transform = board_transform(record.observation, strings=len(tuning),
                                                inset=inset, flipped=flipped)
                    if transform is not None:
                        contacts = finger_contacts(Hand(record.hand_points, record.hand_label, 1.),
                                                   transform, max_fret=max_fret)
                contacts_cache[index] = contacts
            # Near the string's centre line, well inside the fret cell, right pitch.
            keys = {(c.string, c.fret, c.finger) for c in contacts_cache[index]
                    if c.on_board and 1 <= c.string <= len(tuning) and c.fret > 0
                    and abs(c.string_float-c.string) <= .25
                    and .08 <= c.fret_float-(c.fret-1) <= .95
                    and tuning[c.string-1]+c.fret == pitch}
            for key in keys:
                matches[key] += 1
                observed_times.setdefault(key, []).append(times[index])
        # Accept only if exactly one position is seen consistently: in >=3
        # frames, >=2/3 of nearby frames, spread over >=40 ms.
        supported = [key for key, count in matches.items()
                     if count >= 3 and count/max(high-low, 1) >= 2/3
                     and max(observed_times[key])-min(observed_times[key]) >= .04]
        if len(supported) != 1:
            report['visual_rejected'] += 1
            continue
        string, fret, finger = supported[0]
        # Don't put it on a string that is already playing another note.
        if any(n.string == string and (abs(n.start-start) <= .05 or
                                       n.start < end and start < n.end)
               for n in list(existing) + added):
            report['string_conflicts'] += 1
            continue
        note = TabNote(pitch, start, end, amplitude, string=string, fret=fret,
                       support='rescued', confidence=min(onset, amplitude), finger=finger,
                       alternatives=[(string, fret)], flags=['visual-audio-rescue'])
        note.evidence = {'onset_score': onset, 'mean_frame_score': amplitude,
                         'matching_frames': matches[supported[0]],
                         'nearby_frames': high-low}
        added.append(note)
    report['accepted'] = len(added)
    return added, report
