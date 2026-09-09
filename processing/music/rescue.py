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
    """Yield actual onset peaks with contiguous post-attack pitch support."""
    times = activations.times
    if len(times) < 3:
        return
    step = float(np.median(np.diff(times)))
    if not np.isfinite(step) or step <= 0:
        return
    for column in range(activations.note.shape[1]):
        onsets = activations.onset[:, column]
        frames = activations.note[:, column]
        peaks = np.flatnonzero((onsets[1:-1] >= onset_threshold)
                               & (onsets[1:-1] > onsets[:-2])
                               & (onsets[1:-1] >= onsets[2:])) + 1
        for peak in peaks:
            start = float(times[peak])
            if not 0 <= start < duration:
                continue
            # Require energy at the attack; distant sustain cannot confirm it.
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
        if any(n.pitch == pitch and (abs(n.start-start) <= .08 or
                                     n.start < end and start < n.end)
               for n in list(existing) + added):
            report['duplicates'] += 1
            continue
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
            keys = {(c.string, c.fret, c.finger) for c in contacts_cache[index]
                    if c.on_board and 1 <= c.string <= len(tuning) and c.fret > 0
                    and abs(c.string_float-c.string) <= .25
                    and .08 <= c.fret_float-(c.fret-1) <= .95
                    and tuning[c.string-1]+c.fret == pitch}
            for key in keys:
                matches[key] += 1
                observed_times.setdefault(key, []).append(times[index])
        supported = [key for key, count in matches.items()
                     if count >= 3 and count/max(high-low, 1) >= 2/3
                     and max(observed_times[key])-min(observed_times[key]) >= .04]
        if len(supported) != 1:
            report['visual_rejected'] += 1
            continue
        string, fret, finger = supported[0]
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


def check_uncertain_notes(records, existing, tuning, max_fret, inset, flipped,
                          uncertain_below=.60):
    """Check weak baseline audio against nearby pitch-compatible fingers.

    Missing visual evidence is unknown, not a rejection. Open-string pitches
    cannot be disproved by absent fingertips. Scores are not probabilities.
    """
    times = [r.timestamp for r in records]
    cache = {}
    kept = []
    report = {'checked': 0, 'confirmed': 0, 'unknown': 0, 'open_exempt': 0,
              'removed': [], 'uncertain_below': uncertain_below}
    for note in existing:
        if note.amplitude >= uncertain_below:
            kept.append(note)
            continue
        report['checked'] += 1
        if note.pitch in tuning:
            note.flags.append('visual-check-open-ambiguous')
            report['open_exempt'] += 1
            kept.append(note)
            continue
        low, high = bisect_left(times,note.start-.08), bisect_right(times,note.start+.08)
        usable = []
        matching = []
        for index in range(low, high):
            if index not in cache:
                r = records[index]
                contacts = None
                if r.state == 'detected' and r.observation is not None and r.hand_points is not None:
                    transform = board_transform(r.observation, strings=len(tuning), inset=inset,
                                                flipped=flipped)
                    if transform is not None:
                        contacts = finger_contacts(Hand(r.hand_points,r.hand_label,1.), transform,
                                                   max_fret=max_fret)
                cache[index] = contacts
            contacts = cache[index]
            if contacts is None or len(contacts) < 4:
                continue
            usable.append(times[index])
            # Test all pitch-compatible cells, including near a fret boundary.
            found = any(c.on_board and abs(c.string_float-c.string) <= .45
                        and 1 <= c.string <= len(tuning)
                        and 1 <= note.pitch-tuning[c.string-1] <= max_fret
                        and note.pitch-tuning[c.string-1]-1-.25 <= c.fret_float
                        <= note.pitch-tuning[c.string-1]+.25 for c in contacts)
            if found:
                matching.append(times[index])
        evidence = {'mean_audio_score': note.amplitude, 'usable_frames': len(usable),
                    'matching_frames': len(matching)}
        note.evidence['reverse_visual_check'] = evidence
        if len(matching) >= 2 and max(matching)-min(matching) >= .025:
            note.flags.append('weak-audio-visually-confirmed')
            report['confirmed'] += 1
            kept.append(note)
        elif len(usable) < 3 or max(usable)-min(usable) < .05:
            note.flags.append('visual-check-unavailable')
            report['unknown'] += 1
            kept.append(note)
        else:
            report['removed'].append({'pitch': note.pitch, 'start': note.start,
                                      'string': note.string, 'fret': note.fret,
                                      'reason': 'weak-audio-without-nearby-finger', **evidence})
    return kept, report
