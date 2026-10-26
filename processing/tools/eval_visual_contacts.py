"""Audit/cache public Kaggle finger labels and measure placement with oracle pitches.

This is a finger-placement diagnostic, NOT sounding-note transcription accuracy.
The held-out recordings share the performer/setup and are not a new-player test.
Images and labels are joined by their explicit recording/frame IDs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from processing.music import fusion
from processing.music.fretboard import board_transform, DEFAULT_STRING_INSET
from processing.music.hands import FRETTING_FINGERS, HandTracker, fretting_hand
from processing.music.tuning import STANDARD_TUNING, candidates
from processing.vision import DEFAULT_MODEL
from processing.vision.model import DenseDetector
from processing.vision.tracker import FretboardTracker, decode_maps

# Fixed before any scores were calculated. No random frame splitting.
HELD_OUT = {6, 7, 8, 15, 16, 17, 21, 22, 23, 24}


def entries(root):
    """Validated frame rows; return an audit including absent images/timestamps."""
    times = {}
    for row in csv.DictReader((root / 'timestamps.csv').open()):
        if row['frame'] in times:
            raise ValueError(f"Duplicate timestamp ID: {row['frame']}")
        times[row['frame']] = row
    rows, missing, absent_time = [], [], []
    for path in sorted((root / 'tablature_labels').glob('*.npy'),
                       key=lambda p: int(p.stem)):
        labels = np.load(path, allow_pickle=False)
        if labels.ndim != 3 or labels.shape[1:] != (4, 3):
            raise ValueError(f'Unexpected finger label shape: {path}: {labels.shape}')
        for index, label in enumerate(labels):
            name = f'{path.stem}_{index}.png'
            if not (root / 'tablature_frames' / name).is_file():
                missing.append(name)
                continue
            if name not in times:
                absent_time.append(name)
                continue
            row = times[name]
            if Path(row['audio']).stem != path.stem:
                raise ValueError(f'Audio/frame recording mismatch: {name}')
            rows.append(dict(frame=name, recording=int(path.stem),
                             timestamp=float(row['timestamp']), label=label.tolist()))
    used = {row['frame'] for row in rows}
    return rows, dict(labelled_images=len(rows), missing_images=missing,
                      missing_timestamps=absent_time,
                      unused_timestamp_rows=sorted(set(times) - used),
                      held_out_recordings=sorted(HELD_OUT))


def extract(root, output, model, device, limit=0):
    rows, audit = entries(root)
    detector = DenseDetector(model, device)
    tracker, hands, previous = None, None, None
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(audit=audit, model=str(model),
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    inset=DEFAULT_STRING_INSET, flipped=False,
                    protocol='All available labelled stills in timestamp order per recording')
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2))
    started = time.perf_counter()
    try:
        with output.open('w') as stream:
            for index, row in enumerate(rows[:limit or len(rows)]):
                if row['recording'] != previous:
                    if hands is not None:
                        hands.close()
                    hands = HandTracker()
                    tracker = FretboardTracker()
                    previous = row['recording']
                assert hands is not None and tracker is not None
                frame = cv2.imread(str(root / 'tablature_frames' / row['frame']))
                if frame is None:
                    raise ValueError(f"Unreadable image: {row['frame']}")
                observation = tracker.update(frame, decode_maps(detector.predict(frame)),
                                             row['timestamp'])
                hand = fretting_hand(hands.detect(frame, row['timestamp']), observation)
                transform = board_transform(observation) if observation is not None else None
                contacts = fusion.finger_contacts(hand, transform)
                encoded = []
                for contact in contacts:
                    item = asdict(contact)
                    item['point'] = contact.point.tolist()
                    item['on_board'] = bool(contact.on_board)
                    encoded.append(item)
                quality = (dict(residual=transform.residual, edge_residual=transform.edge_residual,
                                anchors=transform.anchors,
                                nut=observation is not None and observation.nut is not None,
                                state=tracker.state) if transform is not None else None)
                stream.write(json.dumps(dict(**row, board=transform is not None,
                                             hand=hand is not None, contacts=encoded,
                                             quality=quality)) + '\n')
                if (index + 1) % 50 == 0:
                    stream.flush()
                    print(f'{index+1}/{min(limit or len(rows), len(rows))} frames, '
                          f'{time.perf_counter()-started:.0f}s', flush=True)
    finally:
        if hands is not None:
            hands.close()
    print(json.dumps(audit, indent=2))


def measure(rows, split, fret_weight=0.):
    """Same frames and oracle pitches for every variant; no sound/press inference."""
    counts = Counter()
    for row in rows:
        if (row['recording'] in HELD_OUT) != (split == 'test'):
            continue
        counts['frames'] += 1
        counts['board'] += row['board']
        counts['hand'] += row['hand']
        contacts = [fusion.FingerContact(**c) for c in row['contacts']]
        by_finger = {c.finger: c for c in contacts}
        for finger, (press, fret, string) in zip(FRETTING_FINGERS, row['label']):
            # Barre annotations have no single string; do not invent one.
            if press != 1 or not 1 <= string <= 6:
                continue
            counts['labelled_contacts'] += 1
            c = by_finger.get(finger)
            if c is not None and c.on_board:
                counts['located'] += 1
                counts['correct_fret'] += c.fret == fret
                counts['correct_string'] += c.string == string
                counts['correct_cell'] += c.fret == fret and c.string == string
            # The label supplies pitch only, not which contact the fuser should use.
            note = fusion.Sounded(STANDARD_TUNING[string - 1] + fret, 0., .1)
            assigned = fusion.assign_positions([note], contacts, board=row['board'],
                                               hand=row['hand'], audio_weight=0.,
                                               fret_weight=fret_weight)[0]
            counts['oracle_correct'] += assigned.string == string
            if len(candidates(note.pitch)) > 1:
                counts['ambiguous_contacts'] += 1
                counts['ambiguous_correct'] += assigned.string == string
    result: dict[str, int | float] = dict(counts)
    denominator = counts['labelled_contacts']
    for key in ('correct_fret', 'correct_string', 'correct_cell', 'oracle_correct'):
        result[key + '_rate'] = counts[key] / denominator if denominator else 0.
    result['ambiguous_rate'] = (counts['ambiguous_correct'] / counts['ambiguous_contacts']
                                if counts['ambiguous_contacts'] else 0.)
    return result


def stabilize(rows, window=.2):
    """Research ablation: interpolate only between agreeing bracketing frames.

    Uses predictions only. Never crosses a recording or pools two hand poses
    more than .75 fret / 1 string apart. This is not yet a runtime default.
    """
    result = deepcopy(rows)
    for index in range(1, len(rows) - 1):
        before, current, after = rows[index - 1:index + 2]
        if (before['recording'] != current['recording']
                or after['recording'] != current['recording']
                or not 0 < current['timestamp'] - before['timestamp'] <= window
                or not 0 < after['timestamp'] - current['timestamp'] <= window):
            continue
        previous = {c['finger']: c for c in before['contacts'] if c['on_board']}
        following = {c['finger']: c for c in after['contacts'] if c['on_board']}
        present = {c['finger']: c for c in result[index]['contacts']}
        for name in previous.keys() & following.keys():
            a, b = previous[name], following[name]
            if (abs(a['fret_float'] - b['fret_float']) > .75
                    or abs(a['string_float'] - b['string_float']) > 1.):
                continue
            c = present.get(name)
            if c is not None and c['on_board']:
                # Do not overwrite a different pose that happened in between.
                if (max(abs(c['fret_float'] - p['fret_float']) for p in (a, b)) > .75
                        or max(abs(c['string_float'] - p['string_float']) for p in (a, b)) > 1.):
                    continue
                samples = [a, c, b]
            else:
                samples = [a, b]
            pooled = deepcopy(a)
            for key in ('fret_float', 'string_float'):
                pooled[key] = float(np.median([p[key] for p in samples]))
            pooled['fret'] = int(np.ceil(pooled['fret_float'] - 1e-9))
            pooled['string'] = min(6, max(1, round(pooled['string_float'])))
            present[name] = pooled
        result[index]['contacts'] = list(present.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(
        'data/visual-benchmarks/kaggle/tablature_dataset/tablature_dataset'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--fret-weight', type=float, default=0.)
    parser.add_argument('--temporal-window', type=float, default=0.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.extract:
        extract(args.root, args.cache, args.model, args.device, args.limit)
    rows = [json.loads(line) for line in args.cache.read_text().splitlines()]
    if args.temporal_window:
        rows = stabilize(rows, args.temporal_window)
    result = measure(rows, args.split, args.fret_weight)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
