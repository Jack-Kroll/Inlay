"""Compare fusion on identical audio events and cached vision from a user clip.

Fingertip support and changed placements are diagnostics, not accuracy labels.
Cached posteriors from tab JSONL have three-decimal precision; both variants use
the same rounded evidence. Cache/source fingerprints are retained in metadata.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from processing.music.fusion import Sounded
from processing.music.transcribe import FrameRecord, analyse, transcribe_notes, fingered_rate
from processing.vision import DEFAULT_MODEL
from processing.vision.tracker import Observation, EstimatedFret


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path):
    document = json.loads(path.read_text())
    records = []
    for row in document['records']:
        if row['observation'] is not None:
            observation = row['observation']
            for name in ('neck', 'frets'):
                observation[name] = np.asarray(observation[name], np.float32)
            if observation['nut'] is not None:
                observation['nut'] = np.asarray(observation['nut'], np.float32)
            observation['estimates'] = [EstimatedFret(np.asarray(e['line'], np.float32),
                                                     e['number'], e['method'])
                                        for e in observation['estimates']]
            row['observation'] = Observation(**observation)
        if row['hand_points'] is not None:
            row['hand_points'] = np.asarray(row['hand_points'], np.float32)
        records.append(FrameRecord(**row))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = [json.loads(line) for line in args.notes.read_text().splitlines()]
    summary = next(row for row in source if row['kind'] == 'summary')
    if args.extract:
        options = SimpleNamespace(imgsz=None, max_gap=.5, hand_model=None, hands=2,
                                  max_frames=None, threshold=.5, progress=True)
        records, fps = analyse(args.video, args.model, args.device, options)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(dict(
            video=str(args.video), video_sha256=fingerprint(args.video),
            model=str(args.model), model_sha256=fingerprint(args.model),
            options=vars(options), fps=fps, records=[asdict(r) for r in records]),
            default=encode))
    else:
        metadata = json.loads(args.cache.read_text())
        if (metadata['video_sha256'] != fingerprint(args.video)
                or metadata['model_sha256'] != fingerprint(args.model)):
            raise ValueError('Vision cache does not match this video/model; rerun with --extract')
    records = read_records(args.cache)
    notes = [Sounded(row['pitch'], row['startSeconds'],
                     row['startSeconds'] + row['durationSeconds'], row['amplitude'],
                     np.asarray(row['evidence']['audio_string'])
                     if 'audio_string' in row.get('evidence', {}) else None)
             for row in source if row['kind'] == 'note']
    results, baseline = {}, None
    for weight in (0., .75):
        assigned, groups = transcribe_notes(
            records, notes, summary['tuning'], summary['max_fret'],
            summary['string_inset'], summary['string_order'] == 'flipped',
            audio_weight=summary['audio_string_weight'], fret_weight=weight)
        placements = [asdict(note) for note in assigned.values()]
        if baseline is None:
            baseline = placements
        changes = [dict(pitch=a['pitch'], start=a['start'],
                        before=[a['string'], a['fret']], after=[b['string'], b['fret']])
                   for a, b in zip(baseline, placements)
                   if (a['string'], a['fret']) != (b['string'], b['fret'])]
        results[str(weight)] = dict(notes=len(placements),
                                    fingertip_support=fingered_rate(groups),
                                    open_notes=sum(n['fret'] == 0 for n in placements),
                                    changes=changes, placements=placements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        source_notes=str(args.notes), source_sha256=fingerprint(args.notes),
        caveat='Qualitative replay; same rounded audio evidence; fixed baseline string order.',
        variants=results), indent=2))
    for name, result in results.items():
        print(name, {k: v for k, v in result.items() if k != 'placements'})


if __name__ == '__main__':
    main()
