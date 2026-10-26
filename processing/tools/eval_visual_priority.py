"""Measure stable visual priority using public finger labels and real audio cues.

Pitch comes from each finger label. Audio string posteriors are used only when
an actual onset proposal is within 50 ms at that pitch. This isolates string
placement; it is not a sounded-note or complete-video accuracy benchmark.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

from processing.music.fusion import FingerContact, Sounded, assign_positions
from processing.music.hands import FRETTING_FINGERS
from processing.music.tuning import STANDARD_TUNING, candidates
from processing.music.visual_evidence import VisualContext, VisualPolicy
from processing.music import verify
from processing.tools.eval_visual_contacts import HELD_OUT


POLICIES = {
    'stable': replace(VisualPolicy(), window=.2),
    'relaxed': replace(VisualPolicy(), window=.2, min_contact=.5, min_margin=.06,
                       max_fret_motion=.25, max_string_motion=.5),
    'strict': replace(VisualPolicy(), window=.2, min_contact=.85, min_margin=.12,
                      max_fret_motion=.25, max_string_motion=.25,
                      max_residual=.04, max_edge_residual=.005),
}


def evaluate(rows, audio_root, policy, split):
    counts = Counter()
    cases = []
    recordings = sorted({row['recording'] for row in rows
                         if (row['recording'] in HELD_OUT) == (split == 'test')})
    for recording in recordings:
        frames = [row for row in rows if row['recording'] == recording]
        samples = [dict(timestamp=row['timestamp'], quality=row['quality'],
                        contacts=[FingerContact(**c) for c in row['contacts']]) for row in frames]
        context = VisualContext(samples, policy)
        with np.load(audio_root/f'{recording}.npz') as audio:
            pitches = audio['pitch'] + verify.LOWEST_BIN
            times = audio['times'][audio['frame']]
            for row, sample in zip(frames, samples):
                for finger, (press, fret, string) in zip(FRETTING_FINGERS, row['label']):
                    if press != 1 or not 1 <= string <= 6:
                        continue
                    counts['labels'] += 1
                    pitch = STANDARD_TUNING[string-1] + fret
                    nearby = np.flatnonzero((pitches == pitch) & (np.abs(times-row['timestamp']) <= .05))
                    posterior = None
                    if len(nearby):
                        chosen = nearby[np.argmin(np.abs(times[nearby]-row['timestamp']))]
                        posterior = audio['strings'][chosen]
                        counts['with_audio_proposal'] += 1
                    note = Sounded(pitch, row['timestamp'], row['timestamp']+.05,
                                   string_posterior=posterior)
                    contacts = sample['contacts']
                    baseline = assign_positions([note], contacts, board=row['board'], hand=row['hand'])[0]
                    proof = context.placement(row['timestamp'], pitch)
                    placed = assign_positions([note], contacts, board=row['board'], hand=row['hand'],
                                               visual_evidence=[proof])[0]
                    ambiguous = len(candidates(pitch)) > 1
                    counts['baseline_correct'] += baseline.string == string
                    counts['priority_correct'] += placed.string == string
                    if ambiguous and posterior is not None:
                        counts['ambiguous_with_audio'] += 1
                        counts['ambiguous_baseline_correct'] += baseline.string == string
                        counts['ambiguous_priority_correct'] += placed.string == string
                    if proof is not None:
                        counts['stable_proofs'] += 1
                        counts['proof_correct'] += proof['string'] == string
                    applied = 'visual_priority' in placed.evidence
                    counts['priority_applied'] += applied
                    if baseline.string != placed.string:
                        counts['changed'] += 1
                        counts['corrected'] += placed.string == string
                        counts['harmed'] += baseline.string == string
                        cases.append(dict(frame=row['frame'], finger=finger, pitch=pitch,
                                          reference=string, baseline=baseline.string,
                                          priority=placed.string, proof=proof))
    result: dict[str, int | float | None] = dict(counts)
    result['baseline_accuracy'] = counts['baseline_correct']/counts['labels'] if counts['labels'] else 0.
    result['priority_accuracy'] = counts['priority_correct']/counts['labels'] if counts['labels'] else 0.
    result['proof_precision'] = counts['proof_correct']/counts['stable_proofs'] if counts['stable_proofs'] else None
    return dict(metrics=result, changes=cases, policy=asdict(policy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contacts', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--policy', choices=['all', 'stable', 'strict', 'relaxed'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.contacts.read_text().splitlines()]
    chosen = POLICIES if args.policy == 'all' else {args.policy: POLICIES[args.policy]}
    report = {name: evaluate(rows, args.audio, policy, args.split) for name, policy in chosen.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v['metrics'] for k, v in report.items()}, indent=2))


if __name__ == '__main__':
    main()
