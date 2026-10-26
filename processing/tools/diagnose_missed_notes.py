"""Partition final misses and test fixed recovery rules on cached verifier outputs.

Development diagnostics only. References are used for scoring and diagnosis,
never for the recovery rules. Counts match one-to-one by pitch/onset.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
import numpy as np
import mir_eval

from processing.music import verify
from processing.tools.audit_verifier import rates
from processing.tools.eval_verifier import score_notes, COLUMNS
from processing.tools.guitarset import Note
from processing.tools.tune_pitch import arrays


def matches(reference, starts, pitches):
    if not reference or not len(starts):
        return []
    ri, rp = arrays([(n.start, n.end, n.pitch) for n in reference])
    ei, ep = arrays([(float(t), float(t) + .01, int(p)) for t, p in zip(starts, pitches)])
    return mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=.05,
                                             pitch_tolerance=50., offset_ratio=None)  # type: ignore[arg-type]


def recover_scores(s, rule):
    scores = s['scores'].copy()
    frame, pitch = s['frame'], s['pitch']
    onset, note = s['onset'][frame, pitch], s['note'][frame, pitch]
    if rule in ('strong_attack', 'post_refined_strong_attack'):
        recover = verify.strong_attack_candidates(s['note'], s['onset'], frame, pitch, scores)
    elif rule == 'strong_sustain':
        recover = (scores >= .5) & (onset >= .4) & (note >= .85)
    elif rule == 'near_threshold':
        recover = (scores >= .65) & (onset >= .6) & (note >= .6)
    else:
        recover = np.zeros(len(scores), bool)
    scores[recover] = np.maximum(scores[recover], verify.SCORE_THRESHOLD)
    return scores


def diagnose(directory) -> dict[str, Any]:
    rules = ('baseline', 'no_refinement', 'delta_clipped_2', 'strong_attack',
             'strong_sustain', 'near_threshold', 'refined_suppression', 'post_refined_suppression')
    rules += ('post_refined_strong_attack',)
    totals = {key: np.zeros(len(COLUMNS), np.int64) for key in rules}
    stages, misses, false_positives = Counter(), Counter(), Counter()
    details = []
    for path in sorted(directory.glob('*.npz')):
        with np.load(path) as data:
            s = {key: data[key] for key in data.files}
        reference = [Note(a, b, p, int(string)) for a, b, p, string in s['reference']]
        frame, pitch, score = s['frame'], s['pitch'], s['scores']
        raw = s['times'][frame]
        midi = pitch + verify.LOWEST_BIN
        above = np.flatnonzero(score >= verify.SCORE_THRESHOLD)
        kept = above[verify.suppress(frame[above], pitch[above], score[above],
                                     round(verify.ONSET_TOLERANCE*verify.FPS))]
        refined = np.maximum(0., raw + s['deltas']/verify.FPS)
        for name, indices, times in [('raw_candidates', np.arange(len(frame)), raw),
                                     ('above_threshold_raw', above, raw),
                                     ('after_suppression_raw', kept, raw),
                                     ('after_refinement', kept, refined)]:
            stages[name] += len(matches(reference, times[indices], midi[indices]))
        stages['reference'] += len(reference)
        decoded = None
        for rule in rules:
            delta = np.clip(s['deltas'], -2., 2.) if rule == 'delta_clipped_2' else s['deltas']
            eligible = recover_scores(s, rule) >= verify.SCORE_THRESHOLD
            out = verify.decode(s['note'], s['onset'], s['times'], s['scores'],
                                delta, s['strings'], frame, pitch,
                                refine=rule != 'no_refinement', accepted=eligible,
                                suppression_mode=('refined' if rule == 'refined_suppression' else
                                                  'both' if rule in ('post_refined_suppression',
                                                                    'post_refined_strong_attack') else 'raw'))
            totals[rule] += score_notes([(a, b, p, q) for a, b, p, _, q in out], reference)
            if rule == 'baseline':
                decoded = out
        assert decoded is not None
        pairs = matches(reference, [n[0] for n in decoded], [n[2] for n in decoded])
        matched_ref = {a for a, _ in pairs}
        matched_est = {b for _, b in pairs}
        for index, n in enumerate(reference):
            if index in matched_ref:
                continue
            same = np.abs(midi - n.pitch) <= .5
            near = np.flatnonzero(same & (np.abs(raw - n.start) <= .05))
            eligible = near[score[near] >= verify.SCORE_THRESHOLD]
            survived = np.intersect1d(eligible, kept)
            if not len(near):
                cause = 'no_raw_candidate_50ms'
            elif not len(eligible):
                cause = 'verifier_below_threshold'
            elif not len(survived):
                cause = 'suppressed'
            elif not np.any(np.abs(refined[survived] - n.start) <= .05):
                cause = 'refinement_outside_tolerance'
            else:
                cause = 'one_to_one_competition'
            misses[cause] += 1
            best = int(near[np.argmax(score[near])]) if len(near) else None
            details.append(dict(excerpt=path.stem, start=n.start, pitch=n.pitch,
                                string=n.string, cause=cause,
                                score=float(score[best]) if best is not None else None,
                                onset=float(s['onset'][frame[best], pitch[best]]) if best is not None else None,
                                note=float(s['note'][frame[best], pitch[best]]) if best is not None else None,
                                delta_ms=float(s['deltas'][best]/verify.FPS*1000) if best is not None else None))
        for index, n in enumerate(decoded):
            if index in matched_est:
                continue
            nearby = [r for r in reference if abs(r.start - n[0]) <= .05]
            difference = [abs(round(r.pitch) - n[2]) for r in nearby]
            false_positives['with_nearby_reference_attack' if nearby else 'without_nearby_reference_attack'] += 1
            if any(d in (12, 19, 24) for d in difference):
                false_positives['octave_or_harmonic_pitch_distance'] += 1
    baseline_misses = int(totals['baseline'][2] - totals['baseline'][0])
    if sum(misses.values()) != baseline_misses:
        raise ValueError('Miss diagnosis does not partition all baseline misses')
    return dict(cache=str(directory), stages=dict(stages), baseline_misses=baseline_misses,
                miss_causes=dict(misses), false_positive_context=dict(false_positives),
                variants={name: rates(counts) for name, counts in totals.items()},
                misses=details, caveat='Stage match totals are not disjoint losses; miss_causes partitions final misses.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'misses'}, indent=2))


if __name__ == '__main__':
    main()
