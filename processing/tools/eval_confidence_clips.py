"""Replay visual priority and conservative note recovery on cached user clips.

No reference tabs exist here. Report exact changes and support, not accuracy.
The original string orientation is held fixed to isolate the fusion changes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

from processing.music.fusion import Sounded
from processing.music import verify
from processing.music.transcribe import transcribe_notes, fingered_rate
from processing.music.visual_evidence import VisualContext, VisualPolicy
from processing.tools.diagnose_missed_notes import recover_scores
from processing.tools.replay_fusion import read_records


def notes_from_cache(s, scores, suppression_mode='raw'):
    return [Sounded(p, a, b, confidence, q) for a, b, p, confidence, q in verify.decode(
        s['note'], s['onset'], s['times'], s['scores'], s['deltas'], s['strings'], s['frame'], s['pitch'],
        accepted=scores >= verify.SCORE_THRESHOLD, suppression_mode=suppression_mode)]


def supported_recovery(placement):
    proof = placement.evidence.get('visual_priority')
    return (proof is not None and placement.support == 'fingered'
            and (placement.string, placement.fret) == (proof['string'], proof['fret']))


def evaluate(root, audio_root=None):
    report = {}
    for path in sorted(root.glob('*-replay.json')):
        slug = path.name.removesuffix('-replay.json')
        source_path = Path(json.loads(path.read_text())['source_notes'])
        source = [json.loads(line) for line in source_path.read_text().splitlines()]
        summary = source[0]
        records = read_records(root/f'{slug}-vision.json')
        policies = dict(stable=VisualPolicy(),
                         relaxed=replace(VisualPolicy(), min_contact=.5, min_margin=.06,
                                         max_fret_motion=.25, max_string_motion=.5),
                         strict=replace(VisualPolicy(), min_contact=.85,
                         min_margin=.12, max_fret_motion=.25, max_string_motion=.25,
                         max_residual=.04, max_edge_residual=.005))
        base_notes = [Sounded(r['pitch'], r['startSeconds'], r['startSeconds']+r['durationSeconds'],
                             r['amplitude'], np.array(r['evidence']['audio_string'])
                             if 'audio_string' in r.get('evidence', {}) else None)
                      for r in source if r['kind'] == 'note']
        variants: dict[str, tuple[list[Sounded], VisualPolicy | None]] = {'baseline': (base_notes, None)}
        if audio_root is not None:
            with np.load(audio_root/f"{Path(summary['video']).stem}.npz") as stored:
                s = {key: stored[key] for key in stored.files}
            base_notes = notes_from_cache(s, s['scores'])
            variants['baseline'] = (base_notes, None)
            attack_scores = recover_scores(s, 'strong_attack')
            variants['audio_strong_attack'] = (notes_from_cache(s, attack_scores), None)
            variants['deduplicated'] = (notes_from_cache(s, s['scores'], 'both'), None)
            variants['dedup_strong_attack'] = (notes_from_cache(s, attack_scores, 'both'), None)
            variants['dedup_visual_priority'] = (notes_from_cache(s, s['scores'], 'both'), policies['stable'])
            context = VisualContext.from_records(records, len(summary['tuning']),
                summary['string_inset'], summary['string_order'] == 'flipped', summary['max_fret'])
            # Visual confirmation is tested with a more permissive local audio
            # requirement than the audio-only rule, without generating pitches.
            rescued = np.flatnonzero((s['scores'] >= .5) & (s['scores'] < verify.SCORE_THRESHOLD)
                        & (s['onset'][s['frame'], s['pitch']] >= .3)
                        & (s['note'][s['frame'], s['pitch']] >= .3))
            visual_scores = s['scores'].copy()
            for i in rescued:
                timestamp = max(0., float(s['times'][s['frame'][i]] + s['deltas'][i]/verify.FPS))
                if context.placement(timestamp, int(s['pitch'][i])+verify.LOWEST_BIN,
                                     summary['tuning'], summary['max_fret']) is not None:
                    visual_scores[i] = verify.SCORE_THRESHOLD
            variants['visual_confirmed_recovery'] = (notes_from_cache(s, visual_scores), policies['stable'])
        for name, policy in policies.items():
            variants[name] = (base_notes, policy)
        results, baseline = {}, None
        for name, (notes, policy) in variants.items():
            assigned, groups = transcribe_notes(
                records, notes, summary['tuning'], summary['max_fret'], summary['string_inset'],
                summary['string_order'] == 'flipped', audio_weight=summary['audio_string_weight'],
                visual_priority=policy is not None, visual_policy=policy)
            rejected = []
            if name == 'visual_confirmed_recovery':
                bad = {(n.pitch, round(n.start, 7)) for n in assigned.values()
                       if baseline is not None and (n.pitch, round(n.start, 7)) not in baseline
                       and not supported_recovery(n)}
                rejected = [dict(pitch=pitch, start=start, reason='final_assignment_lost_visual_support')
                            for pitch, start in sorted(bad)]
                if bad:
                    notes = [n for n in notes if (n.pitch, round(n.start, 7)) not in bad]
                    assigned, groups = transcribe_notes(
                        records, notes, summary['tuning'], summary['max_fret'], summary['string_inset'],
                        summary['string_order'] == 'flipped', audio_weight=summary['audio_string_weight'],
                        visual_priority=True, visual_policy=policy)
            tab = [asdict(n) for n in assigned.values()]
            keyed = {(n['pitch'], round(n['start'], 7)): n for n in tab}
            if baseline is None:
                baseline = keyed
            changes, added = [], []
            for key, n in keyed.items():
                before = baseline.get(key)
                if before is None:
                    added.append(n)
                elif (before['string'], before['fret']) != (n['string'], n['fret']):
                    changes.append(dict(pitch=n['pitch'], start=n['start'],
                                        before=[before['string'], before['fret']],
                                        after=[n['string'], n['fret']], evidence=n['evidence']))
            results[name] = dict(notes=len(tab), open_notes=sum(n['fret'] == 0 for n in tab),
                                 fingertip_support=fingered_rate(groups),
                                 priority_applied=sum('visual_priority' in n['evidence'] for n in tab),
                                 added=added, changed=changes, rejected_rescues=rejected)
        report[slug] = results
        print(slug, {k: {a: b for a, b in v.items() if a not in ('added', 'changed')}
                     | {'added': len(v['added']), 'changed': len(v['changed'])}
                     for k, v in results.items()}, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.root, args.audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
