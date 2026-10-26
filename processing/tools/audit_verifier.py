"""Cache verifier scores once; compare missed-note causes and decoding choices.

These are development/regression diagnostics on previously inspected players,
not new untouched holdout claims. No training data or checkpoint is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from processing.music import verify
from processing.tools import guitarset
from processing.tools.eval_verifier import score_notes, COLUMNS
from processing.training import verifier_data


def cache_scores(model_path, candidates_path, output, domain, players, device):
    from basic_pitch.note_creation import model_frames_to_time
    candidates = verifier_data.load_candidates(candidates_path)
    members = {i for i, (name, player) in enumerate(zip(candidates['names'],
                                                       candidates['players']))
               if player in players and verifier_data.domain_of(name) == domain}
    if not members:
        raise ValueError(f'No excerpts for {domain}/{players}')
    model, channels, cqt = verify.load(model_path, device)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model=str(model_path),
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    candidates=str(candidates_path), domain=domain, players=players,
                    names=[str(candidates['names'][i]) for i in sorted(members)])
    metadata['candidates_sha256'] = hashlib.sha256(candidates_path.read_bytes()).hexdigest()
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    found = verifier_data.excerpts()
    for count, index in enumerate(sorted(members)):
        name = str(candidates['names'][index])
        path = output / f'{Path(name).name}.npz'
        with np.load(verifier_data.CACHE / f'{name}.npz') as stored:
            note = stored['note'].astype(np.float32)
            onset = stored['onset'].astype(np.float32)
            contour = stored['contour'].astype(np.float32)
        times = model_frames_to_time(contour.shape[0])
        plane = Path('data/verifier/planes') / f'{name}.npy'
        if plane.is_file():
            prepared = verifier_data.unstack(np.load(plane, mmap_mode='r'))
        else:
            spectrum = None
            if cqt:
                with np.load(Path(cqt) / f'{name}.npz') as stored:
                    spectrum = verify.align_cqt(stored['cqt'].astype(np.float32),
                                                times[:len(note)], note.shape[1])
            prepared = verify.prepare(note, onset, contour, spectrum)
        rows = np.flatnonzero(candidates['excerpt'] == index)
        frame = candidates['frame'][rows].astype(np.int64)
        pitch = candidates['pitch'][rows].astype(np.int64)
        scores, deltas, strings = verify.score(model, prepared, frame, pitch,
                                               channels, device=device, batch=256)
        reference = found[Path(name).name].notes()
        np.savez_compressed(path, note=note, onset=onset, times=times,
                            frame=frame, pitch=pitch, scores=scores,
                            deltas=deltas, strings=strings,
                            reference=np.array([[n.start, n.end, n.pitch, n.string]
                                                for n in reference]))
        print(f'{count+1}/{len(members)} {name}: {len(rows)} candidates', flush=True)


def rates(counts):
    matched, estimated, reference, fused = counts[:4]
    return dict(note_f1=2*matched/(estimated+reference) if estimated+reference else 0.,
                precision=matched/estimated if estimated else 0.,
                recall=matched/reference if reference else 0.,
                string=fused/matched if matched else 0.,
                tab_f1=2*fused/(estimated+reference) if estimated+reference else 0.,
                counts=dict(zip(COLUMNS, counts.tolist())))


def evaluate(directory, thresholds, suppressions):
    import mir_eval
    from processing.tools.tune_pitch import arrays
    variants = {(threshold, suppression): np.zeros(len(COLUMNS), np.int64)
                for threshold in thresholds for suppression in suppressions}
    coverage, reference_count = 0, 0
    for path in sorted(directory.glob('*.npz')):
        with np.load(path) as s:
            reference = [guitarset.Note(a, b, p, int(string)) for a, b, p, string in s['reference']]
            # One-to-one matching, so repeated notes cannot share one proposal.
            r_i, r_p = arrays([(n.start, n.end, n.pitch) for n in reference])
            starts = s['times'][s['frame']]
            e_i, e_p = arrays([(a, a + .01, p + verify.LOWEST_BIN)
                              for a, p in zip(starts, s['pitch'])])
            coverage += len(mir_eval.transcription.match_notes(
                r_i, r_p, e_i, e_p, onset_tolerance=.05, pitch_tolerance=50.,
                offset_ratio=None))  # type: ignore[arg-type]
            reference_count += len(reference)
            for (threshold, suppression), total in variants.items():
                decoded = verify.decode(s['note'], s['onset'], s['times'], s['scores'],
                                        s['deltas'], s['strings'], s['frame'], s['pitch'],
                                        threshold=threshold, suppression_seconds=suppression)
                total += score_notes([(a, b, p, q) for a, b, p, _, q in decoded], reference)
    if not reference_count:
        raise ValueError(f'No cached reference notes in {directory}')
    return dict(proposal_matches=coverage, reference_notes=reference_count,
                proposal_coverage=coverage/reference_count,
                variants={f'threshold={t},suppression={w}': rates(counts)
                          for (t, w), counts in variants.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('runs/verifier/v8-gt/best.pt'))
    parser.add_argument('--candidates', type=Path, default=Path('data/verifier/candidates-gt2.npz'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--domain', default='mic')
    parser.add_argument('--players', nargs='+', default=['04'])
    parser.add_argument('--device', default='mps')
    parser.add_argument('--thresholds', default='.5,.65,.75,.85')
    parser.add_argument('--suppression', default='.025,.05')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.extract:
        cache_scores(args.model, args.candidates, args.cache, args.domain,
                     args.players, args.device)
    report = evaluate(args.cache, [float(v) for v in args.thresholds.split(',')],
                      [float(v) for v in args.suppression.split(',')])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
