"""Research-only string-head calibration with the complete note detector frozen.

Samples true candidate labels from training players only. Evaluation reuses
cached development detections, so existence/timing and note counts cannot move.
No checkpoint is installed as the runtime default.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from processing.music import verify
from processing.training import verifier_data as vd
from processing.tools.audit_verifier import rates
from processing.tools.eval_verifier import score_notes, COLUMNS
from processing.tools.guitarset import Note

DOMAINS = ('mic', 'gt-di', 'gt-amp', 'gt-ego', 'gt-exo')


def hidden(model, prepared, frames, pitches, channels, device):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(frames), 256):
            take = slice(start, start + 256)
            x = torch.from_numpy(verify.cut(prepared, frames[take], pitches[take], channels)).to(device)
            p = torch.from_numpy(pitches[take].astype(np.float32)).to(device)
            features = model.pool(model.stem(x)).flatten(1)
            extra = torch.stack([(p - 50.)/25., ((p - 50.)/25.)**2], dim=1)
            outputs.append(model.trunk(torch.cat([features, extra], dim=1)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, 128), np.float32)


def prepare(name, cqt):
    from basic_pitch.note_creation import model_frames_to_time
    plane = Path('data/verifier/planes') / f'{name}.npy'
    if plane.is_file():
        return vd.unstack(np.load(plane, mmap_mode='r'))
    with np.load(vd.CACHE / f'{name}.npz') as s:
        note, onset, contour = s['note'], s['onset'], s['contour']
    spectrum = None
    if cqt:
        with np.load(Path(cqt) / f'{name}.npz') as s:
            spectrum = verify.align_cqt(s['cqt'].astype(np.float32),
                                        model_frames_to_time(len(contour))[:len(note)], note.shape[1])
    return verify.prepare(note, onset, contour, spectrum)


def extract_training(model, channels, cqt, candidates, output, device):
    rng = np.random.default_rng(42)
    positive = np.flatnonzero((candidates['label'] == 1) & (candidates['string'] >= 0))
    selected = []
    for domain in DOMAINS:
        members = [i for i, (name, player) in enumerate(zip(candidates['names'],
                                                           candidates['players']))
                   if vd.domain_of(name) == domain and player in vd.TRAIN_PLAYERS]
        rows = positive[np.isin(candidates['excerpt'][positive], members)]
        selected.extend(rng.choice(rows, min(10000, len(rows)), replace=False).tolist())
    selected = np.array(selected, np.int64)
    selected = selected[np.argsort(candidates['excerpt'][selected], kind='stable')]
    features = np.empty((len(selected), 128), np.float32)
    domains = np.empty(len(selected), np.int64)
    excerpts = candidates['excerpt'][selected]
    unique, first = np.unique(excerpts, return_index=True)
    for count, (index, start, end) in enumerate(zip(unique, first, [*first[1:], len(selected)])):
        name = str(candidates['names'][index])
        rows = selected[start:end]
        features[start:end] = hidden(model, prepare(name, cqt),
                                     candidates['frame'][rows].astype(np.int64),
                                     candidates['pitch'][rows].astype(np.int64),
                                     channels, device)
        domains[start:end] = DOMAINS.index(vd.domain_of(name))
        if count % 100 == 0:
            print(f'Training features: {count+1}/{len(unique)} excerpts', flush=True)
    np.savez_compressed(output, features=features, labels=candidates['string'][selected],
                        domains=domains, candidate_rows=selected)


def evaluation_features(model, channels, cqt, directory, device):
    metadata = json.loads((directory / 'metadata.json').read_text())
    by_name = {Path(name).name: name for name in metadata['names']}
    result = []
    for path in sorted(directory.glob('*.npz')):
        with np.load(path) as stored:
            s = {key: stored[key] for key in stored.files}
        eligible = np.flatnonzero(s['scores'] >= verify.SCORE_THRESHOLD)
        s['eligible'] = eligible
        s['features'] = hidden(model, prepare(by_name[path.stem], cqt),
                               s['frame'][eligible], s['pitch'][eligible], channels, device)
        result.append(s)
    return result


def evaluate(head, excerpts):
    totals = np.zeros(len(COLUMNS), np.int64)
    with torch.inference_mode():
        for s in excerpts:
            strings = s['strings'].copy()
            strings[s['eligible']] = head(torch.from_numpy(s['features'])).softmax(1).numpy()[:, ::-1]
            decoded = verify.decode(s['note'], s['onset'], s['times'], s['scores'],
                                    s['deltas'], strings, s['frame'], s['pitch'])
            reference = [Note(a, b, p, int(string)) for a, b, p, string in s['reference']]
            totals += score_notes([(a, b, p, q) for a, b, p, _, q in decoded], reference)
    return rates(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('runs/verifier/v8-gt/best.pt'))
    parser.add_argument('--candidates', type=Path, default=Path('data/verifier/candidates-gt2.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eval-caches', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    model, channels, cqt = verify.load(args.model, args.device)
    features_path = args.output / 'training-features.npz'
    # Always regenerate: no accidental reuse after changing the base model.
    candidates = vd.load_candidates(args.candidates)
    extract_training(model, channels, cqt, candidates, features_path, args.device)
    del candidates
    evaluated = {p.name: evaluation_features(model, channels, cqt, p, args.device)
                 for p in args.eval_caches}
    original = nn.Linear(128, 6)
    original.load_state_dict({k: v.cpu() for k, v in model.string.state_dict().items()})
    results = {'baseline': {name: evaluate(original, data) for name, data in evaluated.items()}}
    del model
    with np.load(features_path) as stored:
        x = torch.from_numpy(stored['features'])
        y = torch.from_numpy(stored['labels'].astype(np.int64))
        domains = torch.from_numpy(stored['domains'])
    best_score, selected, selected_state = -1., '', None
    for variant in ('equal_domains', 'room_quarter_weight'):
        head = nn.Linear(128, 6)
        head.load_state_dict(original.state_dict())
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        generator = torch.Generator().manual_seed(42)
        for epoch in range(20):
            for batch in torch.randperm(len(y), generator=generator).split(1024):
                loss = F.cross_entropy(head(x[batch]), y[batch], reduction='none')
                weights = (torch.where(domains[batch] >= 3, .25, 1.)
                           if variant == 'room_quarter_weight' else torch.ones(len(batch)))
                loss = (loss*weights).sum()/weights.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        results[variant] = {name: evaluate(head, data) for name, data in evaluated.items()}
        torch.save(head.state_dict(), args.output / f'{variant}-head.pt')
        metric = float(np.mean([r['tab_f1'] for r in results[variant].values()]))
        if metric > best_score:
            best_score, selected = metric, variant
            selected_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(variant, json.dumps(results[variant]), flush=True)
    assert selected_state is not None
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    for key, value in selected_state.items():
        checkpoint['model']['string.' + key] = value
    checkpoint['experiment'] = f'Frozen string probe: {selected}; not a runtime default'
    torch.save(checkpoint, args.output / 'probe.pt')
    (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    (args.output / 'protocol.json').write_text(json.dumps(dict(
        base_model=str(args.model), seed=42, samples=len(y), epochs=20, domains=DOMAINS,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        candidates_sha256=hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        train_players=vd.TRAIN_PLAYERS, evaluation=[str(p) for p in args.eval_caches],
        selected=selected, selection='Mean development tab F1 across named evaluation caches',
        caveat='Development comparison only. Entire feature extractor, note and timing heads frozen.'
    ), indent=2))


if __name__ == '__main__':
    main()
