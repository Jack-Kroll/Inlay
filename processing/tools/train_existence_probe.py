"""Test a frozen-feature verifier decision head on train players only.

Reuses positive embeddings from the prior string probe, extracts random and
hard negative embeddings, and selects a conservative head blend on development
recordings. Timing, strings, features and BatchNorm state are frozen.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from processing.music import verify
from processing.training import verifier_data as vd
from processing.tools.train_string_probe import hidden, prepare
from processing.tools.audit_verifier import rates
from processing.tools.eval_verifier import score_notes, COLUMNS
from processing.tools.guitarset import Note
from processing.tools.replay_fusion import fingerprint


def training_features(model, channels, cqt, candidates, positive_cache, output, device):
    metadata = json.loads((positive_cache.parent/'protocol.json').read_text())
    if metadata['model_sha256'] != fingerprint(Path('runs/verifier/v8-gt/best.pt')):
        raise ValueError('Positive feature model fingerprint mismatch')
    if metadata['candidates_sha256'] != fingerprint(Path('data/verifier/candidates-gt2.npz')):
        raise ValueError('Positive candidate fingerprint mismatch')
    with np.load(positive_cache) as s:
        use = np.isin(s['domains'], [0, 2])  # mic and gt-amp, exactly as prior protocol.
        positive_x = s['features'][use]
        positive_rows = s['candidate_rows'][use]
    rng = np.random.default_rng(20261019)
    selected = []
    for domain in ('mic', 'gt-amp'):
        members = [i for i, (name, player) in enumerate(zip(candidates['names'], candidates['players']))
                   if vd.domain_of(name) == domain and player in vd.TRAIN_PLAYERS]
        rows = np.flatnonzero((candidates['label'] == 0) & np.isin(candidates['excerpt'], members))
        hard = rows[(candidates['onset_value'][rows] >= .5) | (candidates['note_value'][rows] >= .4)]
        random_rows = rng.choice(rows, min(32000, len(rows)), replace=False)
        hard = np.setdiff1d(hard, random_rows)
        hard_rows = rng.choice(hard, min(32000, len(hard)), replace=False)
        selected.extend(np.concatenate([random_rows, hard_rows]).tolist())
    selected = np.array(selected, np.int64)
    selected = selected[np.argsort(candidates['excerpt'][selected], kind='stable')]
    x = np.empty((len(selected), 128), np.float32)
    excerpts = candidates['excerpt'][selected]
    unique, first = np.unique(excerpts, return_index=True)
    for count, (index, start, end) in enumerate(zip(unique, first, [*first[1:], len(selected)])):
        rows = selected[start:end]
        x[start:end] = hidden(model, prepare(str(candidates['names'][index]), cqt),
                              candidates['frame'][rows].astype(np.int64),
                              candidates['pitch'][rows].astype(np.int64), channels, device)
        if count % 100 == 0:
            print(f'Negative features: {count+1}/{len(unique)} excerpts', flush=True)
    features = np.concatenate([positive_x, x])
    labels = np.concatenate([np.ones(len(positive_x), np.float32), np.zeros(len(x), np.float32)])
    np.savez_compressed(output, features=features, labels=labels,
                        candidate_rows=np.concatenate([positive_rows, selected]))
    return torch.from_numpy(features), torch.from_numpy(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--positive-features', type=Path, default=Path(
        'runs/accuracy-2026-10-19/string-probe/training-features.npz'))
    parser.add_argument('--development-root', type=Path, default=Path('runs/accuracy-2026-10-19'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261019)
    model_path = Path('runs/verifier/v8-gt/best.pt')
    candidates_path = Path('data/verifier/candidates-gt2.npz')
    model, channels, cqt = verify.load(model_path, args.device)
    candidate_data = vd.load_candidates(candidates_path)
    x, y = training_features(model, channels, cqt, candidate_data,
                             args.positive_features,
                             args.output/'training-features.npz', args.device)
    del candidate_data
    original = nn.Linear(128, 1)
    original.load_state_dict({k: v.cpu() for k, v in model.exist.state_dict().items()})
    learned = nn.Linear(128, 1)
    learned.load_state_dict(original.state_dict())
    optimizer = torch.optim.AdamW(learned.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator().manual_seed(20261019)
    for epoch in range(10):
        for batch in torch.randperm(len(y), generator=generator).split(1024):
            loss = F.binary_cross_entropy_with_logits(learned(x[batch]).squeeze(1), y[batch])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    blends = {}
    for fraction in (0., .25, .5, 1.):
        head = nn.Linear(128, 1)
        head.load_state_dict({key: (1-fraction)*value + fraction*learned.state_dict()[key]
                              for key, value in original.state_dict().items()})
        blends[fraction] = head
    reports = {str(f): {} for f in blends}
    for domain in ('gs04-scores', 'gt-amp-scores'):
        root = args.development_root/domain
        metadata = json.loads((root/'metadata.json').read_text())
        names = {Path(name).name: name for name in metadata['names']}
        totals = {fraction: np.zeros(len(COLUMNS), np.int64) for fraction in blends}
        for path in sorted(root.glob('*.npz')):
            with np.load(path) as s:
                data = {key: s[key] for key in s.files}
            features = torch.from_numpy(hidden(model, prepare(names[path.stem], cqt),
                                                data['frame'], data['pitch'], channels, args.device))
            reference = [Note(a, b, p, int(string)) for a, b, p, string in data['reference']]
            with torch.inference_mode():
                for fraction, head in blends.items():
                    scores = (data['scores'] if not fraction else
                              head(features).squeeze(1).sigmoid().numpy())
                    decoded = verify.decode(data['note'], data['onset'], data['times'], scores,
                                            data['deltas'], data['strings'], data['frame'], data['pitch'],
                                            suppression_mode='raw')
                    totals[fraction] += score_notes([(a, b, p, q) for a, b, p, _, q in decoded], reference)
        for fraction, total in totals.items():
            reports[str(fraction)][domain] = rates(total)
        print(domain, json.dumps({str(f): rates(t) for f, t in totals.items()}), flush=True)
    selected = max(blends, key=lambda f: np.mean([r['note_f1'] for r in reports[str(f)].values()]))
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    for key, value in blends[selected].state_dict().items():
        checkpoint['model']['exist.'+key] = value
    checkpoint['experiment'] = f'Frozen existence probe blend {selected}; development only'
    torch.save(checkpoint, args.output/'probe.pt')
    (args.output/'results.json').write_text(json.dumps(dict(
        model=str(model_path), model_sha256=fingerprint(model_path),
        candidates_sha256=fingerprint(candidates_path), samples=len(y), positives=int(y.sum()),
        seed=20261019, epochs=10, selected=selected,
        suppression_mode='raw',
        selection='Mean note F1 on GS04 and P3 amp; no gradient from evaluation players',
        variants=reports), indent=2))


if __name__ == '__main__':
    main()
