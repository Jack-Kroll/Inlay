"""Train the note verifier on GuitarSet, holding whole players out.

    uv run python -m processing.tools.cqt
    uv run python -m processing.training.verifier_data
    uv run python -m processing.training.train_verifier \
      --cqt data/guitarset/cqt --output runs/verifier/v2-cqt --epochs 14

Three heads share one trunk: whether a note is attacking, how far the true
onset is from the candidate frame, and which string played it. The string head
is the reason the tab stage improves as much as the note list does -- it reads
timbre, which no amount of musical context can recover.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional

from processing.music import verify
from processing.training import verifier_data


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_sample(candidates, members, ratio, rng):
    """All positives from the split, plus ``ratio`` negatives each.

    Negatives are resampled every epoch, so over a run the model sees far more
    of the 25:1 majority than any one epoch holds. ``ratio`` of 0 trains on
    real notes alone, which is what the string task alone would want.
    """
    belongs = np.isin(candidates['excerpt'], list(members))
    positive = np.flatnonzero(belongs & (candidates['label'] == 1))
    if ratio <= 0:
        chosen = positive.copy()
        rng.shuffle(chosen)
        return chosen
    negative = np.flatnonzero(belongs & (candidates['label'] == 0))
    take = min(len(negative), int(ratio*len(positive)))
    chosen = np.concatenate([positive, rng.choice(negative, take, replace=False)])
    rng.shuffle(chosen)
    return chosen


def batches(candidates, patches, rows, size, device):
    excerpt = candidates['excerpt'][rows].astype(np.int64)
    frame = candidates['frame'][rows].astype(np.int64)
    pitch = candidates['pitch'][rows].astype(np.int64)
    label = candidates['label'][rows].astype(np.float32)
    delta = candidates['delta'][rows]*verify.FPS
    string = candidates['string'][rows].astype(np.int64)
    for start in range(0, len(rows), size):
        stop = start + size
        patch = patches.batch(excerpt[start:stop], frame[start:stop], pitch[start:stop])
        yield (torch.from_numpy(patch).to(device),
               torch.from_numpy(pitch[start:stop].astype(np.float32)).to(device),
               torch.from_numpy(label[start:stop]).to(device),
               torch.from_numpy(delta[start:stop]).to(device),
               torch.from_numpy(string[start:stop]).to(device))


@torch.no_grad()
def score_split(model, candidates, patches, members, device, size=512, sample=None,
                seed=1):
    """Average precision over a split's candidates, plus string and onset error."""
    model.eval()
    rows = np.flatnonzero(np.isin(candidates['excerpt'], list(members)))
    if sample and sample < len(rows):
        # Per-epoch monitoring only. Subsampling candidates uniformly leaves
        # average precision unbiased, and the final report uses all of them.
        rows = np.sort(np.random.default_rng(seed).choice(rows, sample, replace=False))
    scores = np.empty(len(rows), np.float32)
    deltas = np.empty(len(rows), np.float32)
    strings = np.empty(len(rows), np.int64)
    at = 0
    for patch, pitch, *_ in batches(candidates, patches, rows, size, device):
        exist, delta, string = model(patch, pitch)
        count = len(pitch)
        scores[at:at + count] = torch.sigmoid(exist).float().cpu().numpy()
        deltas[at:at + count] = delta.float().cpu().numpy()
        strings[at:at + count] = string.argmax(1).cpu().numpy()
        at += count
    label = candidates['label'][rows]
    from sklearn.metrics import average_precision_score
    positive = label == 1
    truth = candidates['string'][rows][positive]
    reference = candidates['delta'][rows][positive]*verify.FPS
    return dict(ap=float(average_precision_score(label, scores)),
                string=float((strings[positive] == truth).mean()),
                delta_mae=float(np.abs(deltas[positive] - reference).mean()),
                delta_baseline=float(np.abs(reference).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidates', type=Path,
                        default=Path('data/verifier/candidates.npz'))
    parser.add_argument('--output', type=Path, default=Path('runs/verifier/v1'))
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--ratio', type=float, default=4.,
                        help='Negatives per positive each epoch')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--string-weight', type=float, default=.3)
    parser.add_argument('--delta-weight', type=float, default=.1)
    parser.add_argument('--cqt', type=Path,
                        help='CQT cache to stack on as extra patch channels')
    parser.add_argument('--select', choices=['ap', 'string'], default='ap',
                        help='Which validation metric picks best.pt')
    parser.add_argument('--val-sample', type=int, default=80000,
                        help='Candidates to score each epoch; 0 uses all of them')
    parser.add_argument('--device')
    parser.add_argument('--max-steps', type=int, help='Smoke test: stop each epoch early')
    args = parser.parse_args()

    device = pick_device(args.device)
    candidates = verifier_data.load_candidates(args.candidates)
    train, val, test = verifier_data.splits(candidates['players'])
    print(f'device {device}; excerpts train {len(train)} val {len(val)} test {len(test)}')
    print('loading posteriorgrams into memory...', flush=True)
    patches = verifier_data.Patches(candidates['names'], progress=True, cqt=args.cqt)

    model = verify.build(args.width, patches.channels).to(device)
    print(f'verifier: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.

    steps = len(epoch_sample(candidates, train, args.ratio, np.random.default_rng(0)))
    steps = steps//args.batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr,
        total_steps=max(1, args.epochs*(args.max_steps or steps)))

    for number in range(1, args.epochs + 1):
        model.train()
        rows = epoch_sample(candidates, train, args.ratio, rng)
        started, seen, total = time.time(), 0, 0.
        for step, (patch, pitch, label, delta, string) in enumerate(
                batches(candidates, patches, rows, args.batch, device)):
            exist, predicted_delta, predicted_string = model(patch, pitch)
            real = label > .5
            loss = (functional.binary_cross_entropy_with_logits(exist, label)
                    if args.ratio > 0 else exist.sum()*0.)
            if real.any():
                # Onset offset and string are only defined for real notes.
                loss = loss + args.delta_weight*functional.smooth_l1_loss(
                    predicted_delta[real], delta[real])
                loss = loss + args.string_weight*functional.cross_entropy(
                    predicted_string[real], string[real])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimiser.step()
            if schedule.last_epoch < schedule.total_steps - 1:
                schedule.step()
            total += float(loss.detach())*len(pitch); seen += len(pitch)
            if args.max_steps and step + 1 >= args.max_steps:
                break
        metrics = score_split(model, candidates, patches, val, device,
                              sample=args.val_sample)
        print(f'epoch {number:>2}  loss {total/max(1, seen):.4f}  '
              f'val AP {metrics["ap"]:.4f}  string {metrics["string"]:.3f}  '
              f'onset MAE {metrics["delta_mae"]:.2f} frames '
              f'(raw {metrics["delta_baseline"]:.2f})  '
              f'{time.time() - started:.0f}s', flush=True)
        history.append(dict(epoch=number, loss=total/max(1, seen), **metrics))
        if metrics[args.select] > best:
            best = metrics[args.select]
            torch.save({'model': model.state_dict(), 'width': args.width,
                        'channels': patches.channels, 'cqt': str(args.cqt or ''),
                        'epoch': number, 'metrics': metrics}, args.output/'best.pt')
    (args.output/'history.json').write_text(json.dumps(history, indent=2))
    print(f'\nbest val {args.select} {best:.4f}; wrote {args.output}/best.pt')


if __name__ == '__main__':
    main()
