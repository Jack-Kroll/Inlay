"""Fine-tune the pitch front end on guitar.

    uv run python -m processing.training.frontend_data --prepare
    uv run python -m processing.training.train_frontend --output runs/frontend/ft1

Starts from Basic Pitch's released weights (``processing.tools.convert_basic_pitch``)
and trains every layer after the CQT on the windows ``frontend_data`` draws,
with Basic Pitch's own loss: binary cross-entropy on each head, labels
squeezed .2 towards one half, onsets weighted as the release was
(``POSITIVE_WEIGHT``). Batch-norm statistics stay where the release
left them; see ``BasicPitch.freeze_statistics``.

Where a source has no f0 track the contour head is trained towards the
released model's own contour on the same window instead. That head is where
the verifier reads string cues from -- inharmonicity across its three bins per
semitone -- and labels drawn from MIDI would flatten exactly that.

The checkpoint kept is the one whose onset peaks rank real attacks best,
averaged over the three validation sets: average precision over every peak
of the onset head, the space the verifier actually ranks in. GuitarSet 05 and
IDMT-SMT-GUITAR are never read here. See docs/front-end-finetune-plan.md.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from processing.music import frontend
from processing.training import frontend_data

SMOOTHING = .2
# The weight the released onset head was trained at, measured rather than
# assumed: on GuitarSet's training players its gradients from positive and
# negative cells cancel (to .084, against component norms of 7.3 and .38) at
# .0475. Plain averaging would be .0004 and the package's documented default
# .5; the note and contour heads, checked the same way, sit at their plain
# positive fractions. A fine-tune at either default moves the onset head off
# in a few hundred steps.
POSITIVE_WEIGHT = .05


def smoothed(target, smoothing=SMOOTHING):
    return target*(1 - smoothing) + smoothing/2


def head_loss(logits, target, weight=None):
    """Per-window mean cross-entropy, weighted per window when ``weight`` is given."""
    per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    per_window = per_cell.flatten(1).mean(1)
    if weight is None:
        return per_window.mean()
    return (per_window*weight).sum()/weight.sum().clamp_min(1.)


def weighted_loss(logits, target, weight, positive_weight=POSITIVE_WEIGHT):
    """Basic Pitch's ``weighted_transcription_loss``: the mean over positive
    cells and the mean over negative ones, mixed ``positive_weight`` to the
    rest, across the windows ``weight`` keeps.

    An onset is one cell in fifteen thousand. Averaged plainly, the loss is
    met by predicting quieter attacks -- measured, a run trained that way
    halved the onset head at real notes (.65 to .30) and lost a fifth of its
    peak AP in 600 steps while its held-out loss went down."""
    keep = weight[:, None, None].expand_as(target) > 0
    positive = (target > 0) & keep
    negative = (target == 0) & keep
    per_cell = F.binary_cross_entropy_with_logits(logits, smoothed(target), reduction='none')
    zero = per_cell.sum()*0
    mean = lambda mask: per_cell[mask].mean() if mask.any() else zero
    return (1 - positive_weight)*mean(negative) + positive_weight*mean(positive)


def loss(model, teacher, batch, weighted=True):
    audio = batch['audio']
    heads = model(audio, logits=True)
    contour_weight = batch['contour_weight']
    contour_target = smoothed(batch['contour'])
    if teacher is not None and (contour_weight < 1).any():
        with torch.no_grad():
            taught = teacher(audio)['contour']
        keep = contour_weight[:, None, None]
        contour_target = keep*contour_target + (1 - keep)*taught
    parts = {'note': head_loss(heads['note'], smoothed(batch['note'])),
             'onset': (weighted_loss(heads['onset'], batch['onset'], batch['onset_weight'])
                       if weighted else head_loss(heads['onset'], smoothed(batch['onset']),
                                                  batch['onset_weight'])),
             'contour': head_loss(heads['contour'], contour_target)}
    return (parts['note'] + parts['onset'] + parts['contour'],
            {k: float(v.detach()) for k, v in parts.items()})


def summary(results):
    return '  '.join(f'{name} AP {r["ap"]:.4f} R@.15 {r["recall"]:.3f} '
                     f'cand {r["candidates"]:.1f}' for name, r in results.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('runs/frontend/ft1'))
    parser.add_argument('--init', type=Path, default=frontend.SHIPPED)
    parser.add_argument('--groups', nargs='*', default=list(frontend_data.WEIGHTS),
                        help='Which sources to train on: ' + ' '.join(frontend_data.WEIGHTS))
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--steps', type=int, default=1500, help='Steps per epoch')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--onset-loss', choices=['weighted', 'plain'], default='weighted',
                        help='Onset cells weighted as the release was trained, or a plain mean')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-limit', type=int,
                        help='Smoke test: validate on this many excerpts per set')
    parser.add_argument('--device')
    args = parser.parse_args()

    device = torch.device(args.device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    torch.manual_seed(args.seed)
    items = frontend_data.training_items(tuple(args.groups))
    missing = [item for item in items if not item.audio.is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} items not rendered, e.g. {missing[0].audio}; '
                         'run processing.training.frontend_data --prepare')
    counts = {}
    for item in items:
        counts[item.group] = counts.get(item.group, 0) + 1
    print(f'device {device}; {len(items)} training items: ' +
          ', '.join(f'{n} {g}' for g, n in counts.items()), flush=True)
    sets = frontend_data.validation_items()
    print('validating on ' + ', '.join(f'{name} ({len(found)})' for name, found in sets.items()))

    model = frontend.load(args.init, device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    windows = frontend_data.Windows(items, args.epochs*args.steps*args.batch,
                                    {g: w for g, w in frontend_data.WEIGHTS.items()
                                     if g in args.groups}, seed=args.seed)
    loader = torch.utils.data.DataLoader(windows, batch_size=args.batch, shuffle=False,
                                         num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers else None)

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.time()
    baseline = frontend_data.evaluate(model, sets, device, args.val_limit)
    best = float(np.mean([r['ap'] for r in baseline.values()]))
    print(f'epoch  0  {summary(baseline)}  mean AP {best:.4f}', flush=True)
    history.append({'epoch': 0, 'validation': baseline})
    torch.save({'model': model.state_dict(), 'epoch': 0, 'args': vars(args) | {
        'output': str(args.output), 'init': str(args.init)}}, args.output/'best.pt')

    model.train()
    model.freeze_statistics()
    batches = iter(loader)
    for epoch in range(1, args.epochs + 1):
        totals, tick = {}, time.time()
        for _ in range(args.steps):
            batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
            value, parts = loss(model, teacher, batch, args.onset_loss == 'weighted')
            optimiser.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimiser.step()
            model.renormalize()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.) + v
        model.eval()
        results = frontend_data.evaluate(model, sets, device, args.val_limit)
        model.train()
        model.freeze_statistics()
        mean = float(np.mean([r['ap'] for r in results.values()]))
        losses = {k: v/args.steps for k, v in totals.items()}
        print(f'epoch {epoch:2d}  loss ' + ' '.join(f'{k} {v:.4f}' for k, v in losses.items()) +
              f'  {summary(results)}  mean AP {mean:.4f}  {time.time() - tick:.0f}s', flush=True)
        history.append({'epoch': epoch, 'loss': losses, 'validation': results})
        state = {'model': model.state_dict(), 'epoch': epoch,
                 'args': vars(args) | {'output': str(args.output), 'init': str(args.init)}}
        torch.save(state, args.output/'last.pt')
        if mean > best:
            best = mean
            torch.save(state, args.output/'best.pt')
        (args.output/'history.json').write_text(json.dumps(history, indent=1))
    print(f'\nbest mean AP {best:.4f}; {time.time() - started:.0f}s; wrote {args.output/"best.pt"}')


if __name__ == '__main__':
    main()
