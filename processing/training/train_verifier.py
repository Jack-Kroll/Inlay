"""Train the note verifier on GuitarSet and Guitar-TECHS, holding whole players out.

    uv run python -m processing.tools.cqt
    uv run python -m processing.tools.variants --variants 3
    uv run python -m processing.tools.guitar_techs --prepare
    uv run python -m processing.training.verifier_data \
      --domains aug-v1 aug-v2 aug-v3 --measure-domains gt-di gt-amp gt-ego gt-exo
    uv run python -m processing.training.train_verifier \
      --cqt data/guitarset/cqt --planes data/verifier/planes \
      --output runs/verifier/v8-gt --epochs 14 \
      --train-domains mic aug-v1 aug-v2 aug-v3 gt-di gt-amp gt-ego gt-exo \
      --string-only-domains gt-ego gt-exo --epoch-positives 46656

``--train-domains`` picks which rendered recording chains to train on and
``--epoch-positives`` caps how many real notes an epoch draws, so adding
augmented domains compares against the microphone alone at the same number of
optimisation steps rather than at four times the compute.

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

from processing.music import frontend, verify
from processing.tools import guitarset
from processing.training import verifier_data


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_sample(candidates, members, ratio, rng, positives=0, string_only=None):
    """All positives from the split, plus ``ratio`` negatives each.

    Negatives are resampled every epoch, so over a run the model sees far more
    of the 25:1 majority than any one epoch holds. ``ratio`` of 0 trains on
    real notes alone, which is what the string task alone would want.

    ``positives`` caps how many real notes an epoch draws. Training on four
    augmented copies of the same excerpts otherwise quadruples an epoch, and a
    run that takes four times as long is not a comparison.

    ``string_only`` marks excerpts (by index) that teach the string head alone;
    their negatives carry no loss, so none are drawn.
    """
    belongs = np.isin(candidates['excerpt'], list(members))
    positive = np.flatnonzero(belongs & (candidates['label'] == 1))
    if positives and positives < len(positive):
        positive = rng.choice(positive, positives, replace=False)
    if ratio <= 0:
        chosen = positive.copy()
        rng.shuffle(chosen)
        return chosen
    scored = belongs if string_only is None else belongs & ~string_only[candidates['excerpt']]
    negative = np.flatnonzero(scored & (candidates['label'] == 0))
    take = min(len(negative), int(ratio*len(positive)))
    chosen = np.concatenate([positive, rng.choice(negative, take, replace=False)])
    rng.shuffle(chosen)
    return chosen


FRET_BUCKETS = (0, 1, 3, 6, 9, 12, 15)  # Lower edges: open, 1-2, 3-5, ... 15+.


def fret_bucket(candidates):
    """Each candidate's fret bucket, from its pitch and labelled string; -1
    where there is no string (negatives)."""
    string = candidates['string'].astype(np.int64)
    pitch = candidates['pitch'].astype(np.int64) + verify.LOWEST_BIN
    open_pitch = np.asarray(guitarset.OPEN_STRINGS)[np.clip(string, 0, 5)]
    bucket = np.searchsorted(FRET_BUCKETS, pitch - open_pitch, side='right') - 1
    return np.where(string >= 0, bucket, -1)


def fret_weights(candidates, train, reweighted, power):
    """A string-loss weight per candidate that moves ``reweighted`` excerpts'
    fret distribution towards everyone else's in the training split.

    Guitar-TECHS is drills: scales in boxes up the neck, chord sets at every
    position, single notes at every fret. 29% of its notes are at fret 12 or
    above and 0.6% are open, against GuitarSet's 3% and 7.3%. Trained on as
    is, the string head learns that prior along with the timbre, and on IDMT
    it bought .138 -> .352 past fret 15 by giving up .661 -> .578 at frets 0
    to 2, where most playing is. GuitarSet is people playing songs, so its
    fret distribution is the prior to keep. Each reweighted note's string loss
    is scaled by (p_reference(bucket) / p_reweighted(bucket)) ** ``power``,
    normalised to average 1 over the reweighted notes so the corpus keeps its
    share of the loss: 1 matches GuitarSet's distribution exactly, 0 leaves
    the drills as they are.
    """
    weights = np.ones(len(candidates['label']), np.float32)
    if not power or not reweighted.any():
        return weights
    bucket = fret_bucket(candidates)
    in_train = np.isin(candidates['excerpt'], list(train))
    positive = (candidates['label'] == 1) & (bucket >= 0) & in_train
    mine = positive & reweighted[candidates['excerpt']]
    reference = positive & ~reweighted[candidates['excerpt']]
    count = len(FRET_BUCKETS)
    p_mine = np.bincount(bucket[mine], minlength=count) + 1.
    p_reference = np.bincount(bucket[reference], minlength=count) + 1.
    ratio = ((p_reference/p_reference.sum())/(p_mine/p_mine.sum()))**power
    scale = ratio[bucket[mine]].mean()
    rows = reweighted[candidates['excerpt']] & (bucket >= 0)
    weights[rows] = (ratio/scale)[bucket[rows]]
    print('fret reweighting of ' + ', '.join(
        f'{"open" if edge == 0 else f"{edge}+"}: x{value:.2f}'
        for edge, value in zip(FRET_BUCKETS, ratio/scale)))
    return weights


def batches(candidates, patches, rows, size, device, string_only=None, weights=None):
    """Yields (patch, pitch, label, delta, string, scored, weight): ``scored``
    is 1 where the row's existence and onset labels count, 0 for rows from a
    string-only excerpt; ``weight`` scales the row's string loss."""
    excerpt = candidates['excerpt'][rows].astype(np.int64)
    frame = candidates['frame'][rows].astype(np.int64)
    pitch = candidates['pitch'][rows].astype(np.int64)
    label = candidates['label'][rows].astype(np.float32)
    delta = candidates['delta'][rows]*verify.FPS
    string = candidates['string'][rows].astype(np.int64)
    scored = (np.ones(len(rows), np.float32) if string_only is None
              else (~string_only[excerpt]).astype(np.float32))
    weight = (np.ones(len(rows), np.float32) if weights is None
              else weights[rows].astype(np.float32))
    for start in range(0, len(rows), size):
        stop = start + size
        patch = patches.batch(excerpt[start:stop], frame[start:stop], pitch[start:stop])
        yield (torch.from_numpy(patch).to(device),
               torch.from_numpy(pitch[start:stop].astype(np.float32)).to(device),
               torch.from_numpy(label[start:stop]).to(device),
               torch.from_numpy(delta[start:stop]).to(device),
               torch.from_numpy(string[start:stop]).to(device),
               torch.from_numpy(scored[start:stop]).to(device),
               torch.from_numpy(weight[start:stop]).to(device))


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
    parser.add_argument('--train-domains', nargs='*',
                        help='Rendered domains to train on; every one in the '
                             'candidate file by default')
    parser.add_argument('--domain', default=verifier_data.MIC,
                        help='Domain to validate on; the microphone by default')
    parser.add_argument('--string-only-domains', nargs='*', default=[],
                        help='Domains that train the string head only. For labels '
                             'from a MIDI pickup, which knows the string exactly '
                             'and the attack less well than GuitarSet does')
    parser.add_argument('--fret-prior-domains', nargs='*', default=[],
                        help="Domains whose string loss is reweighted by fret "
                             "towards the rest of the training split's distribution")
    parser.add_argument('--fret-prior-power', type=float, default=0.,
                        help='1 matches the distribution exactly, 0 turns it off')
    parser.add_argument('--epoch-positives', type=int, default=0,
                        help='Cap the real notes an epoch draws; 0 takes all of them')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--string-weight', type=float, default=.3)
    parser.add_argument('--delta-weight', type=float, default=.1)
    parser.add_argument('--cqt', type=Path,
                        help='CQT cache to stack on as extra patch channels')
    parser.add_argument('--planes', type=Path,
                        help='Directory of raw prepared excerpts to memory-map '
                             'instead of holding every posteriorgram in RAM; '
                             'written on first use')
    parser.add_argument('--select', choices=['ap', 'string'], default='ap',
                        help='Which validation metric picks best.pt')
    parser.add_argument('--val-sample', type=int, default=80000,
                        help='Candidates to score each epoch; 0 uses all of them')
    parser.add_argument('--front-end', type=Path,
                        help='The processing.music.frontend checkpoint whose '
                             'posteriorgrams to train on; Basic Pitch by default. '
                             'Recorded in the checkpoint, so transcription uses it too')
    parser.add_argument('--cache', type=Path,
                        help='Posteriorgram cache; by default the one --front-end renders '
                             'into (processing.tools.frontend_caches)')
    parser.add_argument('--device')
    parser.add_argument('--max-steps', type=int, help='Smoke test: stop each epoch early')
    args = parser.parse_args()
    if args.cache is None:
        args.cache = frontend.cache_dir(verifier_data.ROOT, args.front_end)
    if args.planes and args.front_end and frontend.tag(args.front_end) not in args.planes.name:
        # Planes are written once and then trusted; another front end's would
        # load without complaint.
        parser.error(f'--planes {args.planes} does not name front end '
                     f'{frontend.tag(args.front_end)}; use e.g. '
                     f'data/verifier/planes@{frontend.tag(args.front_end)}')

    device = pick_device(args.device)
    candidates = verifier_data.load_candidates(args.candidates)
    train, val, test = verifier_data.splits(candidates['players'], candidates['names'],
                                            domain=args.domain)
    if args.train_domains is not None:
        wanted = set(args.train_domains)
        train = {index for index in train
                 if verifier_data.domain_of(candidates['names'][index]) in wanted}
    domains = sorted({verifier_data.domain_of(candidates['names'][index])
                      for index in train})
    string_only = np.array([verifier_data.domain_of(name) in set(args.string_only_domains)
                            for name in candidates['names']])
    print(f'device {device}; excerpts train {len(train)} val {len(val)} test {len(test)}')
    print(f'training on {" ".join(domains)}; validating on {args.domain}')
    if string_only.any():
        print(f'string head only on {" ".join(sorted(set(args.string_only_domains)))} '
              f'({int(string_only[list(train)].sum())} excerpts)')
    reweighted = np.array([verifier_data.domain_of(name) in set(args.fret_prior_domains)
                           for name in candidates['names']])
    weights = fret_weights(candidates, train, reweighted, args.fret_prior_power)
    print('loading posteriorgrams into memory...', flush=True)
    # Only what this run reads: with augmented variants the full set is
    # several gigabytes, and the test split is not scored here.
    patches = verifier_data.Patches(candidates['names'], cache=args.cache, progress=True,
                                    cqt=args.cqt, only=train | val, planes=args.planes)

    model = verify.build(args.width, patches.channels).to(device)
    print(f'verifier: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.

    steps = len(epoch_sample(candidates, train, args.ratio, np.random.default_rng(0),
                             args.epoch_positives, string_only))
    steps = steps//args.batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr,
        total_steps=max(1, args.epochs*(args.max_steps or steps)))

    for number in range(1, args.epochs + 1):
        model.train()
        rows = epoch_sample(candidates, train, args.ratio, rng, args.epoch_positives,
                            string_only)
        started, seen, total = time.time(), 0, 0.
        for step, (patch, pitch, label, delta, string, scored, weight) in enumerate(
                batches(candidates, patches, rows, args.batch, device, string_only,
                        weights)):
            exist, predicted_delta, predicted_string = model(patch, pitch)
            real = label > .5
            # Existence and onset are scored only where the labels are trusted
            # for them; the string is scored on every real note.
            loss = (functional.binary_cross_entropy_with_logits(
                        exist, label, weight=scored, reduction='sum')/scored.sum().clamp(min=1.)
                    if args.ratio > 0 else exist.sum()*0.)
            timed = real & (scored > 0)
            if timed.any():
                loss = loss + args.delta_weight*functional.smooth_l1_loss(
                    predicted_delta[timed], delta[timed])
            if real.any():
                loss = loss + args.string_weight*(functional.cross_entropy(
                    predicted_string[real], string[real], reduction='none')
                    * weight[real]).mean()
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
                        'epoch': number, 'metrics': metrics, 'domains': domains,
                        'front_end': str(args.front_end or '')},
                       args.output/'best.pt')
    (args.output/'history.json').write_text(json.dumps(history, indent=2))
    print(f'\nbest val {args.select} {best:.4f}; wrote {args.output}/best.pt')


if __name__ == '__main__':
    main()
