"""Evaluate dense geometry against held-out annotations, including synthetic occlusion."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from .data import HEADS, read_manifest, render_targets, letterbox, manifest_digest
from .train import select_device
from processing.vision.model import DenseDetector
from processing.vision.tracker import decode_maps


class CounterLike(defaultdict):
    """A dict of integer counts that starts every key at 0."""
    def __init__(self):
        super().__init__(int)


def metrics(prediction, target, valid, threshold=.5, tolerance=3):
    """Raw pixel counts per head for one image.

    Returns counts rather than ratios so they can be summed across images
    before dividing (averaging per-image ratios would let empty images skew it).
    The "matched" counts allow ``tolerance`` pixels of slack, since a wire
    predicted 2 px off is still useful.
    """
    values = {}
    for i, name in enumerate(HEADS):
        mask = valid[..., i] > .5
        if not mask.any():
            continue  # This head wasn't labelled on this image.
        p = (prediction[..., i] > threshold) & mask
        t = (target[..., i] > .5) & mask
        intersection = int((p & t).sum())
        union = int((p | t).sum())
        # Grow each mask by `tolerance` px to count near-misses as hits.
        kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
        td = cv2.dilate(t.astype(np.uint8), kernel) > 0
        pd = cv2.dilate(p.astype(np.uint8), kernel) > 0
        values[name] = {'intersection': intersection, 'union': union,
                        'predicted': int(p.sum()), 'target': int(t.sum()),
                        'matched_prediction': int((p & td).sum()),  # Predicted pixels near a real one.
                        'matched_target': int((t & pd).sum())}      # Real pixels near a predicted one.
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='auto')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--stress', action='store_true', help='Also test with synthetic occluding blocks')
    p.add_argument('--threshold', type=float, default=.5)
    args = p.parse_args()
    if not 0 < args.threshold < 1 or (args.limit is not None and args.limit < 1):
        p.error('Invalid threshold or limit')
    if args.output.exists():
        raise FileExistsError(args.output)

    detector = DenseDetector(args.model, select_device(args.device))
    doc = read_manifest(args.data)
    records = [r for r in doc['records'] if r['split'] == args.split]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError('Empty evaluation split')

    totals = defaultdict(lambda: defaultdict(CounterLike))  # condition -> head -> metric -> count
    outputs = []
    # Occlusion levels: fraction of the neck area to cover with a block.
    levels = [0, .15, .3] if args.stress else [0]
    for record in records:
        image = cv2.imread(record['image'])
        if image is None:
            raise ValueError(f"Cannot decode {record['image']}")
        target, valid = render_targets(record, image.shape, detector.size)
        boxed, _ = letterbox(image, detector.size)
        # Evaluate in letterboxed coordinates at a documented, fixed tolerance.
        for level in levels:
            observed = boxed.copy()
            if level:
                # Cover a square in the middle of the neck with a flat, skin-ish colour.
                ys, xs = np.where(target[..., 0] > .5)
                if len(xs):
                    center = np.array([xs.mean(), ys.mean()])
                    radius = max(2, int(np.sqrt(level * len(xs)) / 2))
                    x, y = np.rint(center).astype(int)
                    observed[max(0, y - radius):y + radius, max(0, x - radius):x + radius] = (70, 100, 150)
            prediction = detector.predict(observed)
            result = metrics(prediction, target, valid, args.threshold,
                             tolerance=max(1, round(detector.size / 256)))
            key = 'clean' if level == 0 else f'synthetic_cover_{level}'
            for head, counts in result.items():
                for metric, value in counts.items():
                    totals[key][head][metric] += value
            # Also record whether the full geometry decoder still finds a board.
            observation = decode_maps(prediction, args.threshold)
            outputs.append({'image': record['image'], 'condition': key,
                            'detected': observation is not None,
                            'numbering_available': observation is not None and observation.numbering != 'unknown'})

    # Turn summed counts into IoU, precision and recall.
    summary = {}
    for condition, heads in totals.items():
        summary[condition] = {}
        for head, c in heads.items():
            precision = c['matched_prediction'] / c['predicted'] if c['predicted'] else None
            recall = c['matched_target'] / c['target'] if c['target'] else None
            summary[condition][head] = {'iou': c['intersection'] / c['union'] if c['union'] else None,
                                        'tolerant_precision': precision, 'tolerant_recall': recall,
                                        **dict(c)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'manifest_sha256': manifest_digest(args.data), 'model': str(args.model.resolve()),
        'split': args.split, 'images': len(records), 'threshold': args.threshold,
        'image_size': detector.size, 'tolerance_px': max(1, round(detector.size / 256)),
        'metrics': summary, 'detections': outputs,
        'limitations': ['Source neck annotations are coarse labels, not exact segmentation.',
                        'Synthetic covers are not a real-hand occlusion benchmark.',
                        'Absolute fret numbering accuracy is unmeasured: current data has no fret-number labels.',
                        'Verify guitar/session holdouts before making generalization claims.']},
        indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
