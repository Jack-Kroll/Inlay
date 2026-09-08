"""Compare PyTorch/Core ML detections on one image in the same coordinate space.

This checks Python runtime export parity, not ground-truth accuracy.
"""
import argparse
import numpy as np


def compare(reference, exported, max_corner_error=3.0, max_conf_error=0.05):
    for detections in (reference, exported):
        for _, confidence, corners in detections:
            if not np.isfinite(confidence) or not np.isfinite(corners).all():
                raise AssertionError('Detections contain non-finite confidence or corners')
    if len(reference) != len(exported):
        raise AssertionError(f"Detection counts differ: {len(reference)} vs {len(exported)}")
    if not reference:
        raise AssertionError("Both models returned no detections; use an image containing a guitar")
    remaining = list(exported)
    worst = 0.0
    for cls, conf, corners in reference:
        candidates = []
        for i, (other_cls, other_conf, other_corners) in enumerate(remaining):
            if cls != other_cls:
                continue
            # Corner order may differ across backends; compare all cyclic orders.
            error = min(float(np.linalg.norm(corners - np.roll(order, shift, axis=0), axis=1).max())
                        for order in (other_corners, other_corners[::-1]) for shift in range(4))
            if abs(conf - other_conf) <= max_conf_error:
                candidates.append((error, i))
        if not candidates:
            raise AssertionError(f"No matching class/confidence for class {cls}")
        error, index = min(candidates)
        if error > max_corner_error:
            raise AssertionError(f"Corner error {error:.3f}px exceeds {max_corner_error}px")
        worst = max(worst, error)
        remaining.pop(index)
    return worst


def detections(model_path, image, imgsz, conf):
    from ultralytics import YOLO
    result = YOLO(str(model_path), task="obb").predict(
        source=str(image), imgsz=imgsz, conf=conf, rect=False,
        device="cpu", verbose=False,
    )[0]
    if result.obb is None:
        return []
    obb = result.obb.cpu().numpy()
    return list(zip(np.asarray(obb.cls).astype(int),
                    np.asarray(obb.conf), np.asarray(obb.xyxyxyxy)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("exported")
    parser.add_argument("image")
    parser.add_argument("--imgsz", type=int, default=640, help="Must match export size")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-corner-error", type=float, default=3.0)
    parser.add_argument("--max-conf-error", type=float, default=0.05)
    args = parser.parse_args()
    ref = detections(args.checkpoint, args.image, args.imgsz, args.conf)
    out = detections(args.exported, args.image, args.imgsz, args.conf)
    worst = compare(ref, out, args.max_corner_error, args.max_conf_error)
    print(f"PASS: {len(ref)} detections; worst corner error {worst:.3f}px")


if __name__ == "__main__":
    main()
