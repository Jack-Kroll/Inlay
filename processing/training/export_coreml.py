"""Export an explicitly selected checkpoint; no model discovery or training."""
import argparse
from pathlib import Path


def export_coreml(checkpoint: Path, image_size: int, half: bool = False):
    from ultralytics import YOLO

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return YOLO(str(checkpoint)).export(format="coreml", imgsz=image_size, half=half)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    print(export_coreml(args.checkpoint, args.imgsz, args.half))


if __name__ == "__main__":
    main()
