"""
Train a YOLO OBB model for guitar neck/fret detection and export it for iOS/CoreML.

Expected dataset: Roboflow YOLOv8-OBB zip with:
  data.yaml
  train/images, train/labels
  valid/images, valid/labels
  test/images, test/labels

Classes in your dataset:
  0: fret
  1: neck
  2: nut

Usage:
  uv run --extra legacy python -m processing.training.train_obb --zip guitar.v1i.yolov8-obb.zip --epochs 100 --imgsz 640

Outputs:
  runs/obb/guitar_fretboard_obb/weights/best.pt
  runs/obb/guitar_fretboard_obb/weights/best.mlpackage  (CoreML export)
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, required=True, help="Path to Roboflow YOLOv8-OBB dataset zip")
    parser.add_argument("--workdir", type=str, default="guitar_obb_dataset", help="Folder to unzip dataset into")
    parser.add_argument("--model", type=str, default="yolov8n-obb.pt", help="Use yolov8n/s/m/l/x-obb.pt")
    parser.add_argument("--name", default="guitar_fretboard_obb", help="Unique experiment name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics auto-pick batch size")
    parser.add_argument("--device", type=str, default=None, help="Examples: cpu, 0, mps")
    parser.add_argument("--export", action="store_true", help="Export best.pt to CoreML after training")
    parser.add_argument("--predict", type=str, default=None, help="Optional image/video path to test after training")
    return parser.parse_args()


def unzip_dataset(zip_path: Path, workdir: Path) -> Path:
    if workdir.exists():
        print(f"Dataset folder already exists: {workdir}")
    else:
        print(f"Unzipping {zip_path} -> {workdir}")
        workdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(workdir)

    data_yaml = workdir / "data.yaml"
    if not data_yaml.exists():
        matches = list(workdir.rglob("data.yaml"))
        if not matches:
            raise FileNotFoundError("Could not find data.yaml after unzipping.")
        data_yaml = matches[0]

    print(f"Using data config: {data_yaml}")
    return data_yaml


def train_model(data_yaml: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/obb",
        name=args.name,
        exist_ok=False,
        patience=30,
        plots=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    if model.trainer is None:
        raise RuntimeError('YOLO training did not create a trainer')
    best_pt = Path(model.trainer.best)
    if not best_pt.is_file():
        raise FileNotFoundError(f"Training did not produce its expected checkpoint: {best_pt}")

    print(f"Best model: {best_pt}")
    return best_pt


def demo_predict(best_pt: Path, source: str, imgsz: int) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    results = model.predict(source=source, imgsz=imgsz, conf=0.25, save=True)

    for result in results[:1]:
        if result.obb is None:
            print("No OBB detections.")
            return
        print("Class names:", result.names)
        print("OBB corners shape:", result.obb.xyxyxyxy.shape)
        print("First boxes as 4 corner points:")
        print(result.obb.xyxyxyxy[:3])


if __name__ == "__main__":
    args = parse_args()
    zip_path = Path(args.zip).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()

    data_yaml = unzip_dataset(zip_path, workdir)
    best_model = train_model(data_yaml, args)

    if args.export:
        from .export_coreml import export_coreml
        export_coreml(best_model, args.imgsz)

    if args.predict:
        demo_predict(best_model, args.predict, args.imgsz)
