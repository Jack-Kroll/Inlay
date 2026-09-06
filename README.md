# inlay

A guitar app prototype for turning recorded playing into musical observations.
The current implementation is a Python fretboard detector and diagnostic preview;
the planned mobile client uses React Native and TypeScript.

**Available:** trainable dense fretboard model, dataset auditing/merging, partial
label supervision, occlusion augmentation, motion-supported tracking, video/JSONL
diagnostics, and held-out pixel evaluation. The previous YOLO OBB detector remains
available as a baseline. Audio, hand tracking, played-note inference and the mobile
app are still to build. The rewrite is training-ready; improved camera accuracy
has not yet been demonstrated.

## Get started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. From the project root:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
```

## Train the replacement

The local dataset is useful for bootstrapping but needs more independent guitars,
recording sessions, negatives and real occlusion examples. Read the
[dataset audit and recommended additions](docs/dataset-review.md).

```sh
uv run python -m processing.training.prepare \
  --obb guitar_obb_dataset --output data/fretboard.json
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps
```

The model uses a pretrained ResNet34 encoder and predicts neck, metal fret lines,
and nut, without depending on decorative markers. Training produces
`runs/dense/fretboard-v1/best.pt`. The first normal run downloads official ImageNet
encoder weights; it does not start with a pretrained guitar detector.

For CUDA use `--device cuda` and increase batch size if memory permits.
See [model commands](docs/models.md) for resume, evaluation and data merging.

## Preview after training

```sh
uv run python -m processing.vision.preview \
  --model runs/dense/fretboard-v1/best.pt --source 0 \
  --device mps --show-heatmaps
```

Use **q** to quit, **r** to reset. Replace `0` with a video path for repeatable
checks. Heatmaps show each model output; the overlay distinguishes measurements
from short-term tracking and reports unknown fret numbering. Board detection can
continue without a visible nut, but absolute numbering needs sufficient evidence.
String positions and played notes are not inferred by this model.

**macOS camera access:** allow the initial prompt, then rerun if needed. If denied,
enable the terminal/app in **System Settings → Privacy & Security → Camera**.
Restart that app after changing permissions and close other camera applications.

## Compare the original detector

Existing OBB checkpoints use the legacy entry point:

```sh
uv run python -m processing.vision.legacy_preview \
  --model runs/obb/runs/obb/guitar_fretboard_obb_gpu-4/weights/best.pt \
  --source 0 --show-detections
```

Local datasets and weights are not included in source control. Preserve dataset
attribution files when sharing them. There is no shared artifact download yet.

## Where to work

| Location | Responsibility |
|---|---|
| `processing/vision/model.py` | Dense model, loss, preprocessing and inference |
| `processing/vision/tracker.py` | Wire geometry, conservative numbering and tracker |
| `processing/vision/preview.py` | Camera/video diagnostics and timestamped JSONL |
| `processing/training/data.py` | Manifest validation, targets and augmentation |
| `processing/training/prepare.py` | Dataset audit, clipping, merging and overlap exclusion |
| `processing/training/train.py` | Training and checkpoint resume |
| `processing/training/evaluate.py` | Clean and synthetic-occlusion pixel evaluation |
| `processing/vision/legacy_preview.py` | Original OBB overlay for comparison |
| `packages/music` | Versioned timed-note contract for the future app |
| `tests` | Geometry, data, model, tracker and compatibility checks |

Keep processing coordinates independent of display transforms. Agree on changes
to `packages/music/performance.schema.json` before integrating musical features.
Run tests before sharing changes. Commit `pyproject.toml`, `uv.lock`, and
`.python-version` together when changing dependencies. Keep `.venv`, data, runs
and model binaries out of source control.

See [architecture and limitations](docs/architecture.md). Core ML tools are optional
(`uv run --extra coreml`) and currently support the legacy OBB model only.
