# inlay

A guitar app prototype for turning recorded playing into musical observations.
The current implementation is a Python fretboard detector and diagnostic preview;
the planned mobile client uses React Native and TypeScript.

**Available:** trainable dense fretboard model, dataset auditing/merging, partial
label supervision, occlusion augmentation, motion-supported tracking, video/JSONL
diagnostics, held-out pixel evaluation, and offline tab transcription combining
pitch detection with fretting-hand tracking. The previous YOLO OBB detector
remains available as a baseline. Live transcription and the mobile app are still
to build. Audio pitch and blind string assignment have been measured on GuitarSet;
end-to-end transcription and camera accuracy remain unmeasured.

## Get started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. From the project root:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
```

Pitch-decoder/scoring tests require `uv sync --locked --extra transcribe`; they
are skipped with the base install. For VS Code, select `.venv/bin/python` as the
Python interpreter. The project includes a basic Pyright configuration.

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

## Transcribe a recording to tab

```sh
uv sync --locked --extra transcribe
uv run --extra transcribe python -m processing.music.transcribe \
  --model runs/dense/fretboard-v1/best.pt \
  --video clips/take-1.mp4 --output runs/transcribe/take-1 \
  --device mps --progress
```

This writes an annotated video and a JSONL note list. Sounding pitch comes from
Spotify Basic Pitch, fret geometry from the dense detector, and fingertip
positions from MediaPipe; the three are combined into `(string, fret)` per note.
Every note reports how well the hand actually supported it, and unexplained
fingers are listed rather than hidden. Strings are interpolated across the board,
not detected, and standard tuning is assumed unless `--tuning` says otherwise.
It runs on a recording because Basic Pitch reads a whole soundtrack.

The audio thresholds are tuned against GuitarSet rather than guessed: F1 .779 on
held-out annotated guitar, up from .740 at Basic Pitch's own defaults. See
[pitch tuning](docs/pitch-tuning.md) for the sweep,
[tab logic](docs/tab-logic.md) for the string-choice measurement, and
[tab transcription](docs/transcription.md) for the geometry, the scoring and the
list of things it cannot do.

## Compare the original detector

Existing OBB checkpoints use the legacy entry point:

```sh
uv run --extra legacy python -m processing.vision.legacy_preview \
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
| `processing/music/fretboard.py` | Image to (fret, string) board coordinates |
| `processing/music/fusion.py` | Pitch plus hand evidence to tab positions; neck-position planning |
| `processing/music/transcribe.py` | Offline transcription CLI and overlay |
| `processing/music/hands.py` | MediaPipe fingertips; `pitch.py` Basic Pitch |
| `processing/tools/tune_pitch.py` | Basic Pitch settings swept against GuitarSet |
| `processing/tools/eval_tab.py` | String choice scored against GuitarSet |
| `processing/tools/rescue_bound.py` | Ceiling for a visual-candidate note rescue |
| `packages/music` | Versioned timed-note contract for the future app |
| `tests` | Geometry, data, model, tracker and compatibility checks |

Keep processing coordinates independent of display transforms. Agree on changes
to `packages/music/performance.schema.json` before integrating musical features.
Run tests before sharing changes. Commit `pyproject.toml`, `uv.lock`, and
`.python-version` together when changing dependencies. Keep `.venv`, data, runs
and model binaries out of source control.

See [architecture and limitations](docs/architecture.md). Core ML tools are optional
(`uv run --extra legacy --extra coreml`) and currently support the legacy OBB model only. The
YOLO baseline now needs `--extra legacy`, which keeps `ultralytics` and its
`opencv-python` dependency out of the default install: MediaPipe requires
`opencv-contrib-python`, and two distributions cannot share the `cv2` namespace.
