# FretboardNet training and evaluation

The default detector is now `fretboard-resnet34-unet-v1`: a ResNet34 encoder with
ImageNet initialization, a U-Net decoder, and three independent dense outputs:
neck occupancy, fret-wire centerlines, and nut centerline. Decoder features reach
half the input resolution before interpolation. All three heads can overlap.
This is a trainable research replacement; improved camera accuracy has not yet
been established. Existing YOLO checkpoints are incompatible with the new preview.

## Train the replacement

From the project root:

```sh
uv sync --locked
uv run python -m processing.training.prepare \
  --obb guitar_obb_dataset --output data/fretboard.json
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps
```

For CUDA, use `--device cuda --batch 4 --accumulate 4` if memory permits. Automatic
selection is `--device auto` (CUDA, then MPS, then CPU). The first normal training
run downloads the official torchvision ResNet34 encoder weights. Training the
fretboard heads is still required. Start at 768 pixels; investigate 1024 only with
higher-resolution source images and enough memory. CPU is supported for checks.

The provided prepared manifest can be used directly on this machine. Regenerate
it after moving datasets or changing sources. It records absolute image paths,
image hashes, source attribution, derived targets, and excluded overlap. The
original export and prior checkpoints remain intact.

Training uses AdamW, cosine decay, frozen encoder BatchNorm running statistics,
GroupNorm in the decoder, weighted BCE plus Dice losses, gradient accumulation,
clipping, CUDA mixed precision, source/original-image balancing, and deterministic
per-epoch augmentation seeds. Augmentation includes perspective, rotation, scale,
translation, reflection, exposure/noise/blur, and local synthetic covers. Under
synthetic covers known targets remain: this teaches completion of known geometry.
It does not teach reliable inference through arbitrary real occlusion by itself.

`best.pt` selects the lowest validation loss; `last.pt` includes optimizer,
scheduler, scaler and RNG state for resuming. Both record the manifest hash and
architecture. `metrics.jsonl`, `config.json` and `dataset-audit.json` accompany
them. Existing output directories are rejected unless resuming. Use the same
training settings and dataset when resuming, for example:

```sh
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps \
  --resume runs/dense/fretboard-v1/last.pt
```

`--max-batches 1 --epochs 2 --imgsz 128 --batch 2 --accumulate 1 --from-scratch`
is an offline smoke-test configuration only. Its checkpoints have `smoke_test=true`
and are not useful guitar detectors. Normal runs must omit those smoke options.

## Evaluate before relying on it

Choose thresholds on validation; keep test evaluation for the selected model.
The current split remains provisional until source sessions/guitars are identified.

```sh
uv run python -m processing.training.evaluate \
  --data data/fretboard.json --model runs/dense/fretboard-v1/best.pt \
  --split val --stress --output runs/dense/fretboard-v1/validation.json
uv run python -m processing.training.evaluate \
  --data data/fretboard.json --model runs/dense/fretboard-v1/best.pt \
  --split test --stress --output runs/dense/fretboard-v1/test.json
```

The report contains IoU and tolerance-based pixel precision/recall per supervised
head, clean and synthetic-cover conditions, and per-image detection/numbering
availability. A cover level controls square-cover area relative to board area;
it is not a measured real-hand occlusion percentage. Coarse OBB masks limit the
meaning of neck IoU. Numbering **availability** is not numbering **accuracy**.
See [the data review](dataset-review.md) for collection and validation criteria.

## Camera/video diagnostics

```sh
uv run python -m processing.vision.preview \
  --model runs/dense/fretboard-v1/best.pt --source 0 \
  --device mps --show-heatmaps
```

Use a recorded video path instead of `0` for repeatable tests. Add
`--jsonl runs/dense/fretboard-v1/clip.jsonl --headless` to record timestamped
image-space observations without windows. Existing JSONL files are not overwritten.
Use **q** to quit, **r** to reset tracking.

Green boundary means detected; amber means motion-supported short-term tracking.
Separate heatmaps show the neck, wires and nut. Fret numbering is fitted only
when nut plus enough consistent wire evidence identify a sufficiently distinct
projective solution. A board can still be detected with no nut. Short-term
numbers can persist through verified motion for up to `--max-gap` seconds
(default 0.5); longer uncertainty clears numbers. Without motion support the
tracker drops an occluded overlay immediately instead of freezing it in place.

The preview shows measured wire lines and optional numbers. It does not synthesize
strings or display guessed notes. Dense neck completion and inferred full wire
extents remain predictions; a hand or off-screen area cannot be guaranteed correct.

## Combining locally downloaded sources

`--sources` accepts a JSON array; use explicit mappings and declare **only heads
that are exhaustively annotated**. Example neck-only COCO source:

```json
[
  {
    "name": "additional-neck-data",
    "format": "coco",
    "root": "/absolute/path/to/export",
    "mapping": {"fretboard": "neck"},
    "supervised": ["neck"],
    "annotation_file": "_annotations.coco.json",
    "attribution": "Exact source URL and author from the export",
    "license": "Exact export license"
  }
]
```

Use `--obb guitar_obb_dataset --sources sources.json` to combine that source with
the baseline. The expected source directories are `train`, `valid`/`val`, `test`.
COCO image paths are relative to their split folder. For OBB exports, each split
has `images/` and `labels/`, and the root `data.yaml` supplies class names. An
empty OBB label file means a reviewed negative; a missing label file is an error.
Wire classes must mean physical fret/nut wires, not fret spaces or markers.
COCO imports currently support neck polygons only, not RLE/crowd masks or poses.

For the audited B101 source, `mask_missing_neck: true` also masks neck loss
when fret wires are annotated but the neck box is missing. This source-specific
rule preserves fully reviewed negative images.

Unknown fret/nut labels in a neck-only dataset receive **zero loss**, not negative
targets. Near duplicates and filename variants stay in one split across sources.
Attribution stays attached in the manifest. Use `--groups groups.json` with a
complete mapping from `source:filename-before-.rf.` to shared guitar/session IDs
to strengthen grouping; choose those IDs according to the evaluation boundary.
Images changing since preparation fail the integrity check.

For native manifest annotations, coordinates are normalized XY, polygons have
at least three vertices, lines at least two. Records contain `image`, `split`,
`source`, `group`, `supervised`, `annotations` and optionally `sha256` and
`original_group`. A reviewed negative has `annotations: []` with appropriate
supervised heads. Version is 1 and heads are `["neck", "fret", "nut"]`.

## Legacy baseline and export

The original code remains available for controlled comparison:

```sh
uv run python -m processing.vision.legacy_preview \
  --model runs/obb/runs/obb/guitar_fretboard_obb_gpu-4/weights/best.pt \
  --source 0 --show-detections
uv run python -m processing.training.train_obb \
  --zip guitar.v1i.yolov8-obb.zip --name obb-baseline
```

`processing.training.export_coreml` and `processing.tools.verify_coreml` remain
**YOLO OBB-only** tools. Do not pass dense checkpoints to them. Dense Core ML
conversion and device parity/latency testing are future deployment work; select
and validate a trained dense checkpoint before building the mobile runtime.
