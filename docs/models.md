# Fretboard model

The default detector is `fretboard-resnet34-unet-v1`: a pretrained ResNet34
encoder, U-Net decoder, and independent neck, fret-wire, and nut heatmaps.
Improved camera accuracy has not yet been established.

## Prepare and train

```sh
uv sync --locked
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps
```

Use `--device cuda --batch 4 --accumulate 4` on CUDA hardware. Automatic
selection tries CUDA, then MPS, then CPU.

Training uses AdamW, cosine learning-rate decay, gradient accumulation and
clipping, frozen encoder BatchNorm statistics, GroupNorm in the decoder, and
weighted BCE plus Dice loss. Augmentation covers perspective, rotation, scale,
reflection, exposure, noise, blur, and synthetic covers.

Synthetic covers teach completion of known geometry. They do not prove
accuracy under real hand occlusion.

`best.pt` contains the lowest-validation-loss model. `last.pt` also preserves
optimizer, scheduler, scaler, and random state for resuming:

```sh
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps \
  --resume runs/dense/fretboard-v1/last.pt
```

The manifest and original settings must match. A tiny `--max-batches` run is a
software smoke test, not a useful detector.

## Evaluate

Choose thresholds on validation, then evaluate the selected model on test:

```sh
uv run python -m processing.training.evaluate \
  --data data/fretboard.json --model runs/dense/fretboard-v1/best.pt \
  --split val --stress --output runs/dense/fretboard-v1/validation.json

uv run python -m processing.training.evaluate \
  --data data/fretboard.json --model runs/dense/fretboard-v1/best.pt \
  --split test --stress --output runs/dense/fretboard-v1/test.json
```

Reports include pixel IoU and tolerance-based precision/recall for each labeled
head, with optional synthetic covers. Coarse source labels limit neck IoU.
Numbering availability is not numbering accuracy.

## Preview

```sh
uv run python -m processing.vision.preview
```

The defaults are the best checkpoint, camera `0`, Apple MPS, mirrored display,
and estimated strings. Use `--source VIDEO` for repeatable checks. Add
`--show-heatmaps` for diagnostic maps or `--jsonl FILE --headless` to record
observations without opening windows.

- Green: current detection.
- Amber: short motion-supported tracking.
- Blue wires: measured.
- Orange `~N` wires: estimated or tracked.
- `q`: quit; `r`: reset; `m`: toggle display mirroring.

The preview processes only the newest live-camera frame so latency does not
build a queue. Recorded videos process every frame. Try `--imgsz 640` for speed,
with a possible loss of fine fret detail.

The normalized manifest records source, split group, supervised heads and
attribution. Unknown heads receive zero loss. Keep the same guitar or recording
session in one split when adding data.
