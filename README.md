# Inlay

Inlay turns a recorded guitar performance into an annotated video and structured
tablature data. It combines a custom fretboard detector, fingertip tracking, and
audio pitch detection.

This is a Python research prototype. Offline transcription works; live
transcription, a mobile app, and validated end-to-end accuracy are future work.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
```

Local datasets, clips, runs, and model weights are gitignored. The commands below
expect the trained checkpoint at `runs/dense/fretboard-v1/best.pt`.

## Preview the fretboard

```sh
uv run python -m processing.vision.preview
```

This opens camera `0` with the best checkpoint and Apple MPS. Use `q` to quit,
`r` to reset, or `m` to toggle mirroring. For a recording or diagnostic maps:

```sh
uv run python -m processing.vision.preview \
  --source "clips/Blackbird.mov" --show-heatmaps
```

## Transcribe a recording

```sh
uv run python -m processing.music.transcribe \
  --video "clips/Blackbird.mov"
```

Results go to `runs/transcribe/blackbird/` by default:

- `Blackbird.overlay.mp4`: video with fret, string, fingertip, and note overlays.
- `Blackbird.tab.jsonl`: structured note assignments and diagnostic evidence.

Use `--model`, `--output`, or `--device` to override the defaults.

## How it works

1. `FretboardNet` predicts heatmaps for the neck, fret wires, and nut.
2. The tracker stabilizes that geometry across short detection gaps.
3. Spotify Basic Pitch detects sounding pitches and onsets.
4. MediaPipe locates the fretting-hand fingertips.
5. Inlay combines pitch, hand position, and guitar constraints to select a
   `(string, fret)` position for each note.

FretboardNet uses a pretrained ResNet34 encoder with a custom U-Net-style
decoder, output heads, loss, training loop, and inference pipeline.

## Train and test

```sh
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps

uv run python -m unittest discover -s tests -v
uv run pyright
```

See [model training](docs/models.md), [architecture](docs/architecture.md), and
[verification](docs/verification.md) for details.

## Current results and limits

- Basic Pitch tuning reached `.779` F1 on held-out GuitarSet excerpts.
- Blind string assignment with perfect reference pitches reached `.698` accuracy.
- Complete audio/video transcription accuracy has not been measured.
- Strings are interpolated rather than detected.
- Capos, bends, slides, barre chords, and alternate tunings are not inferred.

More detail is available in [transcription](docs/transcription.md),
[pitch tuning](docs/pitch-tuning.md), [tab logic](docs/tab-logic.md), and the
[dataset review](docs/dataset-review.md).

## Code layout

- `processing/vision`: fretboard model, decoding, tracking, and preview.
- `processing/music`: pitch, fingertips, tab assignment, and transcription.
- `processing/training`: dense-model dataset, training, and evaluation.
- `processing/tools`: GuitarSet evaluation and tuning utilities.
- `packages/music`: timed-note JSON contract for a future editor.
- `tests`: unit and integration tests.

Inlay source code is available under the [MIT License](LICENSE). Local visual
data and GuitarSet are CC BY 4.0 and require their own attribution.
