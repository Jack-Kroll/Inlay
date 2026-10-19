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

Local datasets, clips, runs, and model weights are gitignored. The commands
below expect the trained checkpoints at `runs/dense/fretboard-v1/best.pt` and
`runs/verifier/v3-aug/best.pt`. Without the verifier checkpoint, transcription
falls back to Basic Pitch's thresholds and says so.

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
3. Spotify Basic Pitch produces a posteriorgram over pitch and time.
4. A learned note verifier reads that posteriorgram, plus a constant-Q
   transform of the audio, and decides which attacks are real notes, when they
   started, and which string played each one.
5. MediaPipe locates the fretting-hand fingertips.
6. Inlay combines pitch, the audio string posterior, hand position, and guitar
   constraints to select a `(string, fret)` position for each note.

FretboardNet uses a pretrained ResNet34 encoder with a custom U-Net-style
decoder, output heads, loss, training loop, and inference pipeline. The note
verifier is a small CNN over posteriorgram patches, trained on GuitarSet with
whole players held out.

## Train and test

The fretboard detector:

```sh
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps
```

The note verifier, from GuitarSet under `data/guitarset`:

```sh
uv run python -m processing.tools.cqt
uv run python -m processing.tools.variants --variants 3
uv run python -m processing.training.verifier_data --domains aug-v1 aug-v2 aug-v3
uv run python -m processing.training.train_verifier \
  --cqt data/guitarset/cqt --output runs/verifier/v3-aug --epochs 14 \
  --train-domains mic aug-v1 aug-v2 aug-v3 --epoch-positives 46656

uv run python -m processing.tools.eval_verifier --split test
```

`variants` renders augmented copies of the training excerpts — level, EQ,
cabinet, amp drive, hiss and codec — which is what carries the string head off
the one microphone it was trained on.

```sh
uv run python -m unittest discover -s tests -v
uv run pyright
```

See [model training](docs/models.md), [the note verifier](docs/note-verifier.md),
[architecture](docs/architecture.md), and [verification](docs/verification.md)
for details.

## Current results and limits

Audio against annotated audio, on the GuitarSet player held out of training,
checkpoint selection, and threshold choice — 8,715 notes, no video:

| | note F1 | string | tab F1 |
|---|---:|---:|---:|
| Basic Pitch thresholds | `.821` | `.653` | `.536` |
| Note verifier + CQT | `.898` | `.882` | `.792` |
| + recording-chain augmentation | `.902` | `.880` | `.794` |

On IDMT-SMT-GUITAR's electric guitars, which nothing is trained on, note F1
goes `.695` to `.817`, string accuracy `.514` to `.634`, and tab F1 `.358` to
`.518`.

The string columns above need `--audio-string-weight .25`, which is **not the
default**. That term is worth 21 points of string accuracy on GuitarSet and is
the only thing that places an open string there, but on a distorted electric it
put 42 of Thunderstruck's open-B notes on the wrong string. It ships off until
the string head has electric training data; without it the shipped default
scores `.902` note F1 and `.603` tab F1 on the table above. See
[tab logic](docs/tab-logic.md).

- Complete audio/video transcription accuracy has still not been measured.
  Nothing above involves the camera.
- The verifier is trained on three hours of acoustic guitar from six players,
  augmented into a wider range of recording chains but not a wider range of
  guitars. Note detection transfers to electric guitars well; the string head
  only partly, and on two of eight IDMT subsets it is still beaten by musical
  context alone.
- Strings are interpolated rather than detected.
- The two settings chosen on GuitarSet, the verifier's score threshold and the
  audio string weight, are both exposed as flags because they are the first
  things to reach for on material that does not sound like GuitarSet.
- Capos, bends, slides, barre chords, and alternate tunings are not inferred.

More detail is available in [the note verifier](docs/note-verifier.md),
[transcription](docs/transcription.md), [pitch tuning](docs/pitch-tuning.md),
[tab logic](docs/tab-logic.md), and the
[dataset review](docs/dataset-review.md).

## Code layout

- `processing/vision`: fretboard model, decoding, tracking, and preview.
- `processing/music`: pitch, the note verifier, fingertips, tab assignment, and
  transcription.
- `processing/training`: dense-model and verifier datasets, training, and
  evaluation.
- `processing/tools`: GuitarSet and IDMT evaluation, CQT caching, and tuning
  utilities.
- `packages/music`: timed-note JSON contract for a future editor.
- `tests`: unit and integration tests.

Inlay source code is available under the [MIT License](LICENSE). Local visual
data and GuitarSet are CC BY 4.0 and require their own attribution.
IDMT-SMT-GUITAR is CC BY-NC-ND 4.0 and is used only to measure generalisation:
nothing is trained on it and nothing derived from it ships.
