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
`runs/verifier/v8-gt/best.pt`. Without the verifier checkpoint, transcription
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
The same files also include `Blackbird.performance.json`: the timed notes in
the [`packages/music`](packages/music) format the app reads.

## Run it as a service

The pipeline also runs behind an HTTP job API, so a phone app can upload a
recording and poll for its tab. Locally, API and worker run on this Mac:

```sh
uv run python -m service.api --worker
```

Moving to a server is configuration, not code: storage, the job store and the
device are environment variables. See [the service](docs/service.md).

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
verifier is a small CNN over posteriorgram patches, trained on GuitarSet and
Guitar-TECHS with whole players held out.

## Train and test

The fretboard detector:

```sh
uv run python -m processing.training.train \
  --data data/fretboard.json --output runs/dense/fretboard-v1 \
  --epochs 150 --imgsz 768 --batch 1 --accumulate 16 --device mps
```

The note verifier, from GuitarSet under `data/guitarset` and Guitar-TECHS under
`data/guitar-techs/raw`:

```sh
uv run python -m processing.tools.cqt
uv run python -m processing.tools.variants --variants 3
uv run python -m processing.tools.guitar_techs --prepare
uv run python -m processing.training.verifier_data \
  --domains aug-v1 aug-v2 aug-v3 --measure-domains gt-di gt-amp gt-ego gt-exo
uv run python -m processing.training.train_verifier \
  --cqt data/guitarset/cqt --planes data/verifier/planes \
  --output runs/verifier/v8-gt --epochs 14 \
  --train-domains mic aug-v1 aug-v2 aug-v3 gt-di gt-amp gt-ego gt-exo \
  --string-only-domains gt-ego gt-exo --epoch-positives 46656

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

Audio against annotated audio, no video, threshold .75, current defaults: the
verifier trained on GuitarSet and Guitar-TECHS, the audio string term at .25,
and audio not allowed to argue a note off an open string. "Previous" is the
default before Guitar-TECHS: the acoustic-only verifier with the string term
off. Current rows include the open-string rival correction and duplicate
suppression after timing refinement. Previous rows are historical measurements
with the earlier decoder. These are previously inspected regression corpora;
see [the confidence and missed-note experiments](docs/visual-confidence-and-missed-notes.md).

| held out of training | | note F1 | string | tab F1 |
|---|---|---:|---:|---:|
| GuitarSet player 05, acoustic, 8,715 notes | previous | `.894` | `.664` | `.594` |
| | current | `.902` | `.869` | `.784` |
| Guitar-TECHS player P3, electric, amp mic | previous | `.703` | `.472` | `.332` |
| | current | `.731` | `.612` | `.448` |
| IDMT-SMT-GUITAR, four electrics, 5,164 notes | previous | `.820` | `.509` | `.418` |
| | current | `.850` | `.585` | `.497` |

The Guitar-TECHS row is player P3; earlier room-microphone measurements gave
`.518` to `.560` (5 ft in front) and `.488` to `.565` (head-mounted).
Nothing on IDMT is trained on, and nothing shipped was chosen on it.

In the earlier Thunderstruck open-string experiment, the string term on its
own put 42 of 48 open-B notes on the G string. The selected configuration kept
34 open and increased fingertip agreement from `.652` to `.721`. This is a
diagnostic of visual consistency, not independent reference accuracy. See
[tab logic](docs/tab-logic.md#the-default-and-the-one-thing-audio-may-not-decide).

- Complete audio/video transcription accuracy has still not been measured.
  Nothing in the table involves the camera.
- The string head has heard six acoustic and three electric guitars. Through a
  room it cannot tell an open string from a fretted one, which is why it is not
  allowed to decide that.
- A high-gain amp breaks note detection: on re-amped Guitar-TECHS, precision
  falls from `.90` to `.52`. Training on synthetic distortion recovered part of
  that and cost clean recordings more; it is not shipped.
- Strings are interpolated rather than detected.
- `--verifier-threshold`, `--audio-string-weight` and `--open-audio-neutral`
  are exposed as flags, and recorded in each transcription's JSONL summary.
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
- `processing/tools`: GuitarSet, Guitar-TECHS and IDMT loaders and evaluation,
  CQT caching, and tuning utilities.
- `service`: the job API, worker, storage and job store.
- `packages/music`: timed-note JSON contract for a future editor.
- `tests`: unit and integration tests.

Inlay source code is available under the [MIT License](LICENSE). Local visual
data and GuitarSet are CC BY 4.0 and require their own attribution.
Guitar-TECHS is CC BY 4.0 and requires its own attribution.
IDMT-SMT-GUITAR is CC BY-NC-ND 4.0 and is used only to measure generalisation:
nothing is trained on it and nothing derived from it ships.
