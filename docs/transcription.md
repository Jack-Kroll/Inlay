# Tab transcription from a recording

Three independent signals are combined into tab: **sounding pitch** from audio,
**fret geometry** from the dense detector, and **fingertip position** from
MediaPipe. None of them alone is enough, and the combination has not been
measured against annotated playing. Treat the output as a diagnostic.

Install the extra and run it on a clip that has sound:

```sh
uv sync --locked --extra transcribe
uv run python -m processing.music.transcribe \
  --model runs/dense/fretboard-v1/best.pt \
  --video clips/take-1.mp4 --output runs/transcribe/take-1 \
  --device mps --progress
```

This writes `take-1.overlay.mp4` (annotated video with the original audio) and
`take-1.tab.jsonl` (a summary line followed by one line per note). The first run
downloads the MediaPipe hand landmarker to `~/.cache/inlay`; set `INLAY_CACHE` or
pass `--hand-model` to place it elsewhere.

Add `--no-video` to skip rendering when only the JSONL matters, `--max-frames`
to cut a long clip short, and `--mirror` to flip the rendered video for a
front-camera recording. Mirroring is display only: the board fit, the tab and
the JSONL always stay in original camera coordinates.

## Why this is offline

Basic Pitch consumes a whole soundtrack, so the audio stage cannot run frame by
frame without redesigning it around a sliding window. The video is decoded
twice: once to run the detector and the hand tracker, once to draw. That order
exists so the string-order decision below is made before any frame is rendered
with it. Live capture would need a streaming pitch stage and a committed string
order up front; neither exists yet.

## From a fingertip to a string and fret

`processing/music/fretboard.py` fits a homography between the image and board
coordinates `(ratio, across)`. `ratio` is `1 - 2**(-n/12)` at fret `n`, which is
proportional to physical distance from the nut, so the map really is projective.
Fret *number* is not, which is why the numbering is undone only after the
projective step.

Detected wires stop where the evidence stops, so their endpoints are not
comparable between wires. Every numbered wire is therefore clipped to the neck
region, one straight edge is fitted per side of the board from all of those
chords, and each wire is re-cut between the two fitted edges. On held-out test
images this dropped the median fit residual from 0.106 to 0.015 board units.
A fit is rejected outright when either residual is too large.

A fingertip at fret coordinate 4.3 lies between wires 4 and 5 and therefore
sounds fret 5. Interpolated fret estimates are deliberately **not** used as
anchors: they are derived from the numbering fit, so they would add apparent
support without adding evidence.

**Strings are interpolated, not detected.** The model has no string head. Six
strings are spread evenly across the detected board with a margin set by
`--inset` (default 0.12 of board width). Errors here scale directly into string
errors, and `--inset` is worth tuning per instrument.

## Which end of the board is string 1

Geometry cannot tell which end of a fret wire carries the high E. `--string-order
auto` (the default) assigns the whole clip both ways and keeps the order that
explains more fretted notes with an actual fingertip; the chosen order and both
scores are printed and recorded. Force it with `--string-order normal|flipped`
when a clip is too short or too sparse for the evidence to separate them.

## How a note gets its position

Every pitch has several `(string, fret)` spellings in the tuning. Each is scored
from the fingertips and from where the hand is sitting on the neck, and one
placement is chosen per chord, not per frame, so the tab does not flicker. No
two simultaneous notes may take the same string, because one string sounds one
pitch; a note displaced by that rule is flagged `string-taken` rather than
quietly respelled.

An unoccupied string counts for less than a fingertip sitting on a cell.
A missing fingertip is only evidence of absence — it may be occluded or simply
missed — so positive evidence always outranks the lack of it.

Each note carries a `support` value:

| Support | Meaning |
|---|---|
| `fingered` | A fingertip sits on the chosen cell |
| `open` | Nothing sits on that string, and fret 0 produces the pitch |
| `position-only` | Nothing supports it; the neck position was the only evidence |
| `no-hand` / `no-board` | The hand or the board was unavailable at that onset |

Flags record the rest: `unsupported-placement` when the hand does not explain the
note at all, `string-taken`, `out-of-range` when the tuning cannot produce the
pitch, and `no-free-string`. Fingertips on the board that no sounding note
accounts for are reported as unexplained; they are usually a finger damping,
hovering or mid-shift, but they can also mean the string geometry is wrong.

## Tuning

`--tuning standard` is EADGBE. Otherwise pass MIDI pitches highest string first,
for example `--tuning 64,59,55,50,45,38` for drop D. Ascending order is rejected
because the string-number convention would silently invert the tab. Nothing
detects the actual tuning; a mistuned or capoed guitar produces confident and
wrong tab.

## What is not established

- No transcription accuracy has been measured. There is no annotated tab to
  measure against, and the fretboard model itself is not finished training.
- A fingertip near a cell is not proof the string is pressed, and the hand
  cannot distinguish a fretted note from a damped or hovering finger.
- Basic Pitch is general purpose. On harmonically rich plucks it reports
  harmonics as separate notes, which then compete for strings.
- Barre chords are not modelled: one finger covers several strings, but only its
  tip is located.
- Fret numbering needs the nut. Without it a board can still be tracked, but
  notes fall back to `position-only` or `no-board`.
- Bends, slides, hammer-ons, vibrato and capos are not modelled at all.
- The right hand is only used to pick the fretting hand; picking and strumming
  are not analysed.
