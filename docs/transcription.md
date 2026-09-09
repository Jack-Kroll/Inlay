# Tab transcription from a recording

Three independent signals are combined into tab: **sounding pitch** from audio,
**fret geometry** from the dense detector, and **fingertip position** from
MediaPipe. None of them alone is enough, and the combination has not been
measured against annotated playing. Treat the output as a diagnostic.

Install the extra and run it on a clip that has sound:

```sh
uv sync --locked --extra transcribe
uv run --extra transcribe python -m processing.music.transcribe \
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
Notes starting beyond the analyzed video are omitted, and notes crossing its
end are shortened to that boundary. Audio detection still reads the full soundtrack.
Short replacement audio is padded with silence to preserve every rendered frame.

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

String geometry now uses nut/fret heatmap endpoints directly, without clipping
wires to the segmented neck boundary. Each edge needs at least three consistent
endpoints; inward-truncated wires can be extended to those fitted edges.
The neck mask still guides detection orientation, search area, and tracking.
Consistently shortened detections remain ambiguous and can underestimate width.

A fingertip at fret coordinate 4.3 lies between wires 4 and 5 and therefore
sounds fret 5. Interpolated fret estimates are deliberately **not** used as
anchors: they are derived from the numbering fit, so they would add apparent
support without adding evidence.

**Strings are interpolated, not detected.** The model has no string head. Six
strings are spread evenly across the detected board with a margin set by
`--inset` (default 0.09 of board width). Errors here scale directly into string
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
This constraint includes notes still sounding from earlier onset groups. If
every candidate string is occupied, the new note stays unassigned with a
`no-free-string` flag. Visual evidence must be within 120 ms of the onset;
distant frames cannot supply a hand or board position.

An open string is never judged on how near the fingers are, since it needs no
hand; what it is worth is a fixed prior, set where the stage stops
systematically refusing to call anything open. Nothing here can actually tell an
open B from the same pitch fretted elsewhere.

An unoccupied string counts for less than a fingertip sitting on a cell.
A missing fingertip is only evidence of absence — it may be occluded or simply
missed — so positive evidence always outranks the lack of it. When *no*
fingertips are located at all there is no evidence either way, so the
open-string credit drops to zero and the neck position decides.

A note still ringing from an earlier onset does not block its string: plucking a
string again is what stops the note on it, so the earlier note is truncated to
the new onset and flagged `stopped-by-repluck`. Only notes sounded together
compete for a string.

The neck position is not decided chord by chord. `plan_positions` runs a
Viterbi pass over the whole clip before any note is placed: groups where the
hand was actually seen are pinned to it, and the rest are filled in from the
chords on either side and the cost of moving between them. A blind chord is
therefore answered by its neighbours in both directions rather than by a running
average of earlier guesses. See [tab logic](tab-logic.md).

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
The JSONL summary records these unexplained contacts at onset-group observations,
along with the support scores for each tested string order.

## Tuning

`--tuning standard` is EADGBE. Otherwise pass MIDI pitches highest string first,
for example `--tuning 64,59,55,50,45,38` for drop D. Ascending order is rejected
because the string-number convention would silently invert the tab. Nothing
detects the actual tuning; a mistuned or capoed guitar produces confident and
wrong tab.
The audio frequency bounds and the overlay's string count and labels follow
the configured tuning and maximum fret. One to twelve strings are supported.

## What is not established

- The audio stage reaches F1 .779 (precision .796, recall .762) on held-out
  GuitarSet excerpts — .868 on single-note solo lines, .742 on strummed comping.
  See [pitch tuning](pitch-tuning.md). A stricter opt-in implementation is
  described under Experimental finger-confirmed weak audio below.
- Given perfect pitch and no visible hand, the tab stage puts 69.8% of notes on
  the string actually played, but only 50.6% of notes that were played open;
  see [tab logic](tab-logic.md). That is the floor,
  not the pipeline: how much the fingertips add is unmeasured, because the fret
  geometry and the fingertip stage still have no ground truth, and the fretboard
  model itself is not finished training.
- Notes 60-100 ms apart are the weak spot: recall .514 and precision .493
  against .864/.834 for notes more than 400 ms apart. No Basic Pitch setting
  fixes it — `min_note_ms` has no effect below about 60 ms — so it is a limit of
  the model's onset resolution, not of the configuration.
- A fingertip near a cell is not proof the string is pressed, and the hand
  cannot distinguish a fretted note from a damped or hovering finger.
- Basic Pitch is general purpose. On harmonically rich plucks it reports
  harmonics as separate notes, which then compete for strings. Raising the
  thresholds cut that back — precision went from .670 to .796 — but a fifth of
  what it still reports was never played.
  It also loses quiet notes inside chords, and no threshold recovers them
  without admitting worse. The activations for those notes are still in the
  forward pass, though: `detect_notes` returns them as a `Posteriorgram`, and a
  stage that already knows which pitch to expect could confirm one at a
  threshold no global detector can afford. With a perfect fretting-hand tracker
  that would move recall .754 → .861, in the original oracle experiment: a
  tracker also proposes every finger that is down without being plucked, and
  one such finger per note already puts F0.5 below the baseline at every gate.
  See [pitch tuning](pitch-tuning.md). A stricter opt-in implementation is
  described under Experimental finger-confirmed weak audio below.
- Barre chords are not modelled: one finger covers several strings, but only its
  tip is located.
- Fret numbering needs the nut. Without it a board can still be tracked, but
  notes fall back to `position-only` or `no-board`.
- Bends, slides, hammer-ons, vibrato and capos are not modelled at all.
- The right hand is only used to pick the fretting hand; picking and strumming
  are not analysed.

## Live string-grid experiment

```sh
uv run python -m processing.vision.preview --model runs/dense/fretboard-v1/best.pt --source 0 --device mps --show-strings --show-heatmaps --inset 0.09
```

Yellow lines are six estimated string paths, drawn only over the numbered fret
range. They use the same transform as transcription. A waiting message means
there is insufficient consistent endpoint/numbering evidence. Mirroring affects
only display; JSONL `estimated_strings` stays in camera coordinates.
`--inset` is the fraction of width inside each edge, not a universal standard.
For example, Graph Tech's 43 mm / 35 mm E-to-E nut gives `(43-35)/(2*43)=0.093`:
https://graphtech.com/products/tusq-slotted-nut-43-x-6-pq-6143-00
Equal string-center spacing and a constant relative inset along the neck are
approximations; nut dimensions alone do not determine bridge spacing or taper.

## Experimental finger-confirmed weak audio

Add `--visual-rescue` to enable a conservative second pass after the baseline
string order and assignments are fixed. It finds actual local onset peaks at
score >= .45 with contiguous note energy >= .35 for at least 60 ms, then requires
an exact pitch/string/fret/finger match in at least three freshly detected frames
within 60 ms of the onset, covering at least 40 ms and two-thirds of that window.
Finger positions near string or fret boundaries are excluded. These thresholds
are experimental model scores, not calibrated confidence probabilities.

Existing notes are preserved. Duplicate/overlapping same-pitch candidates and
candidates conflicting with an assigned string are rejected. This intentionally
misses some re-plucks rather than modifying baseline durations. It cannot rescue
open notes or establish that a visible finger is pressing/plucking a string.

Added notes appear pink with `support: rescued`, `visual-audio-rescue` flags,
and audio/visual evidence in JSONL. The summary separates baseline count and
rescue counts. Use a separate output directory for comparison:

```sh
uv run --extra transcribe python -m processing.music.transcribe \
  --model runs/dense/fretboard-v1/best.pt --video clips/take-4.mov \
  --output runs/transcribe/take-4-rescue --device mps --mirror --visual-rescue
```

The prior oracle experiment in `pitch-tuning.md` found false positives from
unplucked fingers. This stricter temporal implementation is an opt-in experiment,
not evidence that the earlier precision problem is solved.
