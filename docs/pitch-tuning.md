# Audio pitch tuning

This document measures only the audio stage. It does not measure fretboard,
hand, string assignment, or full transcription accuracy.

## Dataset and scoring

GuitarSet contains 360 thirty-second acoustic-guitar recordings from six
players, with 62,476 annotated notes and played-string labels. Inlay uses the
microphone channel because it resembles a phone recording.

A note matches when its onset is within 50 ms and pitch within 50 cents. Offsets
are ignored. Results are micro-averaged so each note has equal weight.

The report includes F1 and F0.5. F0.5 values precision more because an invented
note can take a string away from a real note in the tab stage.

## Run a sweep

```sh
uv run --extra transcribe python -m processing.tools.tune_pitch \
  --limit 48 --output runs/tune-pitch/coarse.json --progress
```

Basic Pitch runs once per excerpt. Candidate thresholds are then decoded from
the same cached posteriorgram. Caches live under
`data/guitarset/posteriorgrams`; use `--cache-dtype float32` when exact
reproduction matters.

`--limit` selects a balanced subset across player and style. `--limit 0` uses
all 360 excerpts.

## Selected thresholds

The defaults are:

```text
onset threshold: 0.60
frame threshold: 0.45
minimum note:    60 ms
```

They were tuned on 48 excerpts and checked on the remaining 312 excerpts
(54,765 notes).

| Setting | F1 | F0.5 | Precision | Recall |
|---|---:|---:|---:|---:|
| Current defaults | .779 | .789 | .796 | .762 |
| Precision-first | .777 | .811 | .836 | .726 |
| Old defaults | .740 | .696 | .670 | .827 |

The current setting trades some recall for much better precision. The best
region was a plateau, so the chosen values performed well on both tuning and
held-out data instead of maximizing one subset.

| Style | F1 | Precision | Recall |
|---|---:|---:|---:|
| Solo | .868 | .828 | .911 |
| Comping | .742 | .782 | .706 |

Chords are harder because quiet notes are masked by stronger simultaneous
partials.

## Quiet and rapid notes

The audio path extracts mono 22,050 Hz WAV. It does not apply compression,
denoising, EQ, gating, or gain normalization.

For an experiment, try:

```sh
--onset-threshold 0.5 --frame-threshold 0.35 --min-note-ms 40
```

Lower thresholds can recover weak events but also add harmonics and fragmented
notes. Notes only 60-100 ms apart remain a major weakness of the model rather
than a simple threshold problem.

## Visual rescue experiment

The raw posteriorgram allows vision to ask whether weak audio evidence exists
for one expected pitch without a second model run.

An oracle experiment showed that perfect visual candidates could improve
recall. The result stopped being attractive once realistic non-sounding fingers
were included: held, damping, hovering, and anchoring fingers created false
notes, pushing F0.5 below the baseline.

The runtime therefore keeps visual rescue conservative and opt-in. It requires
stable exact-cell fingertip evidence plus weak audio, preserves baseline notes,
and labels additions as rescued. It cannot rescue open strings or prove that a
visible finger caused a sound.

Use the bound experiment to revisit the decision:

```sh
uv run --extra transcribe python -m processing.tools.rescue_bound \
  --limit 48 --visual-error 0.10 --output runs/rescue-bound/oracle.json
```

## Limits of these numbers

GuitarSet is acoustic, standard-tuned, and microphone-recorded. The 312-excerpt
set helped choose the final default, so it is validation evidence rather than a
pristine final test. These results say nothing about camera geometry or complete
tab accuracy.
