# Audio pitch tuning

This document measures only the audio stage. It does not measure fretboard,
hand, string assignment, or full transcription accuracy.

The thresholds below are no longer how inlay decodes notes by default. They
were the best a threshold could do, and measuring why they plateaued led to
the [note verifier](note-verifier.md), which replaces them: note F1 on a
held-out GuitarSet player went .821 -> .898. The thresholds remain the
fallback when no verifier checkpoint is present, and `--no-verifier` selects
them. This page is the record of how they were chosen.

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
uv run python -m processing.tools.tune_pitch \
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

This turned out to understate the case. Of the notes these thresholds miss,
only 0.1% have no onset activation at all: 80% are above .30 and 22% are above
the .60 threshold that was meant to accept them, failing instead on the frame
gate. Their median frame activation is .413, just under the .45 cutoff. The
evidence was in the posteriorgram and the decoder was discarding it. See the
[note verifier](note-verifier.md) for the full diagnostic and what it led to.

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
uv run python -m processing.tools.rescue_bound \
  --limit 48 --visual-error 0.10 --output runs/rescue-bound/oracle.json
```

## Limits of these numbers

GuitarSet is acoustic, standard-tuned, and microphone-recorded. The 312-excerpt
set helped choose the final default, so it is validation evidence rather than a
pristine final test. These results say nothing about camera geometry or complete
tab accuracy.
