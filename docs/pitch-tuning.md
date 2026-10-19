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

## Rejected: visual rescue

The raw posteriorgram lets vision ask whether weak audio evidence exists for
one expected pitch without a second model run. An oracle experiment showed
that perfect visual candidates could improve recall, so the runtime carried a
conservative, opt-in `--visual-rescue` that required stable exact-cell
fingertip evidence plus weak audio.

It was removed. Two measurements killed it:

- With realistic non-sounding fingers included, it stopped being attractive at
  all. Held, damping, hovering and anchoring fingers created false notes and
  pushed F0.5 below the baseline. A visible finger never proved a string
  sounded, and it could not rescue open strings in principle.
- The problem it existed to solve was a decoding failure, not a missing-evidence
  one. The [note verifier](note-verifier.md) recovers those notes from the
  audio alone — recall .834 to .894 with precision *rising* — so rescue became
  a worse answer to a question that was no longer open.

The code, its oracle-bound tool and its tests are gone; the finding is this
section. Anything that revisits vision-proposed notes should start from the
verifier's candidate set rather than from a second threshold.

## Limits of these numbers

GuitarSet is acoustic, standard-tuned, and microphone-recorded. The 312-excerpt
set helped choose the final default, so it is validation evidence rather than a
pristine final test. These results say nothing about camera geometry or complete
tab accuracy.
