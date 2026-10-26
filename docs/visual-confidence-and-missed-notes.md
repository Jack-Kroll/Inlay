# Visual confidence and missed-note experiments

Requested experiments: prioritize clear visual placements, and diagnose/recover
missed electric notes. Baseline: `2f48855`, which captures the main checkout's
previous fixes, including its independent open-string rival correction. No
model, global threshold, or global fusion weight is changed by default.
The duplicate-onset fix below is now enabled by default; visual priority and
strong-attack recovery remain optional.

## 1. Confidence-based visual priority

Added an optional gate that lets a specific stable visual placement outweigh
the audio string posterior. It requires:

- At least three agreeing frames, spanning both sides of the note onset.
- At least 80% agreement on the same string, fret and finger.
- Low fretboard-fit error, at least six measured anchors, and detected geometry.
- A clear geometric winner and limited finger motion.
- Agreement with the actual frame used for assignment.

Pitches with an open-string spelling retain the baseline policy: a visible
finger can hover while an equal-pitch open string rings. Evidence includes
the supporting frame count, contact strength, margin, and motion range.
This is a heuristic consistency gate, not a learned probability of correctness.

For the runtime gate the window is ±80 ms. Kaggle's sparse labelled stills use
±200 ms to supply neighboring samples; these are different sampling conditions.

Three policies were examined on development labels: stable, strict, and a
relaxed string-contact requirement. The stable policy remains the optional
runtime setting. The relaxed development variant was not promoted.

| Policy / split | Accepted visual placements | Correct against labels | Placements changed |
|---|---:|---:|---:|
| Stable, development | 178 | 178 | 0 |
| Strict, development | 62 | 62 | 0 |
| Relaxed, development | 425 | 425 | 0 |
| Stable, recording holdout | 39 | 39 | 0 |

**Result: no demonstrated placement-accuracy improvement.** The accepted
placements were already resolved correctly by existing geometry. Tightening
the gate reduced coverage; relaxing it increased coverage without improving
the scored assignments. Correctness on this small accepted subset does not
establish general reliability.

The diagnostic supplied pitch from each pressed-finger label and used the
real audio string posterior only when an actual same-pitch onset proposal was
within 50 ms. That occurred for 183 of 2,823 development labels and 95 of 1,573
holdout labels. The stable priority gate actually changed the audio weighting
for 8 and 1 labels respectively, without changing their assigned strings.
This is not a complete sounded-note transcription benchmark. Recording groups
share a performer/setup; the holdout has been inspected in the previous run
and is a regression split, not fresh independent evidence.

On six native video clips (367 baseline note events), the stable gate applied
to 29 notes and changed no placements. A more permissive visual recovery
prototype proposed one extra Thunderstruck note; it was rejected when the
joint chord assignment moved it off the visually supported string.

That case is also a useful follow-up: the 50 ms chord window can group two
rapid attacks that compete for the same visually indicated string. This run
adds a safety check to the recovery prototype; it does not claim that the
extra pitch was correct or change chord grouping without labelled validation.

## 2. Why electric notes are missed

P3 amp has 1,918 reference notes and 1,168 final matches: 750 misses.
The final misses partition as follows:

| Cause | Notes |
|---|---:|
| Nearby candidate scored below verifier threshold | 683 |
| No same-pitch raw proposal within 50 ms | 55 |
| Candidate lost in suppression | 1 |
| Refinement moved surviving proposal outside tolerance | 5 |
| One-to-one matching competition | 6 |

Raw proposals match 1,857 references one-to-one. Thresholded raw proposals
match 1,171; suppression leaves 1,170; refined predictions match 1,168.
These stage totals are not disjoint losses: refinement can gain some matches
while losing others. The separate table above partitions the *final* misses.

The median best score among the 683 threshold misses is .099; 539 score below
.5. Their median Basic Pitch onset activation is .643, but median note-frame
activation is only .323. This points to verifier transfer/context as the main
problem. Small timing/suppression changes cannot recover most of these misses.

## Strong-attack recovery

Tested a fixed audio rule: at the existing .75 operating point, a candidate
with verifier score at least .5, onset activation at least .75, and note-frame
activation at least .5 may enter the ordinary decoder. It generates no new
pitches and retains existing same-pitch suppression.

Recovery is an eligibility mask. The original verifier score still ranks
suppression candidates and is reported as confidence. Raising all recovered
scores to the threshold would incorrectly tie their suppression priority and
inflate the reported confidence; the implementation explicitly avoids that.

| Evaluation | Baseline note F1 | Recovery note F1 | Baseline tab F1 | Recovery tab F1 |
|---|---:|---:|---:|---:|
| P3 amp development | .7259 | .7396 | .4400 | .4503 |
| GuitarSet 04 development | .8092 | .8085 | .6785 | .6772 |
| GuitarSet 05 regression | .8933 | .8940 | .7675 | .7655 |
| IDMT regression | .8368 | .8389 | .4784 | .4792 |

P3 gains 49 matches with 73 additional predictions (24 additional unmatched
predictions). Precision changes .8985→.8864; recall .6090→.6345. IDMT gains
44 matches with 79 additional predictions. These are modest tradeoffs, and
acoustic tab F1 declines slightly, so recovery remains **off by default**.

GS05, P3, and IDMT are previously inspected corpora. No claim of a fresh
untouched test is made. IDMT is measurement-only and never enters training.
The tab baselines differ from the earlier campaign because this baseline
includes the main checkout's open-string correction; compare within this table.

Also tested disabling refinement, clipping its delta to two frames, a
strong-sustain recovery rule, and near-threshold recovery. Their results,
including negative outcomes, are retained in the compact JSON and raw reports.

## Frozen decision-head test

The existing feature extractor, BatchNorm, timing head and string head were
frozen. Reused 20,000 training-player positive embeddings (10,000 mic and
10,000 gt-amp), and extracted random/hard negative embeddings from the same
training players. Trained only the 128→1 existence head for ten epochs and
tested blends of .25, .5 and 1 with the original head on development data.

**Rejected.** The .25 blend improved GS04 F1 .8092→.8166 but lowered P3 amp
F1 .7259→.7160 and recall .6090→.5876. Larger blends lost more electric
recall. Mean development note F1 selected the original head (blend 0). No
head checkpoint is promoted, and no held-out corpus was used for gradients.

## Duplicate onsets after timing refinement: fixed by default

The relaxed visual-priority replay exposed two E3 predictions in Seven Nation
Army at 4.8549 and 4.8661 seconds, only 11.2 ms apart. Their raw peak times were
4.8439 and 4.9020 seconds, 58.1 ms apart, so the original suppression kept both.
The chord solver then had to place the duplicate on another string despite
the camera showing a stable A-string fret-7 placement. Relaxed visual priority
only swapped which duplicate took each string; it could not solve the cause.

Added a second suppression pass after onset refinement. It keeps the original
raw pass, uses the same nominal .05 s window (rounded to four audio frames,
about 46.4 ms), and retains the stronger original verifier score. This is
enabled by default as `verify.SUPPRESSION_MODE = 'both'`.

| Evaluation | Baseline note F1 | Fixed note F1 | Baseline tab F1 | Fixed tab F1 |
|---|---:|---:|---:|---:|
| P3 amp development | .7259 | .7314 | .4400 | .4475 |
| GuitarSet 04 development | .8092 | .8184 | .6785 | .6949 |
| GuitarSet 05 regression | .8933 | .9021 | .7675 | .7844 |
| IDMT regression | .8368 | .8503 | .4784 | .4971 |

The fix removes 27 / 245 / 179 / 172 predictions respectively, with only
1 / 3 / 2 / 3 fewer pitch/onset matches. It improves both note and tab F1 on
all four sets. Suppressing only on refined times was also tested; keeping
both passes was better on the development measurements. No threshold or
model was tuned on IDMT. Very fast repeated notes and unisons remain a
limitation; the few lost matches are recorded, not hidden.

Strong-attack recovery combined with this fix gives P3 note/tab F1
.7456/.4586 and IDMT .8528/.4984, but slightly lowers acoustic tab F1 versus
the duplicate fix alone. Recovery therefore remains optional.

## Native clip and software checks

Blackbird, Thunderstruck, Sweet Child O Mine, Loser, Seven Nation Army and
take-4 were evaluated with identical cached audio/vision for each variant.
The selected visual gate, audio recovery rule, and guarded visual-recovery
prototype retained all existing placements and open-string counts. There
were no retained extra notes in these clips. Thus there is **no demonstrated
improvement from those optional gates on the user's footage**.

The separate duplicate fix removes ten predictions across the six clips:

| Clip | Before notes | After notes | Before open | After open |
|---|---:|---:|---:|---:|
| Blackbird | 39 | 38 | 12 | 12 |
| Loser | 29 | 29 | 0 | 0 |
| Seven Nation Army | 58 | 53 | 8 | 7 |
| Sweet Child O Mine | 65 | 64 | 0 | 0 |
| take-4 | 79 | 78 | 24 | 24 |
| Thunderstruck | 97 | 95 | 36 | 35 |

Three retained Seven Nation Army placements and one Thunderstruck placement
also change as the duplicates stop competing for strings. This is not an
independent full-tab accuracy score. A fresh complete Seven Nation Army run
and overlay render completed with 53 notes and the new suppression metadata.

Cached replay holds baseline string orientation fixed to isolate changes.
An actual Loser CLI run with both options and automatic string orientation
also completed: 388 frames, 29 notes, normal string order, 27 fingered and
2 position-only placements. Native clips lack independent reference tabs;
support and unchanged outputs are regression evidence, not accuracy scores.

Optional commands:

```sh
python -m processing.music.transcribe --video "clips/Loser.mov" --visual-priority
python -m processing.music.transcribe --video "clips/Loser.mov" --recover-strong-attacks
```

These options are independent: visual priority affects string assignment;
strong-attack recovery uses audio support. Combining the CLI flags does not
require visual confirmation of each recovered note. The guarded visual
recovery experiment lives in `processing.tools.eval_confidence_clips`.

Per-note miss traces, model/audio fingerprints, gate evidence, rejected
experiments and reusable caches are saved in `runs/confidence-2026-10-19/`.
The compact tracked summary is `docs/confidence-results-2026-10-19.json`.
Validation: 221 unit/integration tests passed and Pyright reported zero errors.
