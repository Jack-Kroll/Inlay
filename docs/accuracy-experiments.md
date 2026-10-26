# Accuracy experiments, 2026-10-19

Baseline: commit `e629727`, v8-gt verifier, threshold .75, audio string
weight .25, open-audio neutral 1. The main checkout has independent,
uncommitted changes to open-string fusion; this work uses an isolated
checkout and does not overwrite them.

The user reports that hand placement and fret recognition are usually
accurate on their clips; visual string coordinates are less certain.
The priority is to use reliable fret evidence better and measure gains
on existing public annotations, without asking the user to label data.
Research-only datasets are acceptable for this campaign. Preserve
attribution and distinguish their terms from the repository's MIT license.

## Evaluation decisions made before looking at scores

- Kaggle Guitar Transcription Dataset: audit labels, timestamps and source
  recordings first. Partition by recording, never random neighboring frames.
  Frame finger placements measure visual placement; they do not by themselves
  prove which held notes were sounding. Label any oracle-pitch diagnostic.
- GuiFiR: preserve the published validation/test distinction and inspect
  whether manual labels and pseudo labels are separable. Keep source-video
  overlap explicit. Do not use pseudo labels as final correctness evidence.
- GuitarSet player 04 and Guitar-TECHS P3 are development evidence. P3 has
  already selected configurations. GuitarSet 05 and IDMT have also been
  repeatedly reported; any new measurement is a regression check, not a fresh
  one-shot generalization test. IDMT stays measurement-only.
- User clips are qualitative/regression evidence without reference tabs.
  Fingertip support is not an independent ground-truth accuracy metric.
- Fix candidate variants on development data before final test reporting.
  Use the same note events when isolating string assignment.

## Completed visual comparisons

Downloaded the [Kaggle Guitar Transcription Dataset](https://www.kaggle.com/datasets/jacksonlightfoot/guitar-transcription-dataset)
and [GuiFiR](https://github.com/kmu-ee-ailab/guifir_dataset). Kaggle's evaluated
tablature subset remains locally under `data/visual-benchmarks/kaggle/`.
The unevaluated GuiFiR archive and Kaggle fretboard subset were removed locally
on October 21 to recover disk space; they can be downloaded again from their
sources. These assets stay outside Git and retain their research/noncommercial terms.

Kaggle contains 1,995 labelled images across 25 recordings and 2,181 timestamp
rows. There are 186 extra timestamp rows, but no missing images among the
labelled frame IDs. Join by explicit `recording_frame` ID; do not enumerate
timestamp rows and assume they align with labels. Press class 1 corresponds to
individual fingers; class 2 is a barre without a single target string and is
excluded from this diagnostic.

The fixed held-out recording IDs were 6–8, 15–17, and 21–24. Remaining
recordings were development data. These share a performer/setup and are **not**
an independent-guitar or independent-performer benchmark. Higher frets are
overrepresented in the held-out group. The visual detector was trained on the
B101 export; source-video overlap with these public images is not fully audited.

The test supplies the pitch of each labelled pressed finger, then asks fusion
to recover its string. It evaluates one labelled placement at a time, not
sounding-note detection, chord voicing, or a complete transcription.

| Variant | Development placement | Held-out placement |
|---|---:|---:|
| Existing geometry | 79.49% | 67.26% |
| Fret-only mixture .75 | 80.06% | 68.15% |
| Stable neighboring predictions, ±.2 s | 79.70% | 67.58% |
| Combined | Not used for selection | 68.40% |

Development: 1,235 frames / 2,823 individual finger labels. Held-out: 760
frames / 1,573 individual finger labels. Tested fret mixtures 0, .25, .5, .75,
1 on development only; chose .75 before inspecting the held-out scores.
Temporal pooling requires agreeing bracketing predictions, does not cross
recordings, and refuses changes greater than .75 fret or one string.

The initial low-fret frames had excellent fret localization. Later high-fret
frames often had a +1 fret error and sometimes no usable board transform.
The full dataset does not support treating every visual fret estimate as fact.

**Rejected as a runtime default:** the fret-only mixture. Replaying identical
saved audio notes with new visual inference gave:

| Clip | Notes | Baseline open placements | Fret-only open placements | Changed placements |
|---|---:|---:|---:|---:|
| Blackbird | 39 | 13 | 11 | 4 |
| Thunderstruck | 97 | 36 | 8 | 34 |
| Sweet Child O Mine | 65 | 0 | 0 | 2 |

Thunderstruck's fingertip-support fraction fell from .721 to .449. Support is
not ground truth, but losing the known open-string riff is a strong regression
signal. Fret agreement alone lets unrelated fingers elsewhere across the board
explain an open pitch. The experimental API remains at weight **0** by default;
it is not exposed as a recommended transcription CLI setting.

The replay uses the same three-decimal audio posteriors saved in existing
JSONL, fixed baseline string orientation, and identical notes for both variants.
It does not rerun the audio model or score independent reference tabs.

## Audio threshold and suppression comparisons

Model: v8-gt; candidate file: `candidates-gt2.npz`. Original candidate proposals
had one-to-one 50 ms pitch/onset coverage of 1,857/1,918 = **96.82%** on
Guitar-TECHS P3 amp, and 9,884/10,253 = **96.40%** on GuitarSet 04. These
are proposal-coverage diagnostics, not achievable end-to-end recall bounds:
onset refinement can move proposals and verification rejects many.

| Setting | GS04 note F1 | GS04 tab F1 | P3 amp note F1 | P3 amp tab F1 |
|---|---:|---:|---:|---:|
| Existing .75 threshold / 50 ms suppression | .8092 | .6821 | .7259 | .4549 |
| .50 threshold / 50 ms suppression | .7651 | .6316 | .7345 | .4617 |
| .75 threshold / 25 ms suppression | .7941 | .6576 | .7198 | .4454 |

Also tested thresholds .65 and .85 with both suppression windows. A lower
global threshold trades acoustic precision for electric recall. Shorter
suppression loses on both domains. **Neither default changes.**

The electric gap between 96.8% raw proposal coverage and 60.9% final recall
points toward verification/timing decisions, not simply generating more
candidates. A future recovery rule should require both local acoustic support
and reliable visual evidence, with open-string regression tests.

## Frozen string-head calibration

Trained only the existing 128→6 linear string head, keeping the feature
extractor, BatchNorm state, note-existence head and onset head frozen.
Sampled 10,000 positive candidates per domain (mic, gt-di, gt-amp, gt-ego,
gt-exo), 50,000 total, from GuitarSet players 00–03 and Guitar-TECHS P1/P2.
Seed 42; 20 epochs; batch 1,024; AdamW learning rate .001, weight decay .01.
Started from the shipped string-head weights.

Two variants: equal domain weighting, and quarter weighting for the room
microphones. Equal weighting won mean development tab F1 on GS04 and P3 amp.
P3 was not used for gradient updates. IDMT was never used for training.

| Corpus | Existing string accuracy | Frozen-head string accuracy | Existing tab F1 | Frozen-head tab F1 |
|---|---:|---:|---:|---:|
| GS04 development | .8430 | .8473 | .6821 | .6856 |
| P3 amp development | .6267 | .6318 | .4549 | .4587 |
| GS05 regression | .8654 | .8708 | .7730 | .7778 |
| IDMT regression | .5860 | .5978 | .4904 | .5002 |

GS05 gained 43 correctly assigned matched notes; IDMT gained 51. Note F1,
note counts and pitch/onset matches stayed unchanged. GS05 baseline was
rerun under the same settings; IDMT baseline comes from the existing
`v8-gt-idmt-final.json` with matching threshold/neutrality and counts.
These are previously inspected regression corpora, not untouched new tests.

Fresh audio passes on three user clips asserted bit-identical pitch, start,
end and confidence arrays between the two checkpoints. Thunderstruck kept all
97 note placements and 36 open placements. Blackbird changed one G3 from
A-string fret 10 to open G (support fraction .6538→.6400); Sweet Child changed
one C#5 from B-string fret 14 to high-E fret 9 (support fraction unchanged).
Those individual string changes lack independent reference labels, so the
checkpoint is retained as an experimental option rather than installed as the
default: `runs/accuracy-2026-10-19/string-probe/probe.pt`.

This supports separating string calibration from note detection. It does not
demonstrate a large end-to-end improvement or justify changing global audio
weights. A completely separate string feature extractor and paired-chain
consistency training have not been tested in this run.

## GuiFiR readiness

The archive contains 900 train, 30 validation, and 100 test clips, with
29,051 / 543 / 2,048 annotated frames respectively. Validation and test rows
contain manual pose fields; training rows do not. Training mixes pseudo and
manual tablature according to the paper; the release rows lack a per-row
provenance flag sufficient to separate those automatically.

All 13 validation source-name prefixes also occur in training; 21 of the 32
test prefixes occur in training. These are filename-based overlap checks, not
verified performer identities. Preserve the published split for reproduction,
and add a source-disjoint split before claiming generalization.

The released annotation names omit the `_v5` / `_final_100` suffixes advertised
in the README. Some 1920×1080 test frames have pose points apparently expressed
on a smaller canvas; bounding boxes remain in the declared frame coordinates.
Coordinate scaling, capo convention, and string order need verification before
training or scoring against those rows. No GuiFiR accuracy result is claimed.

## Fixes and reproducibility

- Evaluator preserves distinct unison identities instead of collapsing
  `(pitch, onset)` dictionary keys.
- Chord assignment now actually honors amplitude when equal-scoring notes
  compete for a string, independent of input order.
- Invalid hand-position anchors no longer invalidate the entire position path;
  a mismatched anchor/group count gets an explicit error.
- An onset with a missing hand can use a board-and-hand frame within 40 ms.
  It retains the closest frame when that frame already has a hand and never
  borrows a distant hand just because the board search permits 120 ms.
  The three replayed clips have no such onset dropout and retain all their
  baseline placements; this is a regression check, not a measured accuracy gain.
- Evaluator defaults follow the shipped verifier and threshold. Explicit
  `--open-neutral 0` is honored (also fixed independently in the main checkout).
- Transcription rejects out-of-range/nonfinite audio fusion weights before
  running models.
- The media wrapper now translates Basic Pitch's exclusive upper-frequency
  cutoff into the verifier's inclusive MIDI bound. Previously it admitted one
  extra semitone beyond the requested guitar range.

Tools:

```sh
python -m processing.tools.eval_visual_contacts --extract \
  --root data/visual-benchmarks/kaggle/tablature_dataset/tablature_dataset \
  --cache runs/accuracy-2026-10-19/kaggle-contacts.jsonl
python -m processing.tools.audit_verifier --extract --domain gt-amp --players P3 \
  --cache runs/accuracy-2026-10-19/gt-amp-scores \
  --output runs/accuracy-2026-10-19/gt-amp-audit.json
python -m processing.tools.replay_fusion --help
python -m processing.tools.train_string_probe --help
```

Full raw reports and reusable inference caches live in
`runs/accuracy-2026-10-19/`. Source assets live in `data/visual-benchmarks/`.
Compact results are tracked in `docs/accuracy-results-2026-10-19.json`.
Experimental defaults stay separate from the shipped configuration. No
checkpoint has been replaced by this work.

Integrated into the main working checkout while preserving its independent
open-string rival correction and evaluator edits. The measurements above use
the explicitly named `e629727` baseline, not the combined uncommitted checkout.
Validation after integration: **208 tests passed**, **0 Pyright errors**,
and a clean whitespace check. The isolated experiment checkout passed 207
tests; the additional test was already present in the main checkout's work.

Paired-chain consistency training, learned visual confidence calibration,
and a new string/fret sequence decoder remain follow-up experiments. They are
not claimed as implemented or measured here.
