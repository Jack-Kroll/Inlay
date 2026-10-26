# Fine-tuning the pitch front end — experiment scope

Status: run. What was built, what had to change and the results are in
[the write-up](front-end-finetune.md). The plan is kept as written.

Every audio decision Inlay makes starts from a posteriorgram produced by the
2022 Basic Pitch checkpoint, frozen. The verifier, its string head and the
augmentation work in [the note verifier](note-verifier.md) all learned to read
that fixed output; none of them could change what it contains. This experiment
fine-tunes the front end itself on guitar, keeps its output format, retrains
the verifier on what it now produces, and asks whether the pair beats v8-gt.

## Why the front end

- **It has never heard an electric guitar on purpose.** Basic Pitch trained on
  GuitarSet, MAESTRO, MedleyDB-Pitch, iKala and Slakh. Every electric failure
  the verifier docs record — distorted P3 precision `.898` → `.517`, the string
  head at `.56`–`.63` on held-out electric — is downstream of features tuned for
  piano, voice and an acoustic guitar.
- **The string head reads timbre through it.** The head reaches `.858` from the
  posteriorgram alone. Those cues were never a training objective of the front
  end; fine-tuning on string-diverse audio may sharpen them even with no string
  loss, and a later stage can add one.
- **Swapping the whole model did not work.** YourMT3+ gained 4 points on
  GuitarSet and lost 13 on IDMT
  ([pretrained-audio benchmarks](pretrained-audio-benchmarks.md)). Fine-tuning
  the model the rest of the pipeline already fits is the conservative version
  of that idea.

**Front-end contamination.** Basic Pitch split GuitarSet by shuffling tracks
(`basic_pitch/data/datasets/guitarset.py`), not by player. The shipped front
end has almost certainly heard players 04 and 05 already, so today's GS05
figures carry an overlap the verifier's own split cannot remove. Fine-tuning
does not undo that. IDMT and Guitar-TECHS P3 are the clean comparisons.

## What changes and what does not

The fine-tuned model keeps Basic Pitch's outputs exactly: `note` and `onset`
at (frames, 88), `contour` at (frames, 264), 22,050 Hz input, hop 256. So
`verify.prepare`, peak-picking, `decode`, fusion and every eval tool read it
unchanged. What changes is which model fills the caches, and the verifier
retrained on those caches.

The Basic Pitch threshold fallback in `detect_notes` stays on the original
Core ML model. It is the no-verifier path and is not part of this experiment.

## Stage 0: a PyTorch Basic Pitch that matches the shipped one

The pip package ships Keras training code but runs inference through Core ML;
TensorFlow is not installed, and the rest of Inlay is PyTorch on MPS. Port the
model (harmonic CQT, harmonic stacking, three conv stacks; under 20k weights)
to PyTorch in `processing/music/frontend.py`. Load the shipped weights from
`saved_models/icassp_2022/nmp.onnx` (via `onnx`, as a dev-only dependency) or
from `nmp.mlpackage` via `coremltools`, which is already installed.

Gate, before any training:

- On ten GuitarSet player-04 excerpts, the ported model's three heads match
  `run_inference` to within `1e-3` max absolute difference, at every frame.
- Player-04 `eval_verifier --split val` on posteriorgrams cached from the
  port reproduces the Core ML figures to the third decimal place.

If the port cannot match, stop. A training result on a different front end
would not be comparable to anything already measured.

### Plumbing

- `pitch.raw_activations(audio, front_end=None)`: `None` is the shipped Core
  ML model, unchanged; a path loads a `frontend.py` checkpoint.
- Every place that runs Basic Pitch to fill a cache gets `--front-end PATH`:
  `tune_pitch`, `variants` (which `guitar_techs` renders through), `idmt`,
  `cache_audio_candidates`, `compare_string_probe`. A non-default front end
  writes to a sibling cache named after it, e.g.
  `data/guitarset/posteriorgrams@ft1/`, and never overwrites `posteriorgrams/`.
- `eval_verifier` reads the fixed `CACHE` constant; it gains `--cache`, as
  `eval_idmt` already has.
- The verifier checkpoint records the front end it was trained on.
  `verify.load` and `transcribe` refuse to pair a verifier with a different
  front end. A posteriorgram from the wrong model is input the verifier has
  never seen, the same failure `raw_activations` already guards against for
  masked columns.

## Data

Only sources whose licences already allow training and shipping weights.
Fine-tuned Basic Pitch weights stay under its Apache-2.0 licence.

| Source | Train | Labels used |
|---|---|---|
| GuitarSet mic | players 00–03, about 2 h | onset, note, contour from `pitch_contour` |
| GuitarSet pickup mix | the same takes | same |
| Guitar-TECHS DI, amp mic | P1, P2, about 3.5 h | onset, note; contour from the frozen teacher |
| Guitar-TECHS head and room mics | the same takes | note only; onset masked |

- **GuitarSet contours.** `guitarset.py` reads only `note_midi`. The JAMS also
  carry per-string `pitch_contour`; rendering contour targets from it at three
  bins per semitone with Basic Pitch's Gaussian blur is the one new label
  loader.
- **Guitar-TECHS has no f0 track**, only MIDI. Its contour target is the
  frozen original model's own contour output, as distillation. That anchors
  the head the string cues live in rather than leaving it free to drift on a
  third of the data.
- **Room-mic onsets are loose**, 83% within 50 ms. As in the verifier, they
  teach what sounds, not when: note loss only.
- **The pickup mix is training data here.** For the verifier it is the cheap
  held-out-chain proxy. Train on players 00–03's pickup only; player 04's
  pickup stays a proxy.
- **Not used:** IDMT (measurement only, always); hex-pickup remixes (they cost
  the verifier's note F1 `.817` → `.788` by removing body resonance, and a
  front end would learn the same thing harder); neck-shift resampling (moves
  resonances with pitch). GOAT, EGDB and SynthTab still lack a usable licence
  per the [dataset review](dataset-review.md).

## Held-out splits

The verifier's split is kept as is (`verifier_data.TRAIN_PLAYERS`), so the
front end and the verifier never disagree about who is held out.

| Role | Data | Used for |
|---|---|---|
| Train | GS 00–03, GT P1–P2 | gradients |
| Validate | GS 04 mic and pickup, GT P3 amp mic | early stopping, checkpoint and threshold choice, the two ablations below |
| Test | GS 05 mic, GT P3 DI and room mics | once per final candidate |
| Generalisation | IDMT, 657 recordings | once, for the one candidate that passes validation |

P3's amp mic is already the electric validation set for v8-gt, so treating it
as validation here costs nothing new. The verifier doc alone counts seventeen
configurations scored on IDMT, before the pretrained-model benchmark added
more. This experiment adds exactly one.

## Augmentation

Reuse `processing.training.augment.chain` unchanged — level, tilt,
resonances, a short cabinet impulse, saturation, hiss, MP3 round-trip — drawn
on the fly per 2-second window instead of pre-rendered. A front end trains
on audio directly, so there is no cache to keep in sync. Its existing
guarantees carry over: no energy before an attack, no pitch or time changes,
the codec delay cross-correlated back.

Mix: 50% clean, 50% through a chain at strength 1. The verifier found a harder
chain helped the pickup proxy and hurt IDMT, so strength is not swept.

`augment.amp` (re-amping through a distorted amp) is the one augmentation run
as an ablation rather than by default. For the verifier it recovered distorted
P3 and cost GuitarSet 4 points. Whether the front end makes the same trade is
one of the questions this experiment can answer, and it is decided on
validation only.

## Training

- Initialise from the shipped weights. Fine-tune everything except the CQT
  kernels, which are fixed DSP.
- Basic Pitch's own losses, ported from `basic_pitch/models.py`: weighted BCE
  on onset, BCE on note, label-smoothed BCE on contour, with masks per source
  as in the data table.
- Start with Adam at `1e-4`, 2-second windows, batch 16, MPS. Sample GuitarSet
  and Guitar-TECHS 1:1 per batch. Guitar-TECHS is drills, and 29% of its notes
  are above fret 12, so it must not dominate.
- Select the checkpoint on validation onset AP, averaged over GS 04 mic, GS 04
  pickup and P3 amp mic. GS 04 note F1 through the unchanged threshold decoder
  serves as a regression guard: if it falls more than 1 point, stop.

Two ablations, both decided on validation: `+ augment.amp`, and
`GuitarSet only`. The second shows how much of any gain needs the electric data.

## Retraining the verifier

For the selected front end, rebuild every cache it feeds, then rerun the
v8-gt recipe word for word under a new name:

```sh
uv run python -m processing.tools.tune_pitch --front-end runs/frontend/ft1/best.pt
uv run python -m processing.tools.variants --variants 3 --front-end runs/frontend/ft1/best.pt
uv run python -m processing.tools.guitar_techs --prepare --front-end runs/frontend/ft1/best.pt
uv run python -m processing.training.verifier_data \
  --domains aug-v1 aug-v2 aug-v3 --measure-domains gt-di gt-amp gt-ego gt-exo \
  --output data/verifier/candidates@ft1.npz
uv run python -m processing.training.train_verifier \
  --candidates data/verifier/candidates@ft1.npz \
  --cqt data/guitarset/cqt --planes data/verifier/planes@ft1 \
  --output runs/verifier/v9-ft1 --epochs 14 \
  --train-domains mic aug-v1 aug-v2 aug-v3 gt-di gt-amp gt-ego gt-exo \
  --string-only-domains gt-ego gt-exo --epoch-positives 46656
```

The clean CQT cache is reused, since it is computed from audio. Same epochs,
same positives per epoch, same domains: only the front end differs, which is
the comparison the
[what each source was worth](note-verifier.md#what-each-source-of-data-was-worth)
table set up.

Re-choose the score threshold and audio string weight on player 04 and P3, as
v8-gt's were. Do not carry `.75` and `.25` over blind: a sharper posteriorgram
shifts the verifier's calibration.

The candidate peak threshold `.15` should be re-checked too. Report the recall
ceiling and candidates per real note on player 04, as the
[verifier table](note-verifier.md#the-model) does. If the ceiling drops below
`.98`, lower the threshold until it does not.

## Measurement and acceptance

All through the existing tools, pointed at the new caches and checkpoint:

```sh
V="--model runs/verifier/v9-ft1/best.pt --candidates data/verifier/candidates@ft1.npz \
   --cache data/guitarset/posteriorgrams@ft1"
uv run python -m processing.tools.eval_verifier --split val $V
uv run python -m processing.tools.eval_verifier --split val --domain pickup $V
uv run python -m processing.tools.eval_verifier --split test --players P3 --domain gt-amp $V
# once, for the final candidate:
uv run python -m processing.tools.eval_verifier --split test $V
for chain in gt-di gt-ego gt-exo; do
  uv run python -m processing.tools.eval_verifier --split test --players P3 --domain $chain $V
done
uv run python -m processing.tools.idmt --prepare --front-end runs/frontend/ft1/best.pt
uv run python -m processing.tools.eval_idmt --cache data/idmt-smt-guitar/posteriorgrams@ft1 \
  --model runs/verifier/v9-ft1/best.pt
```

The scoring itself needs no changes, because the tools read the decode from
`processing.music.verify`.

Declared now, before anything trains. v9-ft1 replaces v8-gt only if all hold:

| Measure | v8-gt | v9-ft1 must |
|---|---:|---|
| IDMT note F1 | `.850` | be ≥ `.850` |
| IDMT tab F1 | `.497` | be ≥ `.497` |
| IDMT note F1 or tab F1 | — | rise ≥ 1 point on at least one |
| GS 05 note F1 | `.902` | be ≥ `.897` |
| GS 05 tab F1 | `.784` | be ≥ `.779` |
| P3 amp mic note F1 | `.731` | be ≥ `.731` |

Reported, not gated: precision and recall separately, open vs fretted, IDMT
per subset (`dataset3` especially), string accuracy by fret bucket, distorted
P3 precision (`.517`), the pickup-proxy gap, front-end inference time per
thirty seconds against Core ML's `0.08 s`, and the Thunderstruck open-B and
Seven Nation Army G3 placements from cached vision via `replay_fusion`, which
are qualitative and not scored.

If it fails, the result is still written up here with the same tables, as
every rejected idea in the verifier doc is.

## Stage 2, only if stage 1 moves notes and not strings

Add a string head to the front end: 6 × 88 logits per frame from the shared
trunk, trained on GuitarSet and Guitar-TECHS string labels. That is the
TabCNN/FretNet idea on top of Basic Pitch features. Its output joins the
verifier patch as extra channels. This changes the verifier's input shape, so
it is a separate experiment with its own acceptance table. It is not part of
this one.

## Risks

- **Forgetting.** The original model's breadth may be part of why it
  transfers to IDMT better than YourMT3+ did. The contour distillation, the
  1:1 sampling and the GS 04 regression guard are aimed at this. The
  `GuitarSet only` ablation shows whether the electric data helps or hurts.
- **Small data.** About 5.5 hours of distinct performances from six
  players, however many chains each is heard through. A front end with under 20k weights is
  probably better suited to that than a bigger model, which is why this
  fine-tunes Basic Pitch rather than training a transformer.
- **No help with the open-string problem.** Through a room, the head cannot
  hear an open string, and nothing here adds open-string audio. Guitar-TECHS
  is 0.6% open.
- **Distortion.** If the `augment.amp` ablation repeats the verifier's trade,
  high gain stays unsolved.

## Cost

On the M5: the port and its parity tests are about a day. Front-end training
is an estimated 2–4 hours per run, for the main run and two ablations.
Rebuilding the caches for one front end costs about what the current ones
did, and retraining the verifier is one v8-gt run. Disk: one more copy of the
posteriorgram caches and planes, on the order of `data/verifier`'s 24 GB, so
delete one ablation's caches before building the next.
