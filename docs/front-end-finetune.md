# Fine-tuning the pitch front end

What was built and measured, against [the plan](front-end-finetune-plan.md).

**Result.** Basic Pitch, fine-tuned on guitar in PyTorch with EGFxSet added,
and the verifier retrained on it, passes every condition declared before
training and is now the default. On IDMT's electrics: note F1 .850 → .853,
string .585 → .597, tab F1 .497 → .509. On GuitarSet's test player: note F1
level at .902, tab F1 .784 → .786. On Guitar-TECHS's held-out electric: note F1
.731 → .749. Modest, and largest where it was aimed, on hearing notes on an
electric. Getting there needed three facts about how the release was trained
that its code does not state; they are below, because any further fine-tune
needs them.

## What was built

- `processing/music/frontend.py`: Basic Pitch's network in PyTorch, layer for
  layer, with its CQT, harmonic stacking and three heads, 16,782 weights. It
  loads the released Keras weights through
  `processing.tools.convert_basic_pitch`, which runs once in a throwaway
  TensorFlow environment.
- `processing/training/frontend_data.py` and `train_frontend.py`: training
  windows and targets, and the fine-tune.
- `processing/tools/frontend_caches.py`: renders every posteriorgram the
  verifier reads from a given front end, into `posteriorgrams@<tag>` beside
  the existing caches.
- `processing/tools/egfxset.py`: EGFxSet's single notes as verifier excerpts,
  the `egfx` domain.
- A verifier checkpoint records its front end (`train_verifier --front-end`).
  `pitch.notes_from_media`, `eval_verifier` and `eval_idmt` all use the front
  end the verifier names, so a verifier is never fed another model's
  posteriorgram.

## Stage 0: the port matches

On three GuitarSet player-04 excerpts the port's three heads match
`run_inference` through Core ML to a largest difference of 2.6e-5 on CPU and
1.5e-6 on MPS, against the plan's 1e-3 gate. It runs a thirty-second excerpt
in 0.11 s on MPS. `tests/test_frontend.py` checks the match on a synthetic
recording whenever the converted weights are present.

## Three things the release was trained with that its code does not say

A fine-tune that departs from the released model's own training objective
does not start from a minimum. It walks away from one, and held-out
accuracy falls while the training loss improves. Three such departures
showed up, each found by measuring the released model rather than reading
the package.

**The contour is one bin up.** Labels placed at `3·(midi − 21)` lined up with
the released contour head at .17. One bin higher, the middle of the three
sub-bins for a semitone, they lined up at .45. The note and onset heads read
the contour, so labels one bin low dragged all three.

**The onset head was trained with weighted positives.** Two settings looked
plausible: a plain mean over cells, where an onset is one cell in 15,000, and
the package's documented weighted-loss default, positives at .5. Neither
holds the model still. On GuitarSet's training players the released weights'
gradients from positive and negative onset cells cancel almost exactly (a
residual of .084 against component norms of 7.3 and .38) at a positive weight
of .0475. The note and contour heads, checked the same way, sit at their
plain positive fractions (.016 against .020, .0066 against .0071), so only
the onset head was weighted. Trained with a plain mean, the onset head halved
at real notes (.65 to .30) and peak AP fell from .66 to .52 in 600 steps
while held-out loss went down. At .5 it inflated to 97 candidates per real
note.

**Every kernel has unit norm.** Basic Pitch trains under Keras' `UnitNorm`
constraint, and every one of the released kernels has norm exactly 1 per
output filter. Individual weights then average .03 in the widest layer, so
an unconstrained Adam step at 1e-4 is the size of a weight. The fine-tune
re-projects onto the unit sphere after every step, as Keras does.

With all three in place, the first 300 steps raised GuitarSet player 04's
peak AP rather than lowering it.

## Training the front end

Every source in the [plan's data table](front-end-finetune-plan.md#data) plus
EGFxSet: GuitarSet players 00-03 through the microphone, the pickup and the
verifier's three augmented renders (1,200 recordings); Guitar-TECHS P1 and P2
on all four chains (2,482); and 2,759 EGFxSet notes, clean and through three
pedals, half of them mixed into chords. Batches draw .45 GuitarSet, .40
Guitar-TECHS and .15 EGFxSet. Adam at 2e-5, 1,200 steps of 16 two-second
windows per epoch.

Validation is threshold-free: average precision over every peak of the onset
head, a peak counting as real when a reference note of its pitch starts within
50 ms. That is the space the verifier ranks in. Beside it, the share of real
notes with a peak above the verifier's candidate threshold (.15), and how many
peaks clear it per real note.

| epoch | GS 04 mic AP | GS 04 pickup AP | P3 amp AP | mean | recall at .15 (mic) | candidates per note (mic) |
|---:|---:|---:|---:|---:|---:|---:|
| released | .6562 | .6800 | .6901 | .6754 | .969 | 32.2 |
| 1 | .6844 | .7173 | .6698 | **.6905** | .974 | 35.9 |
| 2 | .6855 | .7198 | .6618 | .6890 | .975 | 35.4 |
| 3 | .6865 | .7215 | .6568 | .6882 | .974 | 34.4 |

Without EGFxSet, same recipe:

| epoch | GS 04 mic AP | GS 04 pickup AP | P3 amp AP | mean |
|---:|---:|---:|---:|---:|
| 1 | .6854 | .7206 | .6511 | .6857 |
| 2 | .6846 | .7219 | .6470 | .6845 |
| 3 | .6859 | .7236 | .6406 | .6834 |

GuitarSet's held-out player gains three to four points within the first
epoch and holds them. Guitar-TECHS's held-out player does not: its amp
microphone loses two points at the first epoch and keeps losing. EGFxSet
slows that loss on every epoch, by 1.6 to 1.9 points, and costs GuitarSet
nothing measurable. The run stopped after epoch 3, and epoch 1 is the front
end carried forward (`runs/frontend/ft1/best.pt`). Candidate recall at the
verifier's threshold did not fall, so `.15` stays.

Why P3 falls is not settled. Its labels and P1-P2's come from the same MIDI
pickup, with each recording's lag read off the released model's onset head,
so fitting P1-P2 more closely should not by itself move P3 the wrong way. The
two differ in content: P1-P2 are mostly drills, P3 is eight minutes of music.

## Retraining the verifier

The verifier reads the front end's output, so a new front end means a new
verifier. `processing.tools.frontend_caches` re-rendered every cache from
`ft1` (4,924 posteriorgrams in eight minutes: GuitarSet's microphone and pickup,
the three augmented renders, Guitar-TECHS's four chains, the re-amped held-out
player, EGFxSet, IDMT), the CQT caches were reused, and three verifiers trained
at once on v8-gt's recipe, word for word:

- **v11-ft1**: the new front end.
- **v11-ft1-egfx**: the same, with EGFxSet's 240 excerpts as a string-only
  domain (`processing.tools.egfxset`).
- **v11-bp**: v8-gt's recipe on the released front end again: the control.
  Nothing seeds the verifier's initialisation, so the gap between it and
  v8-gt is what retraining alone moves.

Validation, at the shipped threshold (.75) and audio weight (.25): note F1 /
string accuracy / tab F1.

| | v8-gt (shipped) | v11-bp (control) | v11-ft1 | v11-ft1-egfx |
|---|---|---|---|---|
| GS 04 mic | .818 / .849 / .695 | .821 / .846 / .694 | .817 / .845 / .691 | .819 / .852 / .698 |
| GS 04 pickup | .826 / .800 / .661 | .831 / .799 / .664 | .830 / .801 / .665 | .831 / .785 / .652 |
| P3 amp mic | .731 / .612 / .448 | .726 / .585 / .425 | .737 / .626 / .461 | .749 / .612 / .458 |
| P3 direct input | .740 / .601 / .445 | .742 / .575 / .427 | .755 / .599 / .452 | .759 / .582 / .442 |
| P3 head mic | .265 / .575 / .152 | .275 / .550 / .151 | .279 / .540 / .150 | .257 / .574 / .148 |
| P3 room mic | .384 / .574 / .221 | .395 / .557 / .220 | .376 / .579 / .218 | .372 / .598 / .223 |
| P3 re-amped, distorted | .543 / .519 / .282 | .545 / .487 / .265 | .503 / .497 / .250 | .531 / .518 / .275 |

(The room microphones' note F1 is low for every model because their labels do
not say when a note started; they are string-only domains in training.)

Retraining alone moves note F1 by half a point and P3's string accuracy by up
to 2.7 points (.612 against .585), so string and tab differences of that size
between these columns are not evidence. Note F1 on electric guitar is: the new
front end with EGFxSet gains 1.8 points on P3's amp microphone and 1.9 on its
direct input over v8-gt, and 2.3 and 1.7 over the control. GuitarSet is level.
The distortion diagnostic goes the other way, by 1.2 points with EGFxSet and 4.0
without it, all of it precision.

v11-ft1-egfx was carried forward, chosen on validation before anything else
was scored: the best mean note F1 over GuitarSet 04's two chains and P3's amp
microphone (.800, against .792 for v8-gt and .795 for v11-ft1), with tab F1
level within the noise above (.603, .601, .606).

## Test, once

v11-ft1-egfx against v8-gt, both measured with today's code at the shipped
settings. GuitarSet 05 and IDMT were each scored once for the new verifier;
v8-gt's IDMT figure is the shipped configuration re-measured, not a new one.
That brings the configurations scored on IDMT to nineteen in the verifier docs'
count.

| | note F1 | P | R | string | tab F1 |
|---|---:|---:|---:|---:|---:|
| GuitarSet 05, v8-gt | .902 | .894 | .910 | .869 | .784 |
| GuitarSet 05, v11-ft1-egfx | .902 | .887 | .916 | .872 | .786 |
| IDMT, v8-gt | .850 | .864 | .837 | .585 | .497 |
| IDMT, v11-ft1-egfx | .853 | .859 | .848 | .597 | .509 |

Against the [acceptance table declared before training](front-end-finetune-plan.md#measurement-and-acceptance):
IDMT note F1 .853 ≥ .850, tab F1 .509 ≥ .497 and up 1.2 points, GuitarSet 05
note F1 .902 ≥ .897 and tab F1 .786 ≥ .779, P3's amp microphone .749 ≥ .731.
Every condition holds, so v11-ft1-egfx is the default verifier
(`pitch.DEFAULT_VERIFIER`), with v8-gt as the fallback on a machine without
the fine-tuned front end.

By IDMT subset (note F1 / string / tab F1):

| subset | notes | v8-gt | v11-ft1-egfx |
|---|---:|---|---|
| dataset2 | 4,278 | .830 / .556 / .462 | .832 / .567 / .472 |
| dataset3 | 90 | .851 / .649 / .552 | .829 / .773 / .641 |
| Fender Strat, neck | 78 | .994 / .641 / .637 | 1.000 / .718 / .718 |
| Fender Strat, neck, chords | 242 | .920 / .767 / .705 | .932 / .717 / .668 |
| Ibanez, bridge humbucker | 78 | 1.000 / .603 / .603 | 1.000 / .603 / .603 |
| Ibanez, bridge humbucker, chords | 242 | .947 / .720 / .682 | .969 / .745 / .722 |
| Ibanez, bridge + neck | 78 | .987 / .649 / .641 | .987 / .701 / .692 |
| Ibanez, neck humbucker | 78 | 1.000 / .718 / .718 | 1.000 / .769 / .769 |

`dataset3`, the subset [the verifier docs](note-verifier.md#the-one-subset-that-still-loses)
single out as losing to musical context, gains twelve points of string
accuracy. EGFxSet is a Stratocaster and two subsets here are Strats, which
should be kept in mind for those two rows.

**What this is and is not.** The gains are real but small, and the biggest is
where the front end was aimed: note detection on electric guitar, 1.8 to 1.9
points on Guitar-TECHS's held-out player and 0.3 on IDMT, where recall rises
1.1 points for 0.5 of precision. String and tab gains sit inside what
retraining alone moves on P3, and are only suggestive there; on IDMT they are
consistent across seven of eight subsets. The distortion diagnostic got worse
(note F1 .543 to .531, all precision), and Guitar-TECHS's held-out player loses
front-end AP the longer the front end trains, which is not understood. Nothing
here involves the camera.


## Reproduce

```sh
uv run --with tensorflow==2.16.2 python -m processing.tools.convert_basic_pitch
uv run python -m processing.training.frontend_data --prepare
uv run python -m processing.training.train_frontend --output runs/frontend/ft1 \
  --epochs 8 --steps 1200 --lr 2e-5
uv run python -m processing.tools.egfxset --prepare
uv run python -m processing.tools.frontend_caches --front-end runs/frontend/ft1/best.pt
uv run python -m processing.training.verifier_data --front-end runs/frontend/ft1/best.pt \
  --output data/verifier/candidates@ft1.npz --domains aug-v1 aug-v2 aug-v3 \
  --measure-domains pickup gt-di gt-amp gt-ego gt-exo gt-drive-v1 \
  --guitar-techs di amp ego exo drive-v1 --egfxset
uv run python -m processing.training.train_verifier --candidates data/verifier/candidates@ft1.npz \
  --front-end runs/frontend/ft1/best.pt --cqt data/guitarset/cqt \
  --planes data/verifier/planes@ft1 --output runs/verifier/v11-ft1-egfx --epochs 14 \
  --train-domains mic aug-v1 aug-v2 aug-v3 gt-di gt-amp gt-ego gt-exo egfx \
  --string-only-domains gt-ego gt-exo egfx --epoch-positives 46656
uv run python -m processing.tools.eval_verifier --model runs/verifier/v11-ft1-egfx/best.pt \
  --candidates data/verifier/candidates@ft1.npz --split val
uv run python -m processing.tools.eval_idmt --model runs/verifier/v11-ft1-egfx/best.pt
```

EGFxSet's four archives (`Clean`, `BluesDriver`, `TubeScreamer`, `RAT`) from
Zenodo record 7044411 unzip under `data/egfxset/extracted`.
