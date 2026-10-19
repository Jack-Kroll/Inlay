# The note verifier

Basic Pitch decides a note exists by thresholding one number per pitch per
frame. On guitar that was where inlay lost most of its notes, and the loss was
not a model limitation — it was a decoding one.

## The diagnostic

For every GuitarSet reference note, read the peak activation Basic Pitch
produced at that pitch within 50 ms of the true onset, and split by whether
the shipped thresholds found the note:

| | onset p50 | onset p90 | frame p50 | frame p90 |
|---|---:|---:|---:|---:|
| Matched | .872 | .951 | .794 | .884 |
| Missed | .469 | .741 | .413 | .727 |

Of 4,416 missed notes across 120 excerpts:

- 0.1% had essentially no onset activation (below .10).
- 79.9% were above .30.
- 22.2% were above .60 — over the threshold that was supposed to accept them.

The missed notes' median frame activation, .413, sits just under the .45 frame
gate. Basic Pitch requires onset *and* sustained frame support together, and a
staccato, muted or quiet note fails the second test however clear its attack.

The false positives tell the same story from the other side. Of 3,621 of them:

| Source | Share |
|---|---:|
| Right pitch, onset off by 150–350 ms | 25.6% |
| Spurious pitch during a real chord | 23.6% |
| Right pitch, onset off by 50–150 ms | 20.9% |
| Spurious, nothing sounding | 14.9% |
| Octave above a real note | 12.0% |
| Fifth or twelfth above a real note | 2.3% |

Nearly half are the correct pitch at the wrong time: one note split into two,
or two merged into one. Those are not invented notes, and fixing one removes a
false positive *and* a miss.

## What a threshold cannot do

A hand-written onset-first decoder was built to exploit this: peak-pick the
onset head, confirm with a low frame gate, follow the frame head down to an
offset, and suppress octaves and fifths below a much stronger simultaneous
attack. Fifty-five settings were swept against cached posteriorgrams.

Its best setting scored F1 .799 against the shipped .797 — a tie, trading
precision for recall roughly one for one. The grid's optimum drifted back
towards the shipped thresholds.

That result is the argument for the model. The activation is there, but its
*value* does not separate a real quiet note from the octave of a loud one:
both look the same at that pitch and that instant. What separates them is the
surrounding harmonic and temporal context, which a single number discards.

## The model

Candidates come from permissive onset peak-picking — a local maximum above
.15. That keeps a candidate within the 50 ms tolerance for 98.7% of GuitarSet's
notes, at 27 candidates per real note:

| Peak ≥ | Recall ceiling | Candidates per real note |
|---:|---:|---:|
| .10 | .993 | 189.7 |
| .15 | .987 | 27.1 |
| .20 | .977 | 13.9 |
| .30 | .948 | 5.8 |
| .45 | .885 | 2.4 |

Around each candidate the model reads a patch of ±10 frames and ±24 semitones
— two octaves, so the harmonics above and the fundamentals below are both in
view — over the note head, the onset head, and the contour head folded into
three channels at its native three bins per semitone. A constant-Q transform of
the audio is stacked on as three more channels on the same grid.

A small CNN (0.67M parameters) over that patch predicts three things:

- **exist**: is a real note attacking here?
- **delta**: how far the true onset is from this frame, in frames.
- **string**: which of the six strings played it.

The forward pass over the audio is the one inlay already ran, so the verifier
adds no audio inference; the CQT costs about 2.5 s per thirty seconds.

### Why a string head works

Basic Pitch's heads describe pitch, not tone, but they do not describe it
perfectly: the contour head's three bins per semitone carry inharmonicity, and
the relative strength of a note's partials carries brightness. Two strings
sounding the same pitch differ in both. Asked on its own, the string head
places .858 of a held-out player's notes on the right string from the
posteriorgram alone, and .879 with the CQT stacked on.

Musical context, given the same notes and no camera, manages .644. The two are
independent evidence, so they are mixed rather than ranked — see
[tab logic](tab-logic.md).

## Train it

GuitarSet must be unzipped under `data/guitarset` and its posteriorgrams
cached, which `processing.tools.tune_pitch` does on its first run.

```sh
uv run python -m processing.tools.cqt
uv run python -m processing.training.verifier_data
uv run python -m processing.training.train_verifier \
  --cqt data/guitarset/cqt --output runs/verifier/v2-cqt --epochs 14
```

Splits hold whole players out: 00–03 train, 04 validates, 05 is touched only
for the final number. A player's guitar, touch and session repeat across all
60 of their excerpts, so splitting any other way reports memory.

Each epoch takes every positive candidate plus four negatives each, resampled,
so a run sees far more of the 25:1 majority than one epoch holds. The score
threshold and the audio string weight were both chosen on player 04.

## Measure it

```sh
uv run python -m processing.tools.eval_verifier --split test
uv run python -m processing.tools.idmt --prepare
uv run python -m processing.tools.eval_idmt
```

Both read the decode from `processing.music.verify`, the same code the CLI
runs, so the numbers describe what inlay does rather than what a harness does.

## Results

GuitarSet player 05, never used for training, checkpoint selection, or
threshold choice. 8,715 notes, at the shipped score threshold of .85, with no
video — the tab stage has musical context and the audio string posterior, and
nothing else:

| | note F1 | P | R | string | tab F1 | fretted | open |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic Pitch thresholds | .821 | .808 | .834 | .653 | .536 | .656 | .553 |
| Verifier | .891 | .884 | .898 | .861 | .767 | .860 | .905 |
| Verifier + CQT | .898 | .902 | .894 | .882 | .792 | .880 | .933 |

"string" is the share of pitch-matched notes placed on the right string;
"fretted" and "open" split that by whether the note was really open. "tab F1"
requires pitch and string at once, which is what a tab reader sees.

Raising the threshold to .90 trades recall for precision at the same F1
(.897, P .921, R .874) — the surface is flat across that range.

On player 04, which chose the threshold and the audio weight: note F1 .754 →
.817, string .611 → .867, tab F1 .461 → .708.

Turning the audio string term off but keeping the verifier's notes gives note
F1 .898 with string .644 and tab F1 .579, so roughly half the tab improvement
is better notes and half is the string head.

## Generalisation

IDMT-SMT-GUITAR is a different corpus: a Fender Stratocaster, a Gibson Les
Paul, an Aristides and an Ibanez, over four pickup settings, plus acoustic mic
and pickup. 5,164 annotated notes, none of them trained on.

| | note F1 | P | R | string | tab F1 |
|---|---:|---:|---:|---:|---:|
| Basic Pitch thresholds | .695 | .590 | .847 | .514 | .358 |
| Verifier | .806 | .803 | .809 | .586 | .473 |
| Verifier + CQT | .803 | .847 | .764 | .607 | .488 |

Note detection transfers: +10.8 points of F1 on instruments and a recording
chain the model never saw, almost all of it precision.

**The string head transfers much less.** It gains 9.3 points here against 22.9
on GuitarSet, and on two subsets it is worse than musical context alone —
`Fender Strat Clean Neck SC Chords` falls .751 → .662 and `dataset3` .818 →
.627. Gating the term on the head's own confidence does not help: at a .9 gate
IDMT string accuracy *drops* to .580, so the head is confidently wrong rather
than uncertain. Those cues are specific to the acoustic guitars and the one
microphone it was trained on. Electric training data is the fix; a calibration
gate is not.

IDMT-SMT-GUITAR is CC BY-NC-ND 4.0. It is used only to measure. Nothing is
trained on it and nothing derived from it ships.

## Limits

- Everything above is audio against annotated audio. It says nothing about
  camera geometry, and the end-to-end accuracy of video plus audio together is
  still unmeasured.
- Trained on 3 hours of acoustic guitar from six players. The string head in
  particular should not be trusted on an unfamiliar instrument.
- The verifier answers whether a note started. Offsets still come from the
  frame head and are not measured.
- Notes 60–100 ms apart remain hard; peak-picking cannot separate two attacks
  inside two frames.
- No augmentation is applied during training. Gain, EQ, amp and room
  augmentation is the obvious next step and is untried.
