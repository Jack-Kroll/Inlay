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
uv run python -m processing.tools.variants --variants 3
uv run python -m processing.training.verifier_data --domains aug-v1 aug-v2 aug-v3
uv run python -m processing.training.train_verifier \
  --cqt data/guitarset/cqt --output runs/verifier/v3-aug --epochs 14 \
  --train-domains mic aug-v1 aug-v2 aug-v3 --epoch-positives 46656
```

The middle two steps are augmentation, described under
[recording-chain augmentation](#recording-chain-augmentation) below.

Splits hold whole players out: 00–03 train, 04 validates, 05 is touched only
for the final number. A player's guitar, touch and session repeat across all
60 of their excerpts, so splitting any other way reports memory.

Each epoch takes every positive candidate plus four negatives each, resampled,
so a run sees far more of the 25:1 majority than one epoch holds. The score
threshold and the audio string weight were both chosen on player 04.

`--epoch-positives` caps the real notes an epoch draws. Adding three augmented
copies of every excerpt otherwise quadruples an epoch, and a run that takes
four times as long is not a comparison; every run below draws the same 46,656
positives per epoch over the same 960 training excerpts, so only the *source*
of the extra data differs.

## Measure it

```sh
uv run python -m processing.tools.eval_verifier --split test
uv run python -m processing.tools.eval_verifier --split val --domain pickup
uv run python -m processing.tools.idmt --prepare
uv run python -m processing.tools.eval_idmt
```

The second line measures a held-out player on GuitarSet's *pickup* transducer
rather than its microphone, which is the cheap cross-instrument check described
under [recording-chain augmentation](#recording-chain-augmentation). It needs
`processing.tools.variants --source pickup --variants 0` to have rendered that
domain first.

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
| + augmentation | .902 | .909 | .896 | .880 | .794 | .878 | .947 |

"string" is the share of pitch-matched notes placed on the right string;
"fretted" and "open" split that by whether the note was really open. "tab F1"
requires pitch and string at once, which is what a tab reader sees.

Raising the threshold to .90 trades recall for precision at the same F1
(.897, P .921, R .874) — the surface is flat across that range.

The last row is what ships. Augmentation was added for the electric guitars of
the next section, and on GuitarSet it is close to free: note F1 .898 → .902,
tab F1 .792 → .794, open strings .933 → .947, against a string accuracy of .880
where it was .882. Two tenths of a point on 7,789 matched notes is sixteen of
them — but it is a regression rather than an improvement, and the gain on
other instruments is the reason to take it.

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
| + augmentation | .817 | .863 | .775 | .634 | .518 |

Note detection transfers: +10.8 points of F1 on instruments and a recording
chain the model never saw, almost all of it precision.

**The string head transfers much less.** Before augmentation it gained 9.3
points here against 22.9 on GuitarSet, and on three of eight subsets it was
worse than musical context alone. Gating the term on the head's own confidence
does not help: at a .9 gate IDMT string accuracy *drops* to .580, so the head
is confidently wrong rather than uncertain. Those cues were specific to the six
acoustic guitars and the one microphone it was trained on, and the verifier
trained with no augmentation at all. That is what the next section is about.

IDMT-SMT-GUITAR is CC BY-NC-ND 4.0. It is used only to measure. Nothing is
trained on it and nothing derived from it ships.

## Recording-chain augmentation

`processing.training.augment` puts an excerpt through a random recording chain:
level, spectral tilt, two to four resonances, a short cabinet impulse, tanh
saturation, white or pink hiss, and an MP3 round-trip. One chain is drawn per
excerpt rather than per frame, because a guitar through an amp through a
microphone is one fixed colouration for the length of a take.

Two things are deliberately absent. Pitch shifting and time stretching would
move the labels the verifier is trained against. And every impulse response is
direct-dominant and under 40 ms, because a reverb tail long enough to be
audible is also long enough to smear the attack the exist head is asked to
find; the tests assert that no stage puts energy before an attack.

The MP3 stage is not only about other people's guitars. Inlay's own input is a
recorded video, whose audio has already been through a lossy codec, while every
excerpt it trains on is uncompressed WAV. A codec also prepends its own delay
— about 1,100 samples for LAME, twice the verifier's onset tolerance — so the
decoded audio is cross-correlated back onto the original timeline.

### Why the whole chain is re-rendered

The verifier reads a posteriorgram and a CQT, both cached per excerpt, so
augmenting the audio means re-running Basic Pitch as well. The alternative was
to augment the CQT channels on the fly and leave the posteriorgram channels
clean, which is cheaper but leaves five of the eight input channels unaugmented
— and the posteriorgram channels are the ones carrying most of the string
signal, since the string head reaches .858 from them alone against .879 with
the CQT stacked on.

It turned out not to be a trade-off. Basic Pitch's Core ML graph runs a
thirty-second excerpt in 0.08 s, two orders of magnitude under the 2.5 s the
CQT costs, so rendering a full variant costs about what the CQT cache already
cost. `processing.tools.variants` pre-renders them; three variants of the 240
training excerpts take about twelve minutes and 2 GB.

Candidates are peak-picked from each variant's *own* posteriorgram, not reused
from the clean one: a frame that is a peak in the clean audio need not be one
once the audio has been through an amp, and a patch the runtime would never
centre on teaches nothing. Augmentation leaves the candidate recall ceiling
essentially intact, .979 against .982, at 24 candidates per real note rather
than 21.

### Measuring it without spending IDMT

GuitarSet ships every take through two transducers, and only the microphone had
been used. The pickup mix is the same players playing the same notes through a
hex pickup, with the same annotations, so it is a held-out recording chain with
exact labels and no licence constraints. Measured on validation player 04:

| | mic string | pickup string | gap |
|---|---:|---:|---:|
| Verifier + CQT | .867 | .785 | −8.2 |
| + augmentation | .861 | .813 | −4.8 |

Changing only the transducer costs the un-augmented model 8.2 points.
Augmentation takes back a third of that, and its +2.8 points on the pickup
predicted its +2.7 on IDMT almost exactly.

The proxy is useful but not a substitute. A harder chain (`--strength 1.8`)
scored *better* on the pickup, .829, and *worse* on IDMT string accuracy, .623
against .634, because IDMT differs in instrument, player and repertoire and not
only in transducer. Where the two disagree, IDMT is the number that counts and
the pickup is the one that is cheap.

### What each source of data was worth

Every run below trains on 960 excerpts and draws the same 46,656 positives per
epoch, so only the source of the extra 720 differs. Measured on IDMT:

| trains on | note F1 | string | audio alone | tab F1 | subsets losing |
|---|---:|---:|---:|---:|---:|
| microphone alone | .803 | .607 | .578 | .488 | 3 of 8 |
| + 3 augmented copies | .817 | .634 | .616 | .518 | 2 of 8 |
| + 3 hex remixes | .788 | .638 | .638 | .503 | 2 of 8 |
| + 2 augmented, 1 hex remix | .780 | .631 | .622 | .492 | 3 of 8 |
| + 3 augmented, harder chain | .832 | .623 | .605 | .518 | 2 of 8 |

"audio alone" is the string head's own argmax, without musical context mixed
in; it is the cleaner measure of the head itself, and augmentation moves it
.578 → .616.

The last row is the clearest finding. Augmentation strength helps note
detection monotonically — .803, .817, .832 — and the string head has an optimum
well short of it, because the cues the string head reads are exactly the ones a
harder chain destroys. The harder chain also drops GuitarSet test string
accuracy to .862 from .880, so it is not shipped.

### Per subset

| subset | notes | context | shipped | augmented | audio alone |
|---|---:|---:|---:|---:|---:|
| `Fender Strat Clean Neck SC` | 78 | .449 | .615 | .628 | .603 |
| `Fender Strat Clean Neck SC Chords` | 227 | .751 | .706 | .718 | .740 |
| `Ibanez Power Strat Clean Bridge HU` | 78 | .474 | .679 | .603 | .628 |
| `Ibanez … Bridge HU Chord` | 230 | .772 | .754 | .791 | .743 |
| `Ibanez … Bridge+Neck SC` | 77 | .468 | .675 | .701 | .688 |
| `Ibanez Power Strat Clean Neck HU` | 78 | .462 | .718 | .667 | .667 |
| `dataset2` | 3,159 | .481 | .583 | .616 | .596 |
| `dataset3` | 76 | .818 | .654 | .605 | .579 |

Two subsets still lose to musical context, down from three, and neither is a
timbre problem:

- **`Fender Strat Clean Neck SC Chords`** is strummed six-string chords, 12.1
  notes per onset, every fret from 0 to 11. A fully voiced barre chord has
  essentially one plausible spelling, so context alone reaches .751 — and the
  string head reaches .740 on its own, yet the two *fused* land at .718, below
  either. That is the fusion weight on saturated chords, not the string head,
  and `AUDIO_STRING_WEIGHT` was left alone because moving it means re-measuring
  both corpora. It is the clearest remaining lead.
- **`dataset3`** is 76 notes of classical repertoire on an Ibanez RG, with
  14.4% of its notes above fret 12 where GuitarSet has 1.2% above fret 12 and
  0.1% above fret 15. It is out of distribution in neck position rather than in
  tone, which no amount of EQ can fix. Resampling an excerpt to shift it up the
  neck — scaling every annotated time by the same ratio, so onsets stay sharp
  and labels stay exact — would address it and is untried.

## Rejected ideas

- **GuitarSet's hex-pickup stems.** The debleeded hex pickup records each string
  on its own channel; measured over isolated notes, 98% of a string's energy
  lands on its own channel, so a remix at random relative levels has string
  labels that are exact. It gave the string head the most of anything tried —
  audio-alone .638, the best of any run — and cost note detection badly, F1
  .788 against .817, because the hex pickup's posteriorgram has no body
  resonance in it and training the exist head on that hurts microphone audio.
  Combining it with augmentation was worse than augmentation alone on every
  measure. The renderer is kept (`--source hex --remixes N`) because the
  negative result is specific to mixing it into one model, not to the data.
- **A harder augmentation chain**, as above: better notes, worse strings, and a
  GuitarSet string regression that disqualifies it.
- **Gating the audio string term on the head's own confidence.** At a .9 gate
  IDMT string accuracy falls to .580. The head is confidently wrong rather than
  uncertain, so there is nothing for a gate to catch.
- **Electric training corpora** were not needed to get this far and so were not
  downloaded. EGDB (arXiv 2202.09907) is re-rendered through several amps and a
  CC BY 4.0 copy exists on Zenodo, which would permit training; GOAT (arXiv
  2509.22655) is 5.9 hours of direct-input electric. Either is the right next
  step if the string head has to improve further, and neither licence was
  verified beyond that Zenodo record.

### A note on how many configurations IDMT scored

IDMT is meant to be measured once. Five were scored against it here, because
the target was defined in terms of its numbers and the pickup proxy disagreed
with it on one comparison. The per-subset figures on 76 and 78 notes should be
read with that in mind: `dataset3` moving .654 → .605 is sixteen notes.

## Limits

- Everything above is audio against annotated audio. It says nothing about
  camera geometry, and the end-to-end accuracy of video plus audio together is
  still unmeasured.
- Trained on 3 hours of acoustic guitar from six players, augmented into a
  wider range of recording chains but not a wider range of guitars. The string
  head should still be treated with suspicion on an unfamiliar instrument:
  .880 on a held-out GuitarSet player, .634 on IDMT's electrics.
- The verifier answers whether a note started. Offsets still come from the
  frame head and are not measured.
- Notes 60–100 ms apart remain hard; peak-picking cannot separate two attacks
  inside two frames.
- Augmentation covers level, EQ, cabinet, amp drive, hiss and codec. It does
  not cover neck position, which is where the one remaining badly-losing IDMT
  subset sits.
- On fully voiced chords the audio string term and musical context are each
  better alone than fused. The fusion weight is a single global constant and
  does not know how many notes share an onset.
