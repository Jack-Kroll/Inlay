# Verification record

This file summarizes what has actually been checked. Passing software tests is
not the same as proving model accuracy.

## Repository and data checks

- The project uses a locked Python 3.11 `uv` environment.
- Manifests validate splits, image hashes, annotations, and partial supervision.
- Near-duplicate grouping removes obvious cross-split overlap.
- Dense checkpoints record architecture, settings, manifest hash, optimizer,
  scheduler, and random state.

## Software checks

The suite covers model training behavior, partial labels, letterboxing, fret
decoding, numbering, tracking expiry, projective transforms, mirroring, pitch
decoding, tab constraints, rendering, audio muxing, custom tunings,
verifier peak-picking and patch geometry, CQT time alignment, note
decoding and re-pluck handling, the audio string term, the whole-onset string
match (checked against brute force), IDMT parsing, and the training
augmentation's filter responses, causality and determinism.

```sh
uv run python -m unittest discover -s tests -v
```

## Runtime checks

- The dense detector, MediaPipe, and Basic Pitch have produced annotated video
  and JSONL from local clips.
- Mirroring preserves original processing coordinates.
- Short replacement audio cannot truncate rendered video.
- MPS and CPU smoke paths have run.
- Pyright reports zero errors across the configured project.

## Measured results

Audio only, against annotated audio. GuitarSet player 05 was never used for
training, checkpoint selection, or threshold choice; 8,715 notes, no video.

| | note F1 | string | tab F1 |
|---|---:|---:|---:|
| Basic Pitch thresholds | `.821` | `.654` | `.537` |
| Note verifier + CQT | `.898` | `.882` | `.792` |
| + recording-chain augmentation | `.902` | `.881` | `.794` |

The middle row predates the whole-onset string assignment and was not
re-measured; the other two are current.

The string and tab columns assume `--audio-string-weight .25`. The shipped
default is 0, which scores `.902` note F1, `.668` string and `.603` tab F1 on
the same notes: the term is worth 21 points of string accuracy here and loses
badly on electric guitars, so it is off by default. See
[tab logic](tab-logic.md).

- Basic Pitch held-out GuitarSet F1 over all 312 non-tuning excerpts: `.779`.
- Blind string assignment with perfect pitches and no audio timbre: `.698`
  note by note, `.707` solving each onset as a whole, over 62,476 notes.
- Cross-dataset, IDMT-SMT-GUITAR electric guitars, 5,164 notes, nothing
  trained on: note F1 `.695` -> `.817`, string `.515` -> `.638`, tab F1 `.358`
  -> `.521`. Augmentation supplied `.803` -> `.817` and `.607` -> `.634` of
  that, and assigning each onset's strings as a whole the last `.634` ->
  `.638`. Nine configurations have been scored against IDMT, which is more
  than a held-out set should see; the assignment change was chosen entirely on
  GuitarSet's validation player and then measured on IDMT once.
- Cross-transducer within GuitarSet, validation player on the pickup mix rather
  than the microphone: string `.785` -> `.813` with augmentation, against
  `.861` on the microphone. The gap a change of transducer costs narrowed from
  8.2 points to 4.8.
- The runtime path was checked against the measurement harness on 12 test
  excerpts: `.879` -> `.911` note F1, same direction and magnitude.
- Unit and integration suite: 173 tests.

## Not established

- End-to-end audio/video transcription accuracy. Every number above is audio
  against annotated audio; none of them involve the camera.
- Whether the audio string posterior holds up on an unfamiliar instrument. On
  IDMT it gains 12 points rather than 23, and on one of eight subsets it is
  still worse than musical context alone: 76 notes played high up a neck
  GuitarSet barely visits, where it costs 24 points. An earlier count of two
  such subsets was wrong — it read "context alone" off the Basic Pitch
  baseline, a different and easier set of notes than the one being scored. See
  [the note verifier](note-verifier.md#per-subset).
- Whether GuitarSet's hex-pickup stems help. Measured: they give the string
  head more than anything else tried and cost note detection more than they
  give, and are not used.
- Whether the string head can be taught the top of the neck. It is worst
  there by a wide margin -- `.636` of IDMT's notes below fret 12 land on the
  right string against `.361` at fret 12 and up, and `.153` past fret 15,
  which is below chance. Resampling excerpts up the neck fixes that region and
  costs more in the 92% of notes below it; measured and rejected.
- Real-hand occlusion or absolute fret-number accuracy.
- Generalization to unseen guitars.
- Live-camera or mobile latency guarantees.
- Dense Core ML parity.

Training loss, smoke tests, and visual inspection support development, but they
do not replace a manually annotated held-out benchmark.
