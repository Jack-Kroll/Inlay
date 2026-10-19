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
decoding, tab constraints, rendering, audio muxing, custom tunings, visual
rescue, verifier peak-picking and patch geometry, CQT time alignment, note
decoding and re-pluck handling, the audio string term, IDMT parsing, and the
training augmentation's filter responses, causality and determinism.

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
| Basic Pitch thresholds | `.821` | `.653` | `.536` |
| Note verifier + CQT | `.898` | `.882` | `.792` |
| + recording-chain augmentation | `.902` | `.880` | `.794` |

- Basic Pitch held-out GuitarSet F1 over all 312 non-tuning excerpts: `.779`.
- Blind string assignment with perfect pitches and no audio timbre: `.698`.
- Cross-dataset, IDMT-SMT-GUITAR electric guitars, 5,164 notes, nothing
  trained on: note F1 `.695` -> `.817`, string `.514` -> `.634`, tab F1 `.358`
  -> `.518`. Augmentation supplied the last `.803` -> `.817` and `.607` ->
  `.634` of that; five configurations were scored against IDMT, which is more
  than a held-out set should see.
- Cross-transducer within GuitarSet, validation player on the pickup mix rather
  than the microphone: string `.785` -> `.813` with augmentation, against
  `.861` on the microphone. The gap a change of transducer costs narrowed from
  8.2 points to 4.8.
- The runtime path was checked against the measurement harness on 12 test
  excerpts: `.879` -> `.911` note F1, same direction and magnitude.
- Unit and integration suite: 171 tests.

## Not established

- End-to-end audio/video transcription accuracy. Every number above is audio
  against annotated audio; none of them involve the camera.
- Whether the audio string posterior holds up on an unfamiliar instrument. On
  IDMT it gains 12 points rather than 23, and on two of eight subsets it is
  still worse than musical context alone. Neither remaining case is a timbre
  problem: one is strummed chords where context is nearly sufficient and the
  fusion of the two is worse than either alone, the other is 76 notes played
  high up a neck GuitarSet barely visits. See
  [the note verifier](note-verifier.md).
- Whether GuitarSet's hex-pickup stems help. Measured: they give the string
  head more than anything else tried and cost note detection more than they
  give, and are not used.
- Real-hand occlusion or absolute fret-number accuracy.
- Generalization to unseen guitars.
- Live-camera or mobile latency guarantees.
- Dense Core ML parity.

Training loss, smoke tests, and visual inspection support development, but they
do not replace a manually annotated held-out benchmark.
