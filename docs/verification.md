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
decoding and re-pluck handling, the audio string term, and IDMT parsing.

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

- Basic Pitch held-out GuitarSet F1 over all 312 non-tuning excerpts: `.779`.
- Blind string assignment with perfect pitches and no audio timbre: `.698`.
- Cross-dataset, IDMT-SMT-GUITAR electric guitars, 5,164 notes, nothing
  trained on: note F1 `.695` -> `.803`, tab F1 `.358` -> `.488`.
- The runtime path was checked against the measurement harness on 12 test
  excerpts: `.879` -> `.911` note F1, same direction and magnitude.
- Unit and integration suite: 146 tests.

## Not established

- End-to-end audio/video transcription accuracy. Every number above is audio
  against annotated audio; none of them involve the camera.
- Whether the audio string posterior holds up on an unfamiliar instrument. On
  IDMT it gains 9 points rather than 23, and on two subsets it is worse than
  musical context alone.
- Real-hand occlusion or absolute fret-number accuracy.
- Generalization to unseen guitars.
- Live-camera or mobile latency guarantees.
- Dense Core ML parity.

Training loss, smoke tests, and visual inspection support development, but they
do not replace a manually annotated held-out benchmark.
