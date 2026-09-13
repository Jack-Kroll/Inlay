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
decoding, tab constraints, rendering, audio muxing, custom tunings, and visual
rescue.

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

- Basic Pitch held-out GuitarSet F1: `.779`.
- Blind string assignment with perfect pitches: `.698` overall.
- Unit and integration suite: 120 tests.

## Not established

- End-to-end audio/video transcription accuracy.
- Real-hand occlusion or absolute fret-number accuracy.
- Generalization to unseen guitars.
- Live-camera or mobile latency guarantees.
- Dense Core ML parity.

Training loss, smoke tests, and visual inspection support development, but they
do not replace a manually annotated held-out benchmark.
