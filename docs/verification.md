# Rewrite verification — 2026-09-06

This records software checks, not a trained-model accuracy claim.

- Audited all 1,615 local image/label pairs and built `data/fretboard.json`.
  The adjacent audit records clipping, source groups and excluded overlap.
- Reviewed an annotated twelve-image contact sheet and eight cross-split pairs.
- Unit checks cover the legacy baseline plus partial-head loss gradients,
  negative images, source overlap, coordinate transforms, clipping, tightly
  spaced wires, rotation, hidden nut, nonconsecutive fret numbering, missing
  motion support, timeout, timestamp rewind and resolution changes.
- Completed two CPU smoke epochs (one train/validation batch per epoch), wrote
  best/last checkpoints, loaded the checkpoint for clean/synthetic-cover
  evaluation, and exercised resume loading. These checkpoints are explicitly
  marked as smoke tests and are not a usable guitar model.
- Downloaded the official torchvision ResNet34 encoder with hash verification.
  Completed one 768 × 768 train batch and one validation batch on Apple MPS
  with ImageNet initialization. It took approximately 4.34 seconds for that
  tiny epoch including checkpoint writes; this is not a full training ETA.
- Checked the geometry decoder against ideal target maps. This exposed and
  fixed the tendency of an unconstrained Hough transform to consume dense fret
  pixels as longitudinal lines. A close-spacing regression now checks 41
  separate wires. Ideal-map output is not neural-network performance.
- The lockfile was regenerated for explicit PyTorch/Torchvision/PyYAML dependencies
  without changing their installed pinned versions, and checked offline.

The remaining accuracy work is a full training run followed by comparison on
independent, manually annotated guitars/recorded clips. The existing test set was
inspected during development and must be treated as a development diagnostic,
not a fresh final benchmark. No live-camera accuracy, real-hand occlusion
accuracy, fret-number accuracy, or mobile latency claim has been established.
