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

## Code review verification — 2026-09-07

- All 86 unit and integration tests pass. The regression suite run against a
  temporary copy of the pre-review code reproduces 18 failing cases.
- Corrected stale onset evidence, overlapping-string assignments, duplicate
  note loss, arbitrary board-axis signs, and absolute-numbering expiry during
  motion-only tracking. Custom tunings now control audio bounds and tab labels.
- Fixed session grouping changing original-image sampling weights, boundary
  wire targets falling outside supervised pixels, extreme-aspect letterboxing,
  and non-finite geometry incorrectly passing export comparison.
- Ran the real dense detector, MediaPipe and Basic Pitch on 60 frames of
  `clips/take-2.mov`, producing JSONL and a two-second overlay with audio.
  A second run exercised explicit seven-string tuning and mirrored rendering;
  that intentionally different tuning is a software check, not ground truth.
- Checked exported note counts, timing, pitch/string/fret consistency and
  overlapping-string exclusion. Inspected the mirrored overlay visually.
- Actual FFmpeg integration checks verify that short replacement audio cannot
  truncate rendered frames and that a video supplied as the audio source cannot
  replace the annotated video stream. Confirmed 60 frames after audio padding.
- Ran one validation image through clean and synthetic-cover evaluation using
  the existing dense checkpoint. No retraining or accuracy claim is implied.


## Follow-up review — 2026-09-07

- 117 tests pass with the transcription extra installed. Basic Pyright checking
  reports zero errors and warnings over all 39 Python files; the lockfile check
  passes without dependency changes. Optional pitch tests skip on base installs.
- Fixed in-place frequency masking contaminating later pitch-sweep candidates,
  and rounded upper frequency cutoffs excluding the highest playable pitch.
  Regression checks reproduce both original failures and pass after the fixes.
- Clarified optional values and NumPy/OpenCV/PyTorch types, added missing
  image-decode/trainer guards, and checked legacy OBB conversion with both tensor
  and NumPy inputs. Most editor diagnostics were typing issues, not runtime
  failures. The mir_eval offset-ignore API needs one documented stub exception.
- Ran Basic Pitch, the real dense checkpoint, MediaPipe, and mirrored rendering
  on 60 frames of `clips/take-3.mov`; verified all 60 rendered frames and exported
  note timing/pitch/string/fret consistency. This is a runtime smoke test, not
  accuracy validation. Core ML required execution outside the filesystem sandbox.
- Documented existing audio preprocessing and a threshold experiment for quiet
  and short notes. No unmeasured compression/denoising or default-threshold change
  was introduced. Corrected stale claims about which stages have been measured
  and the claim that perfect-pitch blind string accuracy bounds full-pipeline
  accuracy.
