# Architecture

Media → timestamped visual observations → musical interpretation → editable music
→ notation/tab rendering. Vision outputs image-space observations, not definitive
played notes. Audio and vision will eventually share the media timestamp origin.

## Fretboard replacement

1. `FretboardDataset` reads a versioned manifest, validates image hashes and split
   groups, letterboxes images, generates coarse neck masks and wire targets, and
   applies joint spatial augmentation. Only annotated heads contribute loss.
2. `FretboardNet` shares a pretrained ResNet34 encoder across three dense heads.
   Its U-Net decoder preserves spatial detail. Neck and wires can overlap; there
   is no softmax forcing a wire pixel to cease belonging to the neck.
3. `decode_maps` selects the largest supported neck region, extracts transverse
   wire evidence inside it, merges duplicate line proposals and extends wire
   segments to the predicted board boundary. It can find a board without a nut.
4. Numbering fits `s(n) = a r(n)/(1+c r(n))`, where
   `r(n)=1−2^(−n/12)`, against nut-relative wire positions. Candidate assignments
   allow missing frets, require enough inliers and reject close alternatives.
   No scale-length/nut-width prior or decorative-inlay pattern is used.
5. `FretboardTracker` uses forward/backward Lucas–Kanade flow and RANSAC homography
   checks to validate motion. It requires enough inliers spread over the board,
   rejects extreme movement/area changes, smooths matched lines using elapsed
   time, and exposes detected/tracked/lost states. Missing measurements can be
   bridged only briefly with motion support. Numbering expires separately.
6. The preview draws the boundary, wire lines and available numbers, displays
   separate heatmaps, and can write timestamped JSONL with states and latency.
   Video time comes from frame index/FPS, not inference wall time; camera time
   comes from a monotonic clock. Timestamp discontinuities reset the tracker.

The original OBB implementation is retained in `legacy_preview.py`, `geometry.py`
and `train_obb.py` to enable controlled comparisons. It is not the default runtime.

## Deliberate limits and next validation

- This architecture has not yet been shown to outperform the old checkpoint.
  Short smoke runs validate software, not recognition quality.
- Existing OBB-derived masks are coarse; unseen guitars and marker/occlusion
  coverage need better data. See `dataset-review.md`.
- One dominant board is supported. Multiple guitars require instance separation.
- Hidden endpoints are model/geometry estimates. The tracker drops unsupported
  occlusion instead of freezing a plausible-looking board indefinitely.
- Long nut occlusion leaves numbering unknown while board detection may continue.
  Stable long-duration numbering needs verified indexed keypoints or calibration.
- A nut-like capo can still confuse an untrained/insufficiently trained model.
- Numbering assumes conventional straight frets and a usable planar projection.
  Fanned frets, severe distortion, curved views and tiny distant boards need
  separate modeling and validation. No strings are measured by these heads.
- Learned confidence is not yet calibrated. Thresholds, smoothing, expiry and
  candidate-fit margins need validation on held-out recorded clips.
- The decoder predicts evidence lines; it does not fill a complete numbered grid
  outside the supported region. Absolute fret-number accuracy cannot be measured
  from the current unnumbered annotations.
- Dense Core ML export and mobile device parity/latency testing remain to do
  after selecting a trained checkpoint. No runnable React Native app exists yet.

The music schema in `packages/music/performance.schema.json` remains an initial
performance contract. Preserve musical spelling, beat/tempo maps and voices when
extending it for score import/export.
