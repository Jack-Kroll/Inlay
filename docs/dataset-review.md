# Dataset review

The current visual dataset supports a prototype, but not a claim of reliable
performance on new guitars or real occlusions.

## Current visual data

The local source is the CC BY 4.0 `guitar by B101` Roboflow export from March
2025.

| Split | Original images | Prepared images |
|---|---:|---:|
| Train | 1,489 | 1,483 |
| Validation | 84 | 48 |
| Test | 42 | 42 |

Important limitations:

- The 1,615 images come from 626 filename families, not 626 verified guitars.
- Many images are rotated variants or nearby frames from the same source.
- All images were stretched to 512 x 512.
- There are no reviewed negative images.
- Labels are coarse boxes, not precise masks or identified fret numbers.
- Guitar and recording-session identities have not been fully verified.

A perceptual-hash audit found related images across the original splits. The
importer groups near duplicates and quarantines 42 lower-priority records. This
reduces obvious leakage but does not prove session independence.

## Import protections

The normalized training manifest:

- Keeps near duplicates in one split.
- Clips coordinates that extend outside the image.
- Preserves source and attribution metadata.
- Masks unknown annotation heads instead of treating them as negatives.
- Rejects missing label files unless they represent reviewed negatives.
- Accepts explicit guitar/session group mappings.

The prepared manifest stores image hashes. Training fails if an image changes
after preparation.

## Additional data

Potential sources still require license and overlap review:

- `guitar-fret-6pt`: possible keypoint annotations.
- `Guitar Transcription Dataset`: useful neck polygons.
- `Code and Chords guitar-frets-segmenter`: inspect whether labels mean wires
  or spaces before importing.
- GAPS: diverse performances, but no ready-made fretboard labels and subject to
  research-use terms.

GuitarSet supplies audio and per-string musical annotations. It does not supply
camera-space fretboard or fingertip labels.

## What to collect next

Target 3,000-5,000 reviewed frames across 30-50 guitars and multiple sessions.
Include different instrument types, inlays, lighting, blur, phones, camera
angles, capos, partial crops, real hand occlusions, and 10-20% reviewed
negatives.

Store guitar and session IDs. Split by guitar for new-instrument evaluation and
by session for same-instrument evaluation. Do not randomly split neighboring
video frames.

Label visible wire centerlines, real endpoints, known occluded geometry, fret
numbers, and string intersections separately. Mark uncertain points unknown.

## Metrics to report

- Wire precision, recall, and endpoint error.
- Fret-number error and abstention rate.
- False positives on negatives.
- Tracking loss, reacquisition time, and jitter.
- p50/p95 latency.
- Results by instrument and occlusion type.
- Results against a simple geometric baseline on the same held-out clips.

Current pixel metrics are training diagnostics, not a final benchmark.
