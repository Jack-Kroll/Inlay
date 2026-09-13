# Architecture

Inlay turns recorded guitar into editable musical events:

media -> visual observations -> pitch and tab inference -> editable music

Vision reports image-space evidence. It does not claim that a note sounded.
Audio and vision are joined later using media timestamps.

## Main pipeline

1. `FretboardNet` predicts three heatmaps: neck, fret wires, and nut.
2. `decode_maps` converts those heatmaps into board geometry.
3. `FretboardTracker` stabilizes valid geometry across short gaps using optical
   flow and a RANSAC homography.
4. Basic Pitch detects sounding pitches and onsets.
5. MediaPipe detects fingertips.
6. `processing/music` combines pitch, board geometry, and hand position to pick
   one `(string, fret)` per note.
7. The CLI writes an annotated video and diagnostic JSONL.

## Key design choices

### Dense geometry

The model uses a pretrained ResNet34 encoder and U-Net decoder. Its three
outputs are independent because neck and wire pixels may overlap.

### Geometry without decorative markers

Fret numbering uses the nut and equal-tempered fret spacing. It allows missing
wires and rejects ambiguous fits. A board can be detected without a nut, but
absolute fret numbers then remain unknown.

### Estimates remain separate from measurements

Short gaps between reliable frets may be estimated. These lines are marked as
estimates and never reused as evidence for later numbering.

### Tracking must have motion support

Tracking requires enough optical-flow inliers spread across the board. It
rejects extreme movement and expires after a short gap. This avoids freezing a
plausible but stale overlay.

### Physical board coordinates

Image points map to `(1 - 2**(-n/12), across)`. Physical fret distance is
projective under a camera view; raw fret number is not. Strings are currently
interpolated across the fitted board rather than detected.

### Musical constraints

Each pitch may have several guitar positions. The system scores them using
fingertips and planned hand position. Simultaneous notes cannot share a string.
A Viterbi pass plans neck position across the whole clip so missing frames can
use context on both sides.

## Current limits

- End-to-end transcription accuracy is not measured.
- The visual dataset needs more independent guitars, sessions, negatives, and
  real occlusions.
- Strings are interpolated; the model has no string head.
- A visible fingertip does not prove pressure or plucking.
- Barre chords, bends, slides, capos, and fanned frets are not modeled.
- Only one dominant fretboard is supported.
- Mobile export and latency testing remain future work.
- The mobile app is planned but not implemented.

`packages/music/performance.schema.json` is the boundary between Python
inference and a future editor. It represents timed performance events, not a
complete musical score.
