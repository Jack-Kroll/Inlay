# Dataset audit and collection plan — 2026-09-06

The current dataset is a useful bootstrap, but it is not enough to establish
reliable performance on new guitars, marker designs, and occluded camera views.
More distinct instruments and recording sessions matter more than extra frames
from the same videos. No architecture change alone establishes higher accuracy.

## What is actually on disk

Source: [guitar by B101](https://universe.roboflow.com/b101/guitar-g65u6),
local export `guitar.v1i.yolov8-obb.zip`, with CC BY 4.0 attribution in its README.
This audits the local March 2025 export, not the potentially newer online version.

| Split | Images | Distinct source filenames | Fret / neck / nut labels |
|---|---:|---:|---|
| Train | 1,489 | 500 | 29,077 / 1,423 / 1,455 |
| Validation | 84 | 84 | 1,575 / 84 / 84 |
| Test | 42 | 42 | 809 / 42 / 42 |

- 1,615 images represent 626 filename families. These are **not** 626 verified
  independent guitars or videos. The export generated three rotation variants.
- All images were stretched to 512 × 512. Higher training resolution cannot
  recover lost detail or undo stretching without the original aspect ratios.
- There are no empty negative images. All validation/test images have a nut
  annotation; just 37 training images lack one. Annotation presence does not
  prove visibility, but no visibility/occlusion metadata is supplied.
- Twelve annotated samples were visually inspected. They include repeated
  performers/backgrounds, mostly acoustic/classical guitars, hands covering
  fret wires, several marker appearances, and some very small boards. This is
  a sample observation, not a measured distribution of all instruments.
- No identical filenames or identical image bytes crossed the original splits.
  A 63-bit DCT perceptual hash (distance ≤ 4) nevertheless found 16 cross-split
  pairs. Visual review of eight pairs confirmed closely related views of the
  same red guitar/player/background in validation and test.
- The importer groups related images transitively and excludes lower-priority
  split records (test > validation > train). This quarantines 42 records:
  prepared counts are **1,483 train / 48 validation / 42 test**. No original
  files are deleted or moved. Hash clustering is conservative triage, not proof
  that all session overlap has been removed.
- 68 boxes extend slightly beyond the image (coordinates ranged from about
  −0.0214 to 1.0660). These are explicitly clipped against the image boundary.
- 66 training images have fret labels but no neck box. Their neck loss is
  masked as unknown; the importer does not teach that these visible boards are
  negative neck examples.
- Labels describe coarse boxes. Derived neck polygons are weak segmentation
  targets; fret targets use the long-axis centerlines of boxes. They do not
  supply precise string intersections, fret identities, or visibility flags.

The older run's final row reported roughly 0.926 mAP50 and 0.579 mAP50–95.
These are detector metrics on its original split, not evidence of reliable fret
numbering, new-instrument generalization, or performance through occlusion.

## Additional datasets worth considering

| Source | What it adds | Decision |
|---|---|---|
| [guitar-fret-6pt v1](https://universe.roboflow.com/s-workspace-y3mjn/guitar-fret-6pt/dataset/1) | 926 images; keypoint task; 710/144/72 split, 640-square stretch, no offline augmentation; page declares CC BY 4.0 | Strong annotation-upgrade candidate. Inspect actual keypoint semantics and overlap before merging. Similar image counts to B101 are not proof of independence. The current importer does not interpret pose labels. |
| [Guitar Transcription Dataset](https://www.kaggle.com/datasets/jacksonlightfoot/guitar-transcription-dataset) | 355 fretboard polygon frames, 250 train / 105 test, COCO and VGG; separate 1,995-frame tablature set | Useful neck-only supervision after checking downloaded attribution. Frame filenames between its two subsets do not identify corresponding frames. COCO neck polygons can be imported now. |
| [Code and Chords guitar-frets-segmenter](https://universe.roboflow.com/code-and-chords/guitar-frets-segmenter) | 922 images, hand/neck and fret1–fret12 classes; page declares CC BY 4.0 | Inspect whether “fret” means wire or space between wires. Neck-only polygons can be reused; do not automatically turn region numbers into wire labels. |
| [Guitar Parts Detection by errai](https://universe.roboflow.com/errai-zca9d/guitar-parts-detection) | Explicit B101 extension adding 200 annotated images and a capo class | Not 1,126 independent new examples. Its own description restricts added content to noncommercial academic use despite a CC BY badge; exclude from the default app-training mix. |
| [GAPS](https://aim-qmul.github.io/GAPS/) | Diverse classical performance videos; paper describes 200+ performers | Potential source for a separately permitted, newly annotated research evaluation set. Not ready-made visual fret labels. The project states noncommercial research conditions. |

These are researched options, not downloaded or merged datasets. Exact licensing,
label conventions and content overlap must come from the selected export. The
new importer accepts locally supplied OBB sources and COCO neck polygons with
explicit class mappings and per-source attribution. No API key is required for
the existing dataset; additional Roboflow exports can be downloaded via its UI.

[GuitarSet](https://zenodo.org/records/3371780) is useful for future audio work;
its audio and musical fret annotations do not add camera-space fretboard labels.

## What to annotate and collect next

Use neck boundaries and **metal fret wires** as the primary visual evidence.
Use the nut as an optional absolute-number anchor. Inlays vary, may be absent,
and should not determine localization. Capos need their own labels or reviewed
hard examples so that the detector does not mistake them for nuts. String paths
need actual string intersections before claiming string-level accuracy.

For an initial expansion, aim for 3,000–5,000 distinct, reviewed frames across
30–50 guitars and multiple sessions each; these are collection targets, not a
promised sufficient sample size. Include dot, block, trapezoid, decorative and
no-marker boards; acoustic/electric/classical; light/dark wood; left-handed and
rotated views; phones; lighting/glare; blur; background clutter; capos; partial
crops; and real hands covering the nut, middle, or body end. Include 10–20%
reviewed negatives (no visible board, cases, furniture, strings without a board).

Keep recording ID and guitar ID; split by guitar for a new-instrument test and by
session for a same-instrument test. Never randomly split neighboring video frames.
Retain source resolution. Label visible wire centerlines and known occluded
geometry separately; mark uncertain endpoints unknown instead of inventing them.
Use a small held-out set with manually verified fret numbers, wire endpoints,
marker type and occlusion fraction. Keep multiscale/fanned frets separate: the
current equal-temperament projective decoder assumes conventional straight frets.

Measure wire precision/recall and endpoint error normalized by board width,
numbering error **and abstention rate**, false positives on negatives, loss and
reacquisition duration, jitter, and p50/p95 latency. Report by instrument/marker/
occlusion bucket, and compare the old checkpoint on those same videos. Current
pixel evaluation is useful for training checks but does not replace this benchmark.
