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
camera-space fretboard or fingertip labels. It is what the
[note verifier](note-verifier.md) is trained on: six players, acoustic guitars,
one microphone, about three hours.

IDMT-SMT-GUITAR adds a Fender Stratocaster, a Gibson Les Paul, an Aristides
and an Ibanez over four pickup settings, with onset, pitch, string and fret per
event — 5,164 usable notes in standard tuning. It is CC BY-NC-ND 4.0, so it is
used only to measure generalisation and never to train. It is also what showed
that the verifier's string head does not transfer off acoustic guitar.

Guitar-TECHS (Zenodo record 14963133, CC BY 4.0, 4.1 GB) is the electric
training data the string head lacked: three players on an Ibanez PF300, an EVH
Wolfgang and a Sire T7 through three amps in three rooms, 5.2 hours, every take
recorded four ways — direct input, a microphone on the amp, and head-mounted
and front-facing stereo microphones captured by a video device (audio only;
the video is not distributed). Labels come from a Fishman TriplePlay pickup,
one MIDI track per string, so the string is read off hardware. The content is
single notes at every fret of every string, scales in boxes, three- and
four-note chord sets, techniques (bends, harmonics, palm mute, vibrato), and
eight minutes of music. 29% of its notes sit at fret 12 or higher, against
GuitarSet's 1.2%. `processing.tools.guitar_techs` slices it into thirty-second
excerpts and renders each chain as its own domain; see
[the note verifier](note-verifier.md) for what it was worth, and that module's
docstring for why the two room-microphone chains carry only string labels.

EGFxSet (Zenodo record 7044411, CC BY 4.0, 5.8 GB) is every note of one
22-fret Stratocaster -- six strings, frets 0 to 22, 138 notes -- through each of
five pickup positions, clean and through twelve hardware effects, five seconds
a note, with string and fret in the file name. Inlay uses the clean set and the
three distortion pedals (Blues Driver, Tube Screamer, RAT): 2,759 notes. They
train the pitch front end, as single notes and mixed into chords
(`processing.training.frontend_data`), and can train the verifier's string head
as the `egfx` domain (`processing.tools.egfxset`). It is the one source here
with open strings on an electric in quantity, and the only real hardware
distortion. Delays, reverbs and modulation are not used. One file,
`TubeScreamer/Bridge-Middle/1-23`, has no clean counterpart (every other set
has a `1-17` there) and is skipped.

Other candidates looked at and not used: GOAT (5.9 h of direct-input electric
from many guitars, CC BY 4.0 but by request and research-only), EGDB (one
guitar, licence not stated where it is hosted), SynthTab (2 TB of synthesised
audio, licence unverified), and GAPS (classical guitar, YouTube links rather
than files).

## Use existing labels first

The [October 19 accuracy experiments](accuracy-experiments.md) downloaded
Kaggle's 1,995 labelled finger-placement images and GuiFiR's 1,030 video clips.
Kaggle's evaluated tablature subset remains locally. The unevaluated GuiFiR
archive and Kaggle fretboard subset were removed locally on October 21 to
recover disk space and can be downloaded again. The retained labels allow
visual evaluation without asking a user to label their own videos.
Kaggle's labels describe pressed fingers rather than sounding notes. GuiFiR
needs coordinate/convention checks and a source-overlap audit before training.
Keep their noncommercial dataset terms separate from Inlay's MIT source code.

## Optional future collection

If public data still leaves important gaps, target 3,000-5,000 reviewed frames
across 30-50 guitars and multiple sessions.
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
