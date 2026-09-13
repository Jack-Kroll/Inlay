# Transcribing a recording

Inlay combines three signals:

- Basic Pitch supplies sounding pitch and onset time.
- The dense detector supplies fretboard geometry.
- MediaPipe supplies fingertip positions.

The result is diagnostic, not validated ground-truth tablature.

## Run it

```sh
uv sync --locked --extra transcribe
uv run --extra transcribe python -m processing.music.transcribe \
  --model runs/dense/fretboard-v1/best.pt \
  --video clips/take-4.mov --output runs/transcribe/take-4 \
  --device mps --progress
```

Outputs:

- `<video>.overlay.mp4`: annotated video with audio.
- `<video>.tab.jsonl`: summary followed by one record per note.

Useful options:

- `--no-video`: write JSONL only.
- `--max-frames N`: analyze only the beginning of a clip.
- `--mirror`: mirror the rendered video only.
- `--audio FILE`: use aligned replacement audio.
- `--string-order normal|flipped`: override automatic orientation.
- `--tuning ...`: supply a custom tuning.

## Why it is offline

Basic Pitch consumes the soundtrack in windows rather than one camera frame at
a time. Inlay analyzes the video before rendering so it can choose string
orientation and plan hand position across the full clip. Live transcription
would need streaming pitch and causal decisions.

## Board coordinates

`processing/music/fretboard.py` fits a homography from image pixels to:

```text
(1 - 2**(-fret/12), across-board position)
```

Physical fret distance is projective under a camera view; raw fret number is
not. The fit uses measured nut/fret-wire endpoints. Estimated missing frets are
not used as anchors because they add no independent evidence.

Strings are interpolated evenly with an adjustable board-edge inset, default
`0.09`. The model does not detect strings, so inset error directly affects
string assignment.

## String orientation

Geometry cannot identify which end of a fret wire carries string 1. Automatic
mode assigns the whole clip both ways and keeps the orientation with more
finger-supported fretted notes. Short or sparse clips may require a manual
override.

## Assigning notes

Each pitch has zero or more valid `(string, fret)` positions for the configured
tuning. Inlay:

1. Groups onsets within 50 ms into chords.
2. Finds nearby board and fingertip evidence.
3. Plans hand position across the full clip.
4. Scores candidate positions.
5. Prevents simultaneous notes from sharing a string.
6. Flags unsupported, displaced, or impossible notes.

Evidence must be within 120 ms of an onset. A later re-pluck ends an earlier
note on the same string rather than treating it as a conflict.

Support values are:

| Support | Meaning |
|---|---|
| `fingered` | Fingertip supports the selected cell. |
| `open` | Open-string prior and an unoccupied string support it. |
| `position-only` | Only planned neck position supports it. |
| `no-hand` | No usable hand evidence. |
| `no-board` | No usable board geometry. |
| `rescued` | Experimental weak audio plus stable visual evidence. |

Unexplained fingertips are reported rather than hidden. They may be damping,
hovering, moving, or evidence of incorrect geometry.

## Tuning

`--tuning standard` means EADGBE. Custom values are MIDI pitches, highest string
first. Drop D is:

```sh
--tuning 64,59,55,50,45,38
```

The tuning is not detected. A capo or wrong tuning can produce confident but
incorrect output.

## Live string-grid preview

```sh
uv run python -m processing.vision.preview \
  --model runs/dense/fretboard-v1/best.pt --source 0 \
  --device mps --show-strings --show-heatmaps --inset 0.09
```

The string grid appears only when sufficient numbering and endpoint evidence
exists. Mirroring affects display, not saved coordinates or tab.

## Experimental visual rescue

Add `--visual-rescue` to inspect weak Basic Pitch activations that have stable,
exact-cell fingertip support:

```sh
uv run --extra transcribe python -m processing.music.transcribe \
  --model runs/dense/fretboard-v1/best.pt --video clips/take-4.mov \
  --output runs/transcribe/take-4-rescue --device mps --mirror --visual-rescue
```

Rescued notes are pink and include evidence in JSONL. The option preserves
baseline assignments and rejects conflicts. It remains experimental because a
visible finger does not prove that a string sounded.

## Known limitations

- Audio F1 is .779 on held-out GuitarSet; complete accuracy is unmeasured.
- Blind string assignment with perfect pitch is .698 overall.
- Fast 60-100 ms notes and strummed chords are difficult.
- Barre chords, bends, slides, hammer-ons, vibrato, and capos are not modeled.
- The picking hand is used only to identify the fretting hand.
- Strings are interpolated and open strings depend partly on a prior.
