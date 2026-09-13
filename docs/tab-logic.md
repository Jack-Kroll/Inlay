# Tab assignment logic

The tab stage converts a sounding pitch into `(string, fret)`. A pitch often has
several valid positions, so this stage combines tuning rules, hand position,
and physical string constraints.

## Isolated evaluation

GuitarSet includes the string used for every annotated note. This command feeds
perfect reference pitches into Inlay without video:

```sh
uv run --extra transcribe python -m processing.tools.eval_tab
```

This measures fallback string assignment only. It is not end-to-end accuracy
and does not test the visual model.

## Results

Over 62,476 notes, string accuracy improved from 41.2% to 69.8%:

| Version | String correct | `string-taken` | `no-free-string` |
|---|---:|---:|---:|
| Baseline | .412 | 5,522 | 768 |
| Current | .698 | 583 | 283 |

The current planner scored .692 on the 312 excerpts not used to tune its two
constants.

## Main rules

### Simultaneous notes cannot share a string

One string cannot produce two independently fretted pitches at the same time.
If a stronger note takes a preferred string, the displaced note is flagged.

A later re-pluck does not conflict with an earlier ringing note. It ends the
earlier note and adds `stopped-by-repluck`. GuitarSet has only 3 overlaps among
60,670 consecutive same-string note pairs.

### Plan hand position across the whole clip

`plan_positions` uses a Viterbi pass. Each state is a neck position; playable
notes reward a state, while movement costs `MOVE_PENALTY` per fret. Visible hand
observations pin their groups.

This lets a blind chord use context before and after it instead of making the
hand teleport or repeating earlier mistakes.

### A blank frame is not an open-string observation

When no fingertips are detected, the system gives no open-string contact credit.
Otherwise every open spelling would beat every unsupported fretted spelling.

### Open strings use a prior

An open note needs no finger, so finger distance is not evidence about it.
`OPEN_REACH = .85` is a calibrated prior, not a detector.

| Case | String accuracy |
|---|---:|
| Truly fretted notes | .713 |
| Truly open notes | .506 |
| Overall | .698 |

The system still cannot distinguish an open B from the same pitch fretted
elsewhere without timbre or reliable pressed-finger evidence.

## Rejected ideas

- String-continuity state looked promising on the tuning subset but did not
  replicate on held-out data.
- Sustain duration barely separated open from fretted notes.
- Maximizing one grid-search result overfit a flat tuning surface, so the final
  constants use stable marginal values instead.

## Remaining limitations

- Only neck position is planned globally; chord voicings are assigned greedily.
- Open-only passages inherit position from neighboring chords.
- Visual fingertip contribution is not measured against ground truth.
- Incorrect audio or visual evidence can make the full pipeline worse than this
  isolated result.
