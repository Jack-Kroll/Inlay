# Tab assignment logic

The tab stage converts a sounding pitch into `(string, fret)`. A pitch often has
several valid positions, so this stage combines tuning rules, hand position,
and physical string constraints.

## Isolated evaluation

GuitarSet includes the string used for every annotated note. This command feeds
perfect reference pitches into Inlay without video:

```sh
uv run python -m processing.tools.eval_tab
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

## Audio's opinion about the string

The rules below work from musical context: which positions are playable, where
the hand is, and which strings are already taken. Over 62,476 notes with
perfect pitch they reach .698.

The [note verifier](note-verifier.md)'s string head answers the same question
from a completely different direction — the timbre in Basic Pitch's
posteriorgram and a constant-Q transform of the audio. Asked on its own it
places .879 of GuitarSet's held-out test player's notes on the right string.

Those are independent kinds of evidence about one question: audio hears tone
and cannot see the neck, the camera sees the neck and cannot hear which string
rang. So a note may carry a `string_posterior`, and `_score` mixes it in:

```text
geometry = CONTACT_WEIGHT*contact + POSITION_WEIGHT*nearness
score    = (1 - AUDIO_STRING_WEIGHT)*geometry + AUDIO_STRING_WEIGHT*posterior[string]
```

Measured on GuitarSet's validation player with no video, over the verifier's
own notes, sweeping the weight:

| Audio weight | String | Tab F1 | Fretted | Open |
|---|---:|---:|---:|---:|
| 0 (context only) | .625 | .511 | .646 | .248 |
| .10 | .715 | .584 | .706 | .875 |
| .20 | .861 | .704 | .861 | .868 |
| **.25 (default)** | **.867** | **.708** | **.866** | **.870** |
| .35 | .867 | .708 | .867 | .866 |
| 1.0 (audio only) | .862 | .704 | .861 | .868 |

The surface is flat from .20 to .35 and falls away below it, so the default is
the middle of the plateau rather than the grid argmax, for the same reason the
constants below are.

Open strings are where this earns its keep: .248 to .870. Nothing in the rules
below can tell an open B from the same pitch fretted elsewhere, and
`OPEN_REACH` is a prior standing in for evidence that was not available.
Timbre is that evidence. Fretted notes gain too, .646 to .866, but the open
case is the one that was structurally unsolvable.

Mixing beats audio alone by about half a point. Without a camera, "geometry"
here is only musical context, and context adds little to timbre; with a camera
in the real pipeline it contributes rather more, which this measurement cannot
show.

It is also the part of the verifier that generalises worst. On electric
guitars it gains 9 points rather than 23, and on two IDMT subsets it is worse
than context alone. See [note verifier](note-verifier.md).

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

Without timbre or reliable pressed-finger evidence the rules cannot distinguish
an open B from the same pitch fretted elsewhere. Timbre is what the audio
string posterior supplies, and open strings are where it helps most.

## Rejected ideas

- String-continuity state looked promising on the tuning subset but did not
  replicate on held-out data.
- Sustain duration barely separated open from fretted notes.
- Maximizing one grid-search result overfit a flat tuning surface, so the final
  constants use stable marginal values instead.

## Remaining limitations

- The audio string posterior is trained on acoustic guitar and should not be
  trusted on an unfamiliar instrument.
- Only neck position is planned globally; chord voicings are assigned greedily.
- Open-only passages inherit position from neighboring chords.
- Visual fingertip contribution is not measured against ground truth.
- Incorrect audio or visual evidence can make the full pipeline worse than this
  isolated result.
