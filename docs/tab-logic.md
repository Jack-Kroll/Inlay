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

Over 62,476 notes, string accuracy improved from 41.2% to 70.7%:

| Version | String correct | `string-taken` | `no-free-string` |
|---|---:|---:|---:|
| Baseline | .412 | 5,522 | 768 |
| Note by note | .698 | 583 | 283 |
| Whole onset | .707 | 1,077 | 40 |

Solving each onset at once is worth most here, where there is no timbre to
break a tie and the scores being compared are nearly flat: +.9 points, and
`no-free-string` falls from 283 notes to 40. It also displaces twice as many
notes from their own favourite spelling, which is the point — a displacement
is now a decision about the chord rather than whatever the note behind it
happened to leave free.

The note-by-note planner scored .692 on the 312 excerpts not used to tune its
two constants; that check has not been repeated.

## Audio's opinion about the string

The rules below work from musical context: which positions are playable, where
the hand is, and which strings are already taken. Over 62,476 notes with
perfect pitch they reach .707.

The [note verifier](note-verifier.md)'s string head answers the same question
from a completely different direction — the timbre in Basic Pitch's
posteriorgram and a constant-Q transform of the audio. Asked on its own it
places .882 of GuitarSet's held-out test player's notes on the right string.

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
| 0 (context only) | .653 | .538 | .669 | .365 |
| .10 | .738 | .608 | .730 | .885 |
| .20 | .857 | .706 | .855 | .885 |
| **.25 (default)** | **.862** | **.711** | **.861** | **.885** |
| .35 | .862 | .711 | .861 | .888 |
| 1.0 (audio only) | .858 | .707 | .856 | .892 |

The surface is flat from .20 to .35 and falls away below it, so the default is
the middle of the plateau rather than the grid argmax, for the same reason the
constants below are. It was re-measured, unchanged, after the assignment below
became a whole-onset one.

Open strings are where this earns its keep: .365 to .885. Nothing in the rules
below can tell an open B from the same pitch fretted elsewhere, and
`OPEN_REACH` is a prior standing in for evidence that was not available.
Timbre is that evidence. Fretted notes gain too, .669 to .861, but the open
case is the one that was structurally unsolvable.

Mixing is level with audio alone — .8624 against the head's own argmax at
.8617, which is six notes in eight thousand. Without a camera, "geometry"
here is only musical context, and on this player context adds nothing to
timbre that timbre did not already have; with a camera in the real pipeline it
contributes rather more, which this measurement cannot show.

It is also the part of the verifier that generalises worst. On electric
guitars it gains 12 points rather than 21, and on one IDMT subset it is worse
than context alone — badly so, 24 points. See
[note verifier](note-verifier.md#per-subset).

## Main rules

### Simultaneous notes cannot share a string

One string cannot produce two independently fretted pitches at the same time.
If a stronger note takes a preferred string, the displaced note is flagged.

A later re-pluck does not conflict with an earlier ringing note. It ends the
earlier note and adds `stopped-by-repluck`. GuitarSet has only 3 overlaps among
60,670 consecutive same-string note pairs.

### The whole onset is assigned at once

Which note gets a string used to be decided note by note, best-scoring first.
That is optimal for each note taken alone and not for the chord: a confident
mistake takes a string from the note that owned it, that note takes the next,
and one wrong nudge can respell a whole voicing. `_match_strings` instead
picks the combination with the highest total over the onset, so a mistake has
to outscore the rest of the chord rather than just the note it lands on.

It is exact, not another heuristic. All a note leaves behind is *which strings
are gone*, and a tuning has at most twelve, so a dynamic program over the
subsets of strings already taken solves an onset in well under a millisecond.
Notes it cannot place -- an onset with more detected notes than strings --
fall back to the old pass one at a time, which is also what lets two notes
inside the 50 ms chord window share a string when the first has already
stopped.

Measured with no video, over the verifier's own notes:

| | GuitarSet val | GuitarSet test | IDMT |
|---|---:|---:|---:|
| string, note by note | .861 | .880 | .634 |
| string, whole onset | .862 | .881 | .638 |
| tab F1, note by note | .710 | .794 | .518 |
| tab F1, whole onset | .711 | .794 | .521 |

On GuitarSet that is eight notes in eight thousand, because its string head is
so much stronger than musical context (.862 against .649) that the two rarely
disagree enough for the order to matter. The gain is in the case that does not
arise there, and is the reason for the change: see
[the note verifier](note-verifier.md#saturated-chords) for the strummed-chord
subset it was built for, where string accuracy goes .718 -> .749.

#### Reaching that case without spending the held-out corpus

The order notes are resolved in only matters when the two sources disagree,
and on GuitarSet they mostly do not. IDMT is where they do, and it is a
held-out corpus that should not be swept. So the regime was reached on
GuitarSet instead, by moving a share of the validation player's posteriors one
string over — a confident error rather than a hesitant one, because on IDMT
the string head is confidently wrong rather than uncertain, and flattening the
distribution would model the wrong failure. String accuracy on player 04:

| share of posteriors moved | audio alone | note by note | whole onset |
|---:|---:|---:|---:|
| none | .862 | .861 | .862 |
| .15 | .759 | .792 | .793 |
| .25 | .695 | .747 | .752 |
| .35 | .629 | .697 | .702 |
| .50 | .529 | .608 | .618 |

Context alone is .649 throughout, since nothing about it changes. The two
rules agree while the string head is reliable and separate as it stops being:
+.001 when it is intact, +.009 by the time it is down to .53. That is the
shape the change was chosen on, before IDMT was asked anything.

On undisturbed posteriors the same harness disposes of the other candidate
fix: deciding the onset jointly *and* tapering the weight scores .850, below
either rule on its own, and the taper is what costs it — see
[rejected ideas](#rejected-ideas).

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
| Truly fretted notes | .714 |
| Truly open notes | .614 |
| Overall | .707 |

Without timbre or reliable pressed-finger evidence the rules cannot distinguish
an open B from the same pitch fretted elsewhere. Timbre is what the audio
string posterior supplies, and open strings are where it helps most.

## Rejected ideas

- **Tapering the audio string weight as an onset fills up.** The idea was that
  a fully voiced chord has one plausible spelling, so musical context is
  nearly sufficient and audio should stand back. GuitarSet says the opposite:
  the more notes share an onset, the *better* the string head does, and the
  worse context does on its own.

  | notes in the onset | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---:|---:|---:|---:|---:|---:|
  | notes | 3,076 | 1,043 | 1,777 | 1,649 | 485 | 162 |
  | context alone | .475 | .751 | .756 | .770 | .730 | .648 |
  | audio alone | .843 | .805 | .860 | .896 | .944 | .963 |
  | fused at .25 | .839 | .803 | .867 | .902 | .924 | .920 |

  (Validation player, note by note, which is the rule the taper was proposed
  against.) A full voicing is where the audio term earns the most, not the
  least: at six notes it is worth 27 points over context. Taking the weight to
  zero as the onset fills costs .861 -> .846 overall and .920 -> .648 on
  six-note onsets. Raising it instead is worth nothing measurable (.862).
  Polyphony is not what the weight should depend on; how much to trust the
  string head is, and nothing available at that point measures it — gating on
  the head's own confidence was tried and rejected, see
  [the note verifier](note-verifier.md#rejected-ideas).
- **Normalising geometry and audio into distributions before mixing them.**
  Without video the geometry term is compressed into .15–.25 while the
  posterior spans 0–1, so a "weight" of .25 is in practice audio-dominant;
  normalising each into a distribution over the note's own feasible strings
  makes the weight mean what it says. It is worth about .002 on GuitarSet at
  its own best weight (.45, since the scale changes), and more under a
  degraded posterior. It was not shipped because it also rescales the
  *visual* evidence: a fingertip squarely on a cell currently scores .95
  against .25 for its rivals and cannot be overruled by audio at .25, and
  under normalisation a confident posterior would overrule it. Nothing
  measures the video path today, so this trades a measured small gain for an
  unmeasured risk to the part of the pipeline that matters most.
- String-continuity state looked promising on the tuning subset but did not
  replicate on held-out data.
- Sustain duration barely separated open from fretted notes.
- Maximizing one grid-search result overfit a flat tuning surface, so the final
  constants use stable marginal values instead.

## Remaining limitations

- The audio string posterior is trained on acoustic guitar and should not be
  trusted on an unfamiliar instrument.
- Neck position is planned across the clip and each onset's voicing is solved
  as a whole, but the two are decided in sequence, not together.
- Open-only passages inherit position from neighboring chords.
- Visual fingertip contribution is not measured against ground truth.
- Incorrect audio or visual evidence can make the full pipeline worse than this
  isolated result.
