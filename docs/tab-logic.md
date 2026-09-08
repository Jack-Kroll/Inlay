# Measuring and improving the tab stage

The audio stage now has a number (see [pitch tuning](pitch-tuning.md)). This is
the same treatment for the stage after it: turning a sounding pitch into a
`(string, fret)`.

## How it can be measured without video

GuitarSet annotates the string every note was played on, so feeding its notes
straight into `transcribe_notes` with no frames removes the detector and the
hand tracker from the picture. What is left is the spelling logic alone: which
of a pitch's several positions it picks, and how it arbitrates a string that two
notes both want.

```sh
uv run --extra transcribe python -m processing.tools.eval_tab
```

This isolates the fallback assignment logic with perfect reference pitches.
It is **not a lower bound on end-to-end accuracy**: real pitch errors and
incorrect visual evidence can both make the combined pipeline worse. It says
nothing about whether the fret geometry itself is right.

## What it found, and what changed

Baseline: **41.2%** of notes placed on the string the player actually used, over
62,476 notes. Four changes took that to **69.8%**, and cut the notes displaced
off their preferred string by 89%.

| | string correct | `string-taken` | `no-free-string` |
|---|---|---|---|
| baseline | .412 | 5,522 | 768 |
| + re-pluck stops a ringing note | .411 | 4,047 | 553 |
| + neck position carried between chords | .516 | 2,530 | 328 |
| + no open-string credit from a blank frame | .558 | 1,696 | 541 |
| + position planned for the whole clip | **.698** | **583** | **283** |

Confirmed at **.692** on the 312 excerpts the two planner constants were not
tuned on.

### A re-pluck stops the note already on that string

One string sounds one pitch, so two notes may not share a string. The old rule
drew the wrong conclusion from that: a note still *ringing* from an earlier
onset blocked its string, and a later note wanting it was displaced or reported
as `no-free-string`. But plucking a string again is exactly what stops the note
already on it.

GuitarSet settles it — of 60,670 consecutive same-string note pairs, **3**
overlap at all. So the earlier note is now truncated to the new onset and
flagged `stopped-by-repluck`, and only notes sounded *together* compete.

This barely moved accuracy on its own, because the scoring underneath it was the
real problem, but it removed about 1,500 spurious displacements — and it matters
most in fast passages, where a ringing note overlaps the next onset. 57% of
detected notes are still sounding when the next note starts.

### The neck position is planned for the whole clip at once

`_hand_position` returns `None` when no fingertips are located, which removed the
position term entirely and left the choice to "open string, else lowest fret" —
deciding every blind chord in isolation, as though the hand teleported.

The notes just played are evidence about where the hand is. Carrying a median
fret forward as a fallback lifted this to .516 — but a backward-looking average
can only average frets it has already guessed, so its own errors feed on
themselves.

`plan_positions` replaces it with a Viterbi pass over the whole clip. States are
whole frets, the emission at each position is how comfortably that group's
pitches sit under a hand there, and shifting costs `MOVE_PENALTY` per fret. A
group where the hand was actually seen is pinned to it, so the search only fills
the gaps between real observations — a blind chord is now answered by the
chords on *both* sides of it, and by the nearest frame that saw a hand.

That is worth **+14 points** over the memory it replaced: .558 to .698.

Both constants were taken from the marginal peak on a 48-excerpt tuning subset,
not the grid argmax — the surface is flat near the top, and the argmax there
(0.5/1.75, scoring .753) fell to .676 held out while the marginal peak held
.692.

| `MOVE_PENALTY` | 0.2 | 0.275 | 0.35 | 0.425 | 0.5 |
|---|---|---|---|---|---|
| mean correct | .7412 | **.7464** | .7431 | .7429 | .7348 |

| `REACH_WIDTH` | 1.5 | 1.75 | 2.0 | 2.25 | 2.5 |
|---|---|---|---|---|---|
| mean correct | .7379 | **.7478** | .7445 | .7434 | .7350 |

A penalty of zero scores .589 and one of 20 scores .550: with no cost the hand
teleports to whatever explains each chord best, and with too much it never
moves. `OPEN_NEARNESS` was re-checked at the same time and 0.5 is still its best
value, so it was left alone.

### A blank frame is not evidence of an open string

An open string was credited when nothing sat on it — but with *no* fingertips
located, "nothing is sitting there" is a blank frame, not an observation. The
module already says as much about missing fingertips; it just did not apply it
to the empty case.

Worse, the credit was decisive: any value above zero made every open spelling
outrank every fretted one, whose contact term is also zero when blind. 0.2, 0.35,
0.5 and 0.7 all score identically. Only zero changes the ordering, and it is
also the honest value.

## Open strings

Open notes were by far the worst case, and the overall figure hid it completely:
they are 6.8% of GuitarSet, so a stage can refuse to place them at all and still
look good.

| | notes | string correct |
|---|---|---|
| truly open | 4,247 | **.139** |
| truly fretted | 58,229 | .739 |

The stage said "open" 611 times when the truth was 4,247, and was right 96.7% of
the times it did. That is not a wobble, it is a refusal — and worse, the
`BLIND_OPEN_CREDIT = 0` change above *caused* part of it. That change raised the
overall figure from .516 to .558 by trading away a class worth 6.8% of the data.
`eval_tab` now always reports open and fretted separately for that reason.

### Two things that did not work

**String continuity.** A run on one string should make an open note on that
string likely — the Thunderstruck intro alternates the open B with frets on the
same string. Adding a string axis to the planner, so a state is a neck position
*and* the string the playing is centred on, looked convincing on the tuning
subset: comping improved on open *and* fretted notes at once. It did not
replicate. That subset holds only 151 open notes, and on the held-out 4,096 the
gain collapsed into the same trade as every other knob. The code was reverted.

**Sustain.** Open strings ought to ring longer. Comparing the same pitch played
open and fretted, the median duration ratio is 1.00, and the best single
threshold separates them with balanced accuracy .540 against .500 for a coin.
Note length is set by the music, not by the string.

### What did work, and what it is

Fret 0 was being scored on how near the fingers were, through `OPEN_NEARNESS`.
That is the wrong question: an open string needs no hand, so distance from the
fingers is not evidence about it. `OPEN_REACH` replaces it in both the planner
and the scorer, held constant across the neck.

Its value is then a **prior, not a measurement**, and it was set by calibration
rather than by maximising accuracy — at .7 the stage called open a fifth as
often as it should have:

| `OPEN_REACH` | overall | open | fretted | says "open" |
|---|---|---|---|---|
| 0.70 | .690 | .196 | .730 | 0.21x the true rate |
| **0.85** | **.698** | **.508** | .713 | 0.67x |
| 0.90 | .693 | .833 | .681 | 1.50x |

Open accuracy went from **.139 to .506** with the overall figure unchanged at
.698. On a real clip of the Thunderstruck intro this moved seven of nine open Bs
onto the right string; the two that stayed wrong are the ones where a fingertip
sits nearest that string.

Be clear about what this is. Nothing available to this stage can distinguish an
open B from the same pitch fretted elsewhere, because they are the same pitch.
Calibrating the prior stops the stage being systematically wrong in one
direction; it does not detect anything. Real detection needs either spectral
timbre from the audio, which is untested here, or hand evidence that can tell a
pressed finger from a hovering one, which the fingertips cannot.

## What is still wrong

**Only the position is planned globally; the spelling is not.** The Viterbi
chooses where the hand is, and `assign_positions` then places notes greedily
within each chord under that constraint. Planning over full chord voicings
rather than a single position per group would subsume both, at a much larger
state space. With 0.9% of notes still displaced, the headroom is now small.

**An open string tells the planner nothing about position**, which is correct —
it is playable from anywhere — but it means a passage of only open strings
inherits its position from its neighbours rather than having one. That is the
intended behaviour, not a bug, but it does mean the planner is confident about
positions it has no evidence for.

**Open-string choice still depends on a prior.** `OPEN_REACH = .85` is constant
across the neck, but strong fingertip evidence for a fretted spelling can still
outweigh it. The model cannot establish whether that finger is pressing.

**This is measured blind.** How much the fingertips add is unknown, because the
fret geometry and the fingertip stage still have no ground truth of their own.
