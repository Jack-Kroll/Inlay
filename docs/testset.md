# The end-to-end test set

Every other number in this repository is audio against annotated audio. This
one runs the whole pipeline -- camera, audio, tab assignment -- on phone
recordings like the ones the app will see, and scores what comes out against
tab a person has checked. It is what tells whether a change helps in real use.

The tool is `processing/tools/testset.py`. Clips and labels live in
`data/testset/`, which is gitignored like the rest of `data/`.

## Record a clip

- Phone on a stand, landscape, the whole neck in frame **including the nut**:
  without the nut, fret numbers are unknown.
- Standard tuning, no capo, unless the label says otherwise.
- Play something you have the tab for, so the label can be written from the
  tab rather than from the pipeline's guess.
- 20 to 60 seconds. Shorter clips are easier to check note by note.
- Vary what the set covers, and say so in the file name: acoustic and electric,
  clean and distorted, single notes, chords and strums, open strings, high
  frets, hammer-ons and slides, quiet and loud rooms.

Fifteen to twenty clips is enough to see a change of a few points; the more
variety the better.

## Write its label

One text file per clip, `data/testset/<name>.tab.txt`:

```
# Comments start with #.
video: clips/Seven Nation Army.mov   # relative to the repository root, or absolute
tuning: standard                     # or drop-d, or six MIDI pitches / note names, high to low
capo: 0
@0.698 5:7                           # [@seconds] then string:fret per note
@1.459 5:7
@1.748 5:10
3:0 2:1                              # two notes at once
B3                                   # pitch known, position not
```

- **Strings are numbered as tab is read: 1 is the high E**, 6 the low E.
- **Frets count from the capo**, as tab is written with one.
- **One line per moment.** Notes on one line sound together; lines are in
  playing order.
- **Times are optional, all or nothing.** If every line has an `@seconds`, notes
  are matched by onset within 50 ms, as in every other benchmark here (widen it
  with `--tolerance` for hand-placed times). If any line has none, the label is
  a *sequence*: the transcription is aligned to it in order, which checks what
  was played and where, but not when. A sequence is the quick way to label a
  clip from a tab.
- A note name (`B3`, `F#2`) instead of `string:fret` counts for note F1 but not
  string accuracy, for a note whose position you are not sure of.

### Starting from a draft

```sh
uv run python -m processing.tools.testset draft "clips/Seven Nation Army.mov"
```

writes the pipeline's own transcription as a label, with times, marked
`status: draft`. Correcting it is quicker than typing a clip from scratch:
watch the video, fix every wrong string or fret, delete notes that were not
played, add the ones that were missed (with a time, or drop the times to make
it a sequence), then **delete the `status: draft` line**. `score` refuses
drafts: scored unchecked, a draft is the pipeline marking its own work. A
draft also nudges you towards the pipeline's answer, so where you have the
real tab, trust the tab.

Drafts for the six clips in `clips/` are already in `data/testset/`, written
from `runs/transcribe-final/`. They contain known mistakes -- Seven Nation
Army's G3 at 1.75 s is drafted as the open G string, `3:0`, where the riff is
played at `5:10` -- which is what correcting them is for.

## Score

```sh
uv run python -m processing.tools.testset score                 # runs the pipeline on every labelled clip
uv run python -m processing.tools.testset score --errors        # and lists every miss, extra and wrong string
uv run python -m processing.tools.testset score --predictions runs/transcribe-final   # scores transcriptions already on disk
uv run python -m processing.tools.testset score --report runs/testset/report.json
```

The pipeline runs with today's defaults, about a minute a clip on this Mac;
`--verifier` and `--verifier-threshold` try another configuration. Output:

| column | meaning |
|---|---|
| mode | `timed` (onsets matched within the tolerance) or `order` (aligned as a sequence) |
| F1, P, R | note F1, precision and recall: the right pitch at the right time or place |
| string | of the matched notes the label gives a position for, how many are on the right string |
| tab F1 | F1 counting a note right only if its pitch and string are both right |
| open | string accuracy on notes the label plays open, the error a reader notices first |
| offset ms | median transcribed-minus-labelled onset, for timed labels: a clock problem shows here first |

`all clips` sums the counts across clips, so a long clip weighs more than a
short one.

## Checks it has passed

- Drafts scored against the transcription they were written from: 1.000 on
  every column, and the pipeline run fresh by `score` reproduces a draft
  exactly.
- The October 19 transcriptions, made before the soundtrack was put on the
  video's clock, score near zero as timed labels -- every onset is about 80 ms
  early -- and agree with today's on 96% of notes and 91% of strings as
  sequences. That is agreement between two versions, not accuracy; accuracy
  needs checked labels.

## Keeping labels

Labels are small and slow to make. `data/` is gitignored because the clips
are personal recordings; if the labels should be versioned, keep them in a
tracked folder and point `--root` at it.
