# Keeping the notes and the fingers on the same clock

Two corrections to where and when inlay reads a fretting hand, found by
reading the note pipeline end to end on 2026-10-26. Both change what the camera
contributes; neither touches the audio benchmarks, which never see a frame.

## The soundtrack started late

Every clip in `clips/` starts its audio 71 to 95 ms after its first video
frame (`ffprobe` `start_time`: Blackbird .079 s, Loser .088, Seven Nation Army
.080, Sweet Child O Mine .082, take-4 .095, Thunderstruck .071). Players honour
that. `extract_audio` decoded the soundtrack to WAV with no regard for it, so
sample zero of the WAV was 79 ms into Blackbird, and every note was matched
against the frame 2-3 frames *before* it was played. Cross-correlating the old
WAV with one decoded on the container's timeline puts the old one exactly
79.2 ms early.

`extract_audio` now reads both streams' start times (`pitch.stream_offset`) and
pads (or trims) the soundtrack so sample zero is the first video frame. The
overlay's sound is muxed from the same WAV, so it moves with it. Video timing
itself was checked and is sound: the clips are constant 30 fps, and
`index/fps` drifts 1.7 ms over Blackbird.

Reading the hand slightly later still does not help: replaying the six clips
with the lookup 33 ms either side of the true onset gives lower fingertip
support at both (.837 before, .826 after, .841 at the onset).

## Fingertips sit past the fret

MediaPipe's fingertip landmark is the end of the finger, seen from above the
board, and the string is stopped by the pad behind it. On Kaggle's labelled
stills ([accuracy experiments](accuracy-experiments.md)), the located fingertip
fell past the pressed cell, towards the body, for 29% of pressed fingers on the
development recordings and 73% on the held-out ones, and short of it for 0.7%
and 9%. That is a constant distance, which becomes a whole fret where frets are
narrow, which is why the high-fret held-out recordings suffer most.

`fusion.FINGERTIP_SETBACK` moves each located fingertip back towards the nut by
a fixed fraction of scale length before its fret is read. Chosen on the
development recordings alone, on the measure the docs already report: string
accuracy when the pitch is given.

| setback (scale length) | dev placement | dev exact fret | dev exact cell |
|---:|---:|---:|---:|
| 0 | .7949 | .4460 | .4343 |
| .008 | .7960 | .5406 | .4694 |
| **.010** | **.7967** | .5522 | .4789 |
| .014 | .7960 | .5668 | .4917 |
| .020 | .7945 | .5491 | .4789 |

Scored once on the held-out recordings at .010:

| | placement | exact fret | exact cell |
|---|---:|---:|---:|
| before | .6726 | .0992 | .0636 |
| setback .010 | .6840 | .2028 | .1424 |

The fingertip's own fret is right twice as often. Placement moves only a point
because, given the pitch, the other spellings are four or five frets away and a
one-fret error rarely chooses between them. The fret matters more where it is
read directly: whether a note counts as fingered, and the hand position the
planner uses for chords where no finger is on a candidate. Held-out exact fret
is still 20%; most of what remains is fret numbering on high frets, which this
does not address. Kaggle is one player on one guitar, under research-only
terms; it chose one constant here and nothing else.

## On the clips

Replaying the six clips' tab stage on their cached vision with the same audio
notes. Fingertip support is the camera's own opinion, not accuracy; there is
still no reference tab for these clips.

| clip | before | offset | setback | both | placements changed |
|---|---:|---:|---:|---:|---:|
| Blackbird | .692 | .808 | .724 | .828 | 7/38 |
| Loser | .931 | .897 | .931 | .862 | 0/29 |
| Seven Nation Army | .891 | .894 | .891 | .938 | 3/53 |
| Sweet Child O Mine | .906 | .922 | .906 | .953 | 2/64 |
| take-4 | .741 | .811 | .786 | .833 | 5/78 |
| Thunderstruck | .717 | .714 | .703 | .720 | 17/95 |
| mean | .813 | .841 | .824 | .856 | |

Thunderstruck's riff alternates an open B with fretted notes on the same
string. Read 70 ms early, the previous note's finger is still down; on the
right clock 45 of its notes are placed open, against 35 before, and the clip
is documented as having 48 open B3s. Loser loses support with no placement
changing.

## Also checked, and left alone

- Chord grouping: 97% of GuitarSet player 04's strums of three or more notes
  span under 50 ms (median 17 ms), inside `CHORD_WINDOW`.
- The verifier's patch geometry against the front end: the contour head puts a
  pitch on the middle of its three sub-bins, which is where `_fold` centres it.
- Legato on electric guitar: on Guitar-TECHS's held-out player, notes that
  follow straight on from the previous note on the same string are found 20% of
  the time, against 72% for the rest, and are a fifth of the notes. Hammer-ons
  and pull-offs have no pick attack, but a MIDI pickup also reports a slide as
  a run of notes, so some of this is the labels. Not fixed; it needs a look at
  the audio.
