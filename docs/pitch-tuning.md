# Measuring and tuning the audio stage

The **audio** stage has annotated ground truth and a measured score. Blind
string assignment is evaluated separately in [tab logic](tab-logic.md).
The fret geometry, fingertips, and combined audio/video pipeline still need
annotated evaluation.

## Ground truth

[GuitarSet](https://zenodo.org/records/3371780) (CC BY 4.0) is 360 thirty-second
excerpts of solo acoustic guitar — six players, five progressions, comp and solo
styles — with one `note_midi` annotation per string. That gives 62,476 annotated
notes carrying onset, offset, sounding pitch **and the string it was played on**.

Download `annotation.zip` and `audio_mono-mic.zip` and unzip both into
`data/guitarset/`:

```sh
mkdir -p data/guitarset && cd data/guitarset
curl -fL -o annotation.zip 'https://zenodo.org/records/3371780/files/annotation.zip?download=1'
curl -fL -o audio_mono-mic.zip 'https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1'
unzip -q annotation.zip -d annotation && unzip -q audio_mono-mic.zip -d audio_mono-mic
```

The mic channel is used rather than either pickup channel because it is the one
that resembles how inlay actually records: one microphone in a room, not a
hexaphonic pickup. `data/` is gitignored, so nothing here enters the repo.

The strings are numbered from the low E: GuitarSet's `data_source` 0 to 5 are
E2 A2 D3 G3 B3 E4, which is the reverse of the highest-string-first order
`--tuning` takes. `processing/tools/guitarset.py` holds that convention in one
place.

## What is measured

A detected note counts as correct when its onset is within 50 ms of a reference
onset and its pitch within 50 cents, using `mir_eval`'s note transcription
matching. Offsets are ignored: tab needs to know when a note starts and what it
is, not how cleanly it decays, and Basic Pitch's offsets are the least reliable
thing it reports.

Scores are micro-averaged over excerpts — every note weighs the same, so a dense
comp excerpt counts for more than a sparse solo one, which is the right weighting
when the question is how many notes come out right overall.

Both F1 and **F0.5** are reported. F0.5 weights precision double, which is closer
to what tab actually wants: a note inlay invents does not merely appear in the
output, it takes a string away from a real note under the one-pitch-per-string
rule in `processing/music/transcribe.py`.

## Running a sweep

```sh
uv run --extra transcribe python -m processing.tools.tune_pitch \
  --limit 48 --output runs/tune-pitch/coarse.json --progress
```

Every threshold Basic Pitch exposes is applied *after* the network, so
`tune_pitch` runs the model once per excerpt and scores every candidate setting
against the same cached posteriorgram. A 180-setting grid therefore costs 48
forward passes, not 8,640. Model output is cached under
`data/guitarset/posteriorgrams`, so a later sweep over a refined grid skips
inference altogether.

Inference is not the expensive part on this machine — about 0.5s per
thirty-second excerpt through Core ML, against roughly 0.03s to apply and score
one setting. Grid size, not excerpt count, is what a sweep costs.

Scoring an uncached float32 posteriorgram reproduces `pitch.detect_notes` note
for note, so the harness measures exactly what inlay runs. The cache defaults to
float16, which halves it; that was measured to shift one note boundary in 180 by
two frames while adding and removing none, comfortably inside the 50 ms onset
tolerance. `--cache-dtype float32` removes the question.

`--limit` takes a subset spread evenly over player and playing style rather than
a contiguous slice, which would weight the result towards whichever players sort
first. `--limit 0` uses all 360.

A caveat that turned out not to matter: GuitarSet can have the same pitch
sounding on two strings at once, which Basic Pitch cannot report twice. Only
160 of 62,476 notes (0.26%) are such unisons, so recall is capped at 99.7% —
far above anything measured here.

## What the sweep found

Two passes over a 48-excerpt tuning subset — a 180-setting coarse grid, then
865 settings refined around its peak — followed by a confirmation run on the
**312 excerpts the tuning subset never saw** (54,765 notes).

The axes are smooth and well separated. Averaged over everything else:

| `frame_threshold` | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|
| mean F1 | .772 | .782 | **.787** | .786 | .780 | .767 | .745 |

| `onset_threshold` | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|---|
| mean F1 | .775 | .780 | **.781** | .778 | .771 | .760 |

| `min_note_ms` | 40 | 50 | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|
| mean F1 | .781 | **.783** | .782 | .776 | .768 | .756 |

`frame_threshold` dominates, and the old 0.3 sat on the wrong side of its peak.
`melodia_trick` and `infer_onsets` moved F1 by less than 0.004 either way, which
is inside the noise, so both keep Basic Pitch's default of on. Turning
`melodia_trick` off trades about a point of recall for a point of precision if
that is ever wanted.

### Held out, 312 excerpts, 54,765 notes

| Setting | F1 | F0.5 | Precision | Recall |
|---|---|---|---|---|
| **0.60 / 0.45 / 60 ms** (now the default) | **.779** | .789 | .796 | .762 |
| 0.65 / 0.50 / 60 ms (precision-first) | .777 | **.811** | **.836** | .726 |
| 0.50 / 0.30 / 70 ms (the old default) | .740 | .696 | .670 | .827 |

The old defaults ranked **last of the 37 settings tested**. Their problem was
never recall — at .827 it is the highest of the three — but precision: a third
of everything they reported was not played. The new defaults trade 6.5 points
of recall for 12.6 points of precision.

The top of the grid is a plateau, not a peak: about twenty settings sit within
0.002 F1 of each other, so the argmax on any one subset is noise. 0.60/0.45/60
was picked because it ranks 34th of 865 on the tuning subset **and** 1st of 37
held out — good on both rather than best on either.

### Comp is much harder than solo

| | F1 | Precision | Recall |
|---|---|---|---|
| solo | .868 | .828 | .911 |
| comp | .742 | .782 | .706 |

Single-note lead lines transcribe well. Strummed chords do not, and the gap is
mostly recall: notes inside a six-string voicing get lost. Any accuracy figure
quoted for inlay should say which of these it refers to.

## What this does and does not establish

It measures the **audio stage alone** — sounding pitch, against a microphone
recording of an acoustic guitar in standard tuning. It says nothing about the
fret geometry, the fingertip stage, or the string assignment that turns these
notes into tab, and GuitarSet's electric-guitar and alternate-tuning coverage is
nil. The per-string annotations needed to measure the tab stage are sitting in
the same files and are evaluated separately in [tab logic](tab-logic.md).


## Quiet and quick notes: preprocessing versus decoding

Inlay extracts a mono 22,050 Hz WAV from video. It does not currently apply
compression, denoising, EQ, a noise gate, or waveform gain normalization.
Basic Pitch itself loads/resamples mono audio, processes overlapping windows,
and computes a constant-Q spectrogram with logarithmic power normalization
inside the model. This is feature normalization, not automatic recovery of
quiet notes. See Spotify's [inference code](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/inference.py),
[model](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/models.py), and
[normalization layer](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/layers/signal.py).

Start by testing decoding settings on the same recording. Add these flags to
the normal transcription command as an experiment, not a new calibrated preset:

```sh
--onset-threshold 0.5 --frame-threshold 0.35 --min-note-ms 40
```

The current defaults are 0.60 / 0.45 / 60 ms. Lower thresholds can admit weaker
model activations; a shorter minimum duration can preserve brief notes. These
are model activation thresholds, not microphone volume levels. Basic Pitch's
256-sample hop is about 11.6 ms at 22,050 Hz, and the duration cutoff is rounded
to frames, so it is not an exact millisecond boundary. Its temporal model and
onset decoding can still miss or merge rapid notes.

If threshold changes do not help, compare an unprocessed recording with gentle
compression or targeted rumble removal using `--audio`. Keep audio aligned to
the original video: no silence trimming or speed changes. Whole-file gain alone
does not improve signal-to-noise ratio, and the internal normalization already
reduces level variation. Aggressive compression can blunt attacks; denoising
and gates can erase the quiet tails or notes we want to retain. These are
hypotheses to test on labeled clips, not established accuracy improvements.
The rendered overlay currently uses the same audio supplied for inference.

## Could visual candidates rescue the missed notes?

Lowering the global thresholds to catch quiet notes costs more precision than
it buys recall -- that is what the sweep above measured. A rescue stage inverts
the question: the camera proposes a pitch from where the fingers are, and the
audio is only asked to confirm it, at a threshold no global detector could
afford because it applies at one pitch and one instant rather than everywhere.

Nothing extra has to be computed for this. Basic Pitch's thresholds are applied
*after* the network, so the activations for every pitch it rejected are already
in the forward pass inlay runs. `detect_notes` now returns them alongside the
notes as a `Posteriorgram`, and `Posteriorgram.evidence(pitch, start, end)`
reports the peak onset and frame activation for one pitch over one window.
Confirming a candidate is an array lookup, not a second pass over the waveform.

Whether the stage is worth building is measured by
`processing/tools/rescue_bound.py`, which stands GuitarSet's per-string
annotations in for a **perfect** fretting-hand tracker. Candidates carry a
window rather than an onset -- a camera sees a finger arrive, not a pluck -- so
the audio still has to locate the attack, and a rescued note whose onset lands
more than 50 ms off scores as a false positive like any other.

```sh
uv run --extra transcribe python -m processing.tools.rescue_bound \
  --limit 48 --visual-error 0.10 --output runs/rescue-bound/oracle.json
```

### 48 excerpts, 7,711 annotated notes

Only 74 of the 1,898 missed notes are open strings, so a fingertip-driven stage
that cannot propose an open string still has a recall ceiling of .990. Open
strings are not what limits this.

With a perfect tracker:

| Gate | F1 | F0.5 | Precision | Recall | Rescued |
|---|---|---|---|---|---|
| baseline (no rescue) | .797 | **.825** | **.845** | .754 | - |
| onset .25 / frame .15 | **.846** | .837 | .831 | **.861** | 1,105 |
| onset .30 / frame .20 | .840 | .835 | .832 | .849 | 983 |
| onset .45 / frame .35 | .822 | .830 | .835 | .810 | 606 |

Recall moves .754 -> .861 for about a point and a half of precision, without
touching the global thresholds: the gate only fires where the camera already
points.

### The visual stage, not the audio, is the binding constraint

`--visual-error` misreads a fraction of candidates by one string or one fret,
which is how interpolated string positions actually fail. F1 at the loosest
gate:

| Visual error | F1 | F0.5 | Precision | Recall |
|---|---|---|---|---|
| 0% | .846 | .837 | .831 | .861 |
| 10% | .835 | .826 | .820 | .851 |
| 25% | .822 | **.813** | .807 | .839 |

F1 stays ahead of the .797 baseline throughout, but **F0.5 -- the measure this
document argues tab actually wants -- barely moves, and at 25% visual error
with a loose gate it drops below the baseline.** Rescue buys recall with
precision, and F0.5 charges for precision twice.

### Fingers that are down but not plucked settle it

Everything above assumes the tracker only ever proposes notes that were
actually *played*. No tracker can do that. A camera sees fingertips, not
plucks, so it proposes every finger on the board -- anchoring, damping,
mid-shift -- which is exactly what inlay already reports as unexplained
contacts. `--spurious K` adds K such non-sounding candidates per played note
and makes the gate reject them on audio alone:

| Spurious per note | Best F1 | Best F0.5 | Rescued | Wrong |
|---|---|---|---|---|
| 0 | .835 | .826 | 999 | 124 |
| 1 | .816 | .821 | 739 | 267 |
| 2 | .810 | **.814** | 764 | 476 |

**With even one unplucked finger per note, F0.5 is below the .825 baseline at
every gate**, and one is conservative: a fretting hand normally has two to four
fingers down for one plucked string. At two, the loosest gate admits 1,885
wrong notes against 992 right ones.

F1 still improves by one to two points, but F1 is the measure that treats an
invented note as no worse than a missed one -- which is the tradeoff this
document rejected when the thresholds were set, for the same reason: a spurious
pitch takes a string away from a real note.

### Original conclusion: not built

The rescue is not worth implementing as things stand. It buys recall the
project has already decided it does not want to buy at this price, and the
audio gate cannot tell a plucked string from a held one because that difference
is not in the audio at the quiet end -- which is the whole population being
rescued.

What would change the answer is a candidate stream that proposes only *plucked*
fingers. That is a harder problem than the one being solved: the hand cannot
distinguish a fretted note from a damped or hovering one
([transcription](transcription.md)), and the picking hand is currently used
only to decide which hand frets. Until a candidate can carry evidence that a
string was actually sounded, the gate is being asked to reconstruct from audio
the very thing the audio lost.

`rescue_bound.py` stays in the tree so the decision can be re-measured if the
fingertip stage ever gains that evidence.


Frequency sweeps now copy note/onset arrays before decoding because Basic
Pitch masks frequency bins in place. Earlier saved coarse/fine/holdout runs all
used the same 70–1400 Hz bounds, so this fix does not invalidate those rankings.
The 312-excerpt set was consulted when choosing the final default, so its score
is validation evidence, not a pristine final test or an unseen-player holdout.

A later opt-in `--visual-rescue` experiment is now implemented; see
[transcription](transcription.md#experimental-finger-confirmed-weak-audio).
It adds temporal fingertip agreement and preserves baseline assignments. The
results above still apply to the original oracle experiment, not this new gate.
