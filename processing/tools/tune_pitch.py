"""Find the Basic Pitch settings that transcribe guitar best, using GuitarSet.

Every threshold Basic Pitch exposes is applied *after* the network, so the
model runs once per excerpt and each candidate setting is scored against the
same cached posteriorgram. A grid therefore costs one forward pass per file
rather than one per file and setting.

Notes are matched with mir_eval's transcription metrics: an estimate counts
when its onset lands within 50 ms of a reference onset and its pitch within
50 cents. Offsets are ignored, because tab needs to know when a note starts
and which pitch it is, not how cleanly it decays.
"""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import dataclass, asdict, field
import inspect
import io
import itertools
import json
from pathlib import Path
import time
import numpy as np

from processing.music import pitch as inlay_pitch
from processing.tools import guitarset

ONSET_TOLERANCE = .05
PITCH_TOLERANCE = 50.


@dataclass(frozen=True)
class Setting:
    onset_threshold: float = .6
    frame_threshold: float = .45
    min_note_ms: float = 60.
    minimum_frequency: float = inlay_pitch.MIN_FREQUENCY
    maximum_frequency: float = inlay_pitch.MAX_FREQUENCY
    melodia_trick: bool = True
    infer_onsets: bool = True

    def label(self):
        return (f'onset={self.onset_threshold:.2f} frame={self.frame_threshold:.2f} '
                f'min={self.min_note_ms:g}ms melodia={int(self.melodia_trick)} '
                f'infer={int(self.infer_onsets)} '
                f'freq={self.minimum_frequency:g}-{self.maximum_frequency:g}')


def _shipped():
    """Whatever ``detect_notes`` currently defaults to, read from its signature.

    A sweep always ranks the settings inlay ships today against the grid, and
    reading them here means the two can never drift apart.
    """
    defaults = inspect.signature(inlay_pitch.detect_notes).parameters
    return Setting(onset_threshold=defaults['onset_threshold'].default,
                   frame_threshold=defaults['frame_threshold'].default,
                   min_note_ms=defaults['minimum_note_length'].default,
                   minimum_frequency=defaults['minimum_frequency'].default,
                   maximum_frequency=defaults['maximum_frequency'].default)


CURRENT = _shipped()


@dataclass
class Tally:
    matched: int = 0
    estimated: int = 0
    referenced: int = 0

    def add(self, matched, estimated, referenced):
        self.matched += matched
        self.estimated += estimated
        self.referenced += referenced

    @property
    def precision(self):
        return self.matched/self.estimated if self.estimated else 0.

    @property
    def recall(self):
        return self.matched/self.referenced if self.referenced else 0.

    def fbeta(self, beta=1.):
        precision, recall = self.precision, self.recall
        weight = beta*beta
        denominator = weight*precision + recall
        return (1 + weight)*precision*recall/denominator if denominator else 0.

    @property
    def f1(self):
        return self.fbeta(1.)

    @property
    def f_half(self):
        """F0.5: a note inlay invents costs more than one it misses, because a
        spurious pitch takes a string away from a real note."""
        return self.fbeta(.5)

    def summary(self):
        return {'precision': round(self.precision, 4), 'recall': round(self.recall, 4),
                'f1': round(self.f1, 4), 'f_half': round(self.f_half, 4),
                'matched': self.matched, 'estimated': self.estimated,
                'referenced': self.referenced}


def posteriorgram(audio, model, cache=None, dtype=np.float16):
    """Model output for one excerpt, cached so a later sweep skips inference.

    Scoring an uncached float32 posteriorgram reproduces
    ``processing.music.pitch.detect_notes`` note for note. A float16 cache
    halves the size and was measured to move one note boundary in 180 by two
    frames, adding and removing none; that is well inside the 50 ms onset
    tolerance, but pass ``--cache-dtype float32`` to remove the question.
    """
    keys = ('note', 'onset', 'contour')
    if cache is not None and cache.is_file():
        with np.load(cache) as stored:
            if stored[keys[0]].dtype == dtype:
                return {key: stored[key].astype(np.float32) for key in keys}
    from basic_pitch.inference import run_inference
    # The Core ML path prints tensor diagnostics straight to stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        output = run_inference(str(audio), model)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{key: output[key].astype(dtype) for key in keys})
    return output


def estimate(output, setting):
    """Apply one setting to a cached posteriorgram, as ``predict`` would."""
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.note_creation import model_frames_to_time, output_to_notes_polyphonic
    # Basic Pitch takes the minimum note length in milliseconds and converts;
    # going straight to note creation means doing that conversion here.
    min_note_len = int(np.round(setting.min_note_ms/1000*(AUDIO_SAMPLE_RATE/FFT_HOP)))
    notes = output_to_notes_polyphonic(
        # The decoder applies frequency masks in place. Each setting needs its
        # own arrays or a narrow range poisons all subsequent candidates.
        output['note'].copy(), output['onset'].copy(), onset_thresh=setting.onset_threshold,
        frame_thresh=setting.frame_threshold, infer_onsets=setting.infer_onsets,
        min_note_len=min_note_len, min_freq=setting.minimum_frequency,
        max_freq=setting.maximum_frequency, melodia_trick=setting.melodia_trick)
    # Pitch bends are skipped: they never move a note's onset, offset or pitch,
    # so they cannot change a score, and estimating them dominates the sweep.
    times = model_frames_to_time(output['contour'].shape[0])
    last = len(times) - 1
    return [(float(times[min(start, last)]), float(times[min(end, last)]), int(midi))
            for start, end, midi, _ in notes]


def _arrays(notes):
    """``notes`` are (start, end, midi) triples; mir_eval matches in Hz."""
    intervals = np.array([[start, end] for start, end, _ in notes], dtype=float)
    pitches = np.array([440.*2**((midi - 69)/12) for _, _, midi in notes], dtype=float)
    return intervals, pitches


def counts(reference, estimated, onset_tolerance=ONSET_TOLERANCE,
           pitch_tolerance=PITCH_TOLERANCE):
    """Matched, estimated and reference note counts for one excerpt."""
    import mir_eval
    if not reference or not estimated:
        return 0, len(estimated), len(reference)
    reference_intervals, reference_pitches = _arrays(
        [(note.start, note.end, note.pitch) for note in reference])
    estimated_intervals, estimated_pitches = _arrays(estimated)
    matches = mir_eval.transcription.match_notes(
        reference_intervals, reference_pitches, estimated_intervals, estimated_pitches,
        onset_tolerance=onset_tolerance, pitch_tolerance=pitch_tolerance, offset_ratio=None)  # type: ignore[arg-type]  # mir_eval documents None to ignore offsets.
    return len(matches), len(estimated), len(reference)


def grid(args):
    """Every combination the command line asked for, current settings included."""
    axes = (args.onset, args.frame, args.min_note_ms, args.min_freq, args.max_freq,
            args.melodia, args.infer_onsets)
    settings = [Setting(*values) for values in itertools.product(*axes)]
    if CURRENT not in settings:
        settings.append(CURRENT)
    return settings


def sweep(excerpts, settings, cache_dir=None, progress=False, dtype=np.float16):
    totals = {setting: Tally() for setting in settings}
    styles = {}
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model
    model = None
    started = time.time()
    for index, excerpt in enumerate(excerpts, 1):
        cache = cache_dir/f'{excerpt.name}.npz' if cache_dir else None
        if model is None:
            model = Model(ICASSP_2022_MODEL_PATH)
        output = posteriorgram(excerpt.audio, model, cache, dtype)
        reference = excerpt.notes()
        for setting in settings:
            matched, estimated, referenced = counts(reference, estimate(output, setting))
            totals[setting].add(matched, estimated, referenced)
            styles.setdefault((setting, excerpt.style), Tally()).add(
                matched, estimated, referenced)
        if progress:
            elapsed = time.time() - started
            print(f'[{index}/{len(excerpts)}] {excerpt.name} '
                  f'({len(reference)} reference notes, {elapsed/index:.1f}s/excerpt)',
                  flush=True)
    return totals, styles


def report(totals, styles, excerpts, top=15):
    ranked = sorted(totals.items(), key=lambda item: -item[1].f1)
    place = {setting: rank for rank, (setting, _) in enumerate(ranked, 1)}
    header = f'{"#":>3} {"F1":>6} {"F0.5":>6} {"P":>6} {"R":>6}  setting'
    print(f'\n{len(totals)} settings over {len(excerpts)} excerpts '
          f'({sum(t.referenced for t in [ranked[0][1]])} reference notes)\n')
    print(header)
    print('-'*len(header))
    for rank, (setting, tally) in enumerate(ranked[:top], 1):
        print(f'{rank:>3} {tally.f1:6.3f} {tally.f_half:6.3f} {tally.precision:6.3f} '
              f'{tally.recall:6.3f}  {setting.label()}')
    current = totals[CURRENT]
    print('-'*len(header))
    print(f'{place[CURRENT]:>3} {current.f1:6.3f} {current.f_half:6.3f} '
          f'{current.precision:6.3f} {current.recall:6.3f}  {CURRENT.label()}'
          '   <- inlay defaults')
    best = ranked[0][0]
    print('\nBy playing style:')
    for setting, name in ((best, 'best'), (CURRENT, 'current')):
        for style in sorted({excerpt.style for excerpt in excerpts}):
            tally = styles.get((setting, style))
            if tally:
                print(f'  {name:<8} {style:<5} F1 {tally.f1:.3f} '
                      f'P {tally.precision:.3f} R {tally.recall:.3f}')
    return ranked


def numbers(text):
    return [float(value) for value in text.split(',')]


def flags(text):
    return [value.strip().lower() in ('1', 'true', 'yes') for value in text.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path('data/guitarset'),
                        help='Directory holding GuitarSet annotation/ and audio_mono-mic/')
    parser.add_argument('--limit', type=int, default=48,
                        help='Excerpts to use, spread over player and style; 0 uses all')
    parser.add_argument('--style', choices=['comp', 'solo'], help='Restrict to one style')
    parser.add_argument('--exclude', type=Path,
                        help='A previous run\'s JSON; its excerpts are held out of this one')
    parser.add_argument('--onset', type=numbers, default=[.3, .4, .5, .6, .7])
    parser.add_argument('--frame', type=numbers, default=[.1, .2, .3, .4, .5])
    parser.add_argument('--min-note-ms', type=numbers, default=[30., 50., 70., 100., 128.])
    parser.add_argument('--min-freq', type=numbers, default=[inlay_pitch.MIN_FREQUENCY])
    parser.add_argument('--max-freq', type=numbers, default=[inlay_pitch.MAX_FREQUENCY])
    parser.add_argument('--melodia', type=flags, default=[True])
    parser.add_argument('--infer-onsets', type=flags, default=[True])
    parser.add_argument('--cache', type=Path, default=Path('data/guitarset/posteriorgrams'),
                        help='Where to keep model output between sweeps; "none" to disable')
    parser.add_argument('--cache-dtype', choices=['float16', 'float32'], default='float16',
                        help='float32 caches reproduce detect_notes exactly, at twice the size')
    parser.add_argument('--output', type=Path, help='Write the full ranking here as JSON')
    parser.add_argument('--top', type=int, default=15)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    found, missing = guitarset.excerpts(args.root)
    if missing:
        print(f'{len(missing)} annotated excerpts have no audio and were skipped')
    if args.style:
        found = [excerpt for excerpt in found if excerpt.style == args.style]
    if args.exclude:
        # Settings chosen on a subset must be confirmed on excerpts that subset
        # never saw, because a flat optimum is easy to overfit.
        tuned_on = set(json.loads(args.exclude.read_text())['excerpts'])
        found = [excerpt for excerpt in found if excerpt.name not in tuned_on]
        print(f'Holding out {len(tuned_on)} excerpts used by {args.exclude}')
    chosen = guitarset.stratified(found, args.limit or None)
    settings = grid(args)
    cache_dir = None if str(args.cache).lower() == 'none' else args.cache
    print(f'{len(settings)} settings x {len(chosen)} excerpts')
    totals, styles = sweep(chosen, settings, cache_dir, args.progress,
                           getattr(np, args.cache_dtype))
    ranked = report(totals, styles, chosen, args.top)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'excerpts': [excerpt.name for excerpt in chosen],
            'onset_tolerance': ONSET_TOLERANCE, 'pitch_tolerance': PITCH_TOLERANCE,
            'current': {'setting': asdict(CURRENT), **totals[CURRENT].summary()},
            'ranking': [{'setting': asdict(setting), **tally.summary(),
                         'by_style': {style: styles[(setting, style)].summary()
                                      for style in ('comp', 'solo')
                                      if (setting, style) in styles}}
                        for setting, tally in ranked],
        }, indent=2))
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
