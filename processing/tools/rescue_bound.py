"""How much could a visual-candidate rescue stage recover? An upper bound.

Basic Pitch decides what a recording contains from the audio alone, and on
guitar it loses quiet notes inside chords. A rescue stage would invert that:
the camera proposes a pitch from where the fingers are, and the audio is asked
only to confirm it, at a threshold far below the one a global detector can
afford. Nothing about that is free -- every wrong visual candidate that the
audio happens to support becomes a confident wrong note.

This measures the ceiling before the stage is built. GuitarSet annotates the
string each note was played on, so it can stand in for a *perfect* fretting-hand
tracker: candidates come from the annotations rather than from inlay's own
geometry. Whatever this reports is therefore the best case. `--visual-error`
degrades the oracle towards a real tracker by misreading a fraction of
candidates, one string or one fret out, which is how inlay's interpolated
string positions actually fail.

    uv run --extra transcribe python -m processing.tools.rescue_bound \\
      --limit 48 --output runs/rescue-bound/oracle.json --progress

Inference is skipped entirely when `processing.tools.tune_pitch` has already
cached the posteriorgrams, because the gate is applied to the same arrays a
sweep scores.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np

from processing.music.pitch import Posteriorgram
from processing.tools import guitarset, tune_pitch

# How a fretting-hand tracker gets a note wrong. Reading the wrong string moves
# the pitch by the interval between neighbouring strings -- four semitones
# across the G-B pair, five everywhere else -- and reading the wrong fret moves
# it by one. Strings are interpolated across the detected board rather than
# detected, so the string errors are the ones inlay is most exposed to.
VISUAL_ERRORS = (-5, -4, -1, 1, 4, 5)
# A rescued note needs an offset to be an interval at all. Offsets are ignored
# when scoring, so this is presentation, not a claim about decay.
RESCUED_DURATION = .15


@dataclass(frozen=True)
class Gate:
    """The evidence a visual candidate must find before it is believed."""
    onset: float
    frame: float

    def label(self):
        return f'onset>={self.onset:.2f} frame>={self.frame:.2f}'

    def passes(self, evidence):
        return evidence.onset >= self.onset and evidence.frame >= self.frame


@dataclass(frozen=True)
class Candidate:
    pitch: int
    start: float   # When the finger is first down, the earliest a pluck could be.
    end: float     # When it lifts, the latest.
    truthful: bool  # False once --visual-error has misread it.


def _sounding(reference):
    """Pitch -> spans it actually sounds, so a decoy can be checked against it."""
    spans = {}
    for note in reference:
        spans.setdefault(int(round(note.pitch)), []).append((note.start, note.end))
    return spans


def spurious(reference, rng, per_note, lead, spans):
    """Candidates for fingers that are down but not sounding.

    A tracker cannot see a pluck. It proposes every fingertip on the board, and
    most of them are anchoring, damping or mid-shift -- inlay already reports
    these as unexplained contacts. They are the dominant candidate the oracle
    above never generates, and the gate has to reject them on audio alone.

    Modelled as neighbours of a real note: the fingers sharing a chord shape sit
    one string or one fret away. Any that genuinely sound at that moment are
    dropped, because those are not spurious.
    """
    proposals = []
    for note in reference:
        for _ in range(per_note):
            pitch = int(round(note.pitch)) + int(VISUAL_ERRORS[rng.integers(len(VISUAL_ERRORS))])
            start, end = note.start - lead, max(note.end, note.start + .05)
            if any(begin - .05 <= end and start <= finish
                   for begin, finish in spans.get(pitch, ())):
                continue
            proposals.append(Candidate(pitch=pitch, start=start, end=end, truthful=False))
    return proposals


def candidates(reference, rng, visual_error, lead):
    """One oracle candidate per fretted note, as a tracker would propose them.

    A camera sees a finger arrive and sit on a cell; it does not see a pluck.
    So a candidate carries a *window*, not an onset, and the audio is what
    locates the attack inside it. ``lead`` is how long before the note the
    finger is assumed to be in place.

    Open strings are skipped. No fingertip sits on them, so a fingertip-driven
    stage cannot propose one -- which is a real part of the ceiling, not an
    omission.
    """
    proposals = []
    for note in reference:
        if note.fret <= 0:
            continue
        pitch, truthful = int(round(note.pitch)), True
        if visual_error and rng.random() < visual_error:
            pitch += int(VISUAL_ERRORS[rng.integers(len(VISUAL_ERRORS))])
            truthful = False
        proposals.append(Candidate(pitch=pitch, start=note.start - lead,
                                   end=max(note.end, note.start + .05), truthful=truthful))
    return proposals


def rescue(activations, proposals, estimated, gate, tolerance=tune_pitch.ONSET_TOLERANCE):
    """Notes the gate accepts that the global thresholds did not already report.

    A candidate whose pitch is already being reported over the same window is
    dropped rather than rescued: the point is to recover what the detector
    missed, not to report it twice.
    """
    detected = {}
    for start, _, midi in estimated:
        detected.setdefault(midi, []).append(start)
    recovered = []
    for proposal in proposals:
        if any(proposal.start - tolerance <= start <= proposal.end
               for start in detected.get(proposal.pitch, ())):
            continue
        evidence = activations.evidence(proposal.pitch, proposal.start, proposal.end)
        if evidence is None or not gate.passes(evidence):
            continue
        recovered.append((evidence.time, evidence.time + RESCUED_DURATION,
                          proposal.pitch, proposal.truthful))
    return recovered


def load(excerpt, cache_dir, setting, dtype):
    """Posteriorgram and baseline estimate for one excerpt."""
    cache = cache_dir/f'{excerpt.name}.npz' if cache_dir else None
    model = None
    if cache is None or not cache.is_file():
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        model = Model(ICASSP_2022_MODEL_PATH)
    output = tune_pitch.posteriorgram(excerpt.audio, model, cache, dtype)
    from basic_pitch.note_creation import model_frames_to_time
    activations = Posteriorgram(
        note=np.asarray(output['note'], dtype=np.float32),
        onset=np.asarray(output['onset'], dtype=np.float32),
        times=np.asarray(model_frames_to_time(output['note'].shape[0]), dtype=np.float64))
    return activations, tune_pitch.estimate(output, setting)


def measure(excerpts, gates, setting, cache_dir, visual_error, lead, seed,
            per_note=0, dtype=np.float16, progress=False):
    """Score the baseline and every gate over the same cached model output."""
    rng = np.random.default_rng(seed)
    totals = {gate.label(): tune_pitch.Tally() for gate in gates}
    baseline = tune_pitch.Tally()
    reach = {'fretted': 0, 'open': 0, 'missed_fretted': 0, 'missed_open': 0}
    rescued_counts = {gate.label(): [0, 0] for gate in gates}  # [truthful, misread]
    started = time.time()
    for index, excerpt in enumerate(excerpts, 1):
        activations, estimated = load(excerpt, cache_dir, setting, dtype)
        reference = excerpt.notes()
        matched = tune_pitch.counts(reference, estimated)
        baseline.add(*matched)
        missed = _missed(reference, estimated)
        for note in reference:
            reach['open' if note.fret <= 0 else 'fretted'] += 1
        for note in missed:
            reach['missed_open' if note.fret <= 0 else 'missed_fretted'] += 1
        proposals = candidates(reference, rng, visual_error, lead)
        if per_note:
            proposals += spurious(reference, rng, per_note, lead, _sounding(reference))
        for gate in gates:
            recovered = rescue(activations, proposals, estimated, gate)
            counts = rescued_counts[gate.label()]
            counts[0] += sum(1 for *_, truthful in recovered if truthful)
            counts[1] += sum(1 for *_, truthful in recovered if not truthful)
            augmented = estimated + [(s, e, p) for s, e, p, _ in recovered]
            totals[gate.label()].add(*tune_pitch.counts(reference, augmented))
        if progress:
            print(f'  [{index}/{len(excerpts)}] {excerpt.name} '
                  f'({time.time() - started:.0f}s)', flush=True)
    return baseline, totals, reach, rescued_counts


def _missed(reference, estimated):
    """Reference notes with no matching estimate, in reference order."""
    import mir_eval
    if not reference or not estimated:
        return list(reference)
    reference_intervals, reference_pitches = tune_pitch._arrays(
        [(note.start, note.end, note.pitch) for note in reference])
    estimated_intervals, estimated_pitches = tune_pitch._arrays(estimated)
    matches = mir_eval.transcription.match_notes(
        reference_intervals, reference_pitches, estimated_intervals, estimated_pitches,
        onset_tolerance=tune_pitch.ONSET_TOLERANCE,
        pitch_tolerance=tune_pitch.PITCH_TOLERANCE, offset_ratio=None)  # type: ignore[arg-type]
    hit = {index for index, _ in matches}
    return [note for index, note in enumerate(reference) if index not in hit]


def report(baseline, totals, reach, rescued, excerpts, visual_error):
    print(f'\n{len(excerpts)} excerpts, {baseline.referenced} annotated notes')
    ceiling = ((baseline.matched + reach['missed_fretted'])/baseline.referenced
               if baseline.referenced else 0.)
    print(f'  missed: {reach["missed_fretted"]} fretted (rescuable), '
          f'{reach["missed_open"]} open (not: no fingertip sits on an open string)')
    print(f'  recall ceiling for any fingertip-driven rescue: {ceiling:.3f}')
    if visual_error:
        print(f'  visual error rate: {visual_error:.0%} of candidates misread '
              'by one string or one fret')
    print(f'\n{"gate":28} {"F1":>6} {"F0.5":>6} {"prec":>6} {"recall":>6} '
          f'{"rescued":>8} {"wrong":>6}')
    print(f'{"baseline (no rescue)":28} {baseline.f1:6.3f} {baseline.f_half:6.3f} '
          f'{baseline.precision:6.3f} {baseline.recall:6.3f} {"-":>8} {"-":>6}')
    for label, tally in totals.items():
        good, bad = rescued[label]
        print(f'{label:28} {tally.f1:6.3f} {tally.f_half:6.3f} {tally.precision:6.3f} '
              f'{tally.recall:6.3f} {good:8d} {bad:6d}')
    print('\n"rescued" counts candidates the gate accepted and the detector had not '
          'reported;\n"wrong" is how many of those the visual error rate had already '
          'corrupted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path('data/guitarset'),
                        help='directory holding annotation/ and audio_mono-mic/')
    parser.add_argument('--cache', type=Path, default=Path('data/guitarset/posteriorgrams'),
                        help='posteriorgram cache shared with tune_pitch')
    parser.add_argument('--limit', type=int, default=48,
                        help='excerpts, spread over player and style; 0 uses all 360')
    parser.add_argument('--gate', action='append', default=None, metavar='ONSET,FRAME',
                        help='evidence a candidate must find; repeatable')
    parser.add_argument('--visual-error', type=float, default=0.,
                        help='fraction of candidates misread by one string or fret')
    parser.add_argument('--lead', type=float, default=.2,
                        help='seconds the finger is assumed down before the note')
    parser.add_argument('--spurious', type=int, default=0, metavar='K',
                        help='extra non-sounding candidates per note, for fingers '
                             'that are down but not plucked')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--output', type=Path, default=None, help='write results as JSON')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    gates = [Gate(*(float(value) for value in text.split(',')))
             for text in (args.gate or ['0.25,0.15', '0.30,0.15', '0.30,0.20',
                                        '0.35,0.25', '0.45,0.35'])]
    found, missing = guitarset.excerpts(args.root)
    if missing:
        print(f'{len(missing)} annotations have no audio and were skipped')
    chosen = guitarset.stratified(found, args.limit or None)
    setting = tune_pitch._shipped()
    print(f'Baseline setting: {setting.label()}')
    baseline, totals, reach, rescued = measure(
        chosen, gates, setting, args.cache, args.visual_error, args.lead, args.seed,
        args.spurious, np.float16 if args.cache_dtype == 'float16' else np.float32,
        args.progress)
    report(baseline, totals, reach, rescued, chosen, args.visual_error)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'setting': setting.label(), 'excerpts': [e.name for e in chosen],
            'visual_error': args.visual_error, 'spurious': args.spurious,
            'lead': args.lead, 'seed': args.seed,
            'reach': reach, 'baseline': baseline.summary(),
            'gates': {label: dict(tally.summary(), rescued=rescued[label][0],
                                  rescued_wrong=rescued[label][1])
                      for label, tally in totals.items()},
        }, indent=2))
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
