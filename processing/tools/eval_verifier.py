"""Measure the verifier against Basic Pitch's thresholds, on GuitarSet.

    uv run python -m processing.tools.eval_verifier --split test

Three numbers over the same matched notes:

  pitch F1  -- a note at the right time and the right pitch
  string    -- of those, how many landed on the right string
  tab F1    -- both at once, which is what a tab reader actually sees

The decode comes from ``processing.music.verify``, the same code the CLI runs,
so these numbers describe what inlay does rather than what a harness does.
There is no video here: the tab stage has musical context and the audio string
posterior to work from, and nothing else.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from processing.music import verify
from processing.music.transcribe import transcribe_notes
from processing.music.tuning import STANDARD_TUNING
from processing.tools import guitarset
from processing.tools.eval_tab import inlay_string
from processing.tools.tune_pitch import CURRENT, estimate, _arrays
from processing.training import verifier_data

ROOT = Path('data/guitarset')
CACHE = ROOT/'posteriorgrams'
TOLERANCE = .05


class Sounded:
    """A decoded note, carrying what audio thought of each string for it."""

    def __init__(self, pitch, start, end, posterior=None):
        self.pitch, self.start, self.end, self.amplitude = pitch, start, end, 1.
        self.string_posterior = posterior


# Columns of the count vector score_notes returns and the report adds up.
COLUMNS = ('matched', 'estimated', 'referenced', 'string_fused', 'string_direct',
           'open_matched', 'open_fused', 'fretted_matched', 'fretted_fused')


def score_notes(notes, reference, audio_weight=None):
    """Counts per COLUMNS for one excerpt.

    Open notes are under a tenth of GuitarSet and are the hardest case -- an
    open string sounds the same pitch as a fretted one elsewhere -- so they are
    always counted separately rather than hidden in an overall figure.
    """
    import mir_eval
    if not reference or not notes:
        counts = np.zeros(len(COLUMNS), np.int64)
        counts[1], counts[2] = len(notes), len(reference)
        return counts
    sounded = [Sounded(pitch, start, end, posterior)
               for start, end, pitch, posterior in notes]
    assigned, _ = transcribe_notes([], sounded, STANDARD_TUNING, 22, .12, False,
                                   audio_weight=audio_weight)
    # transcribe_notes groups by onset and returns its own order, so match the
    # results back to the notes they came from rather than trusting positions.
    placed = {(result.pitch, result.start): result for result in assigned.values()}
    tab = [placed.get((note.pitch, note.start)) for note in sounded]
    reference_intervals, reference_pitches = _arrays(
        [(note.start, note.end, note.pitch) for note in reference])
    intervals, pitches = _arrays([(s, e, p) for s, e, p, _ in notes])
    pairs = mir_eval.transcription.match_notes(
        reference_intervals, reference_pitches, intervals, pitches,
        onset_tolerance=TOLERANCE, pitch_tolerance=50., offset_ratio=None)  # type: ignore[arg-type]  # mir_eval documents None to ignore offsets.
    fused = direct = 0
    open_matched = open_fused = fretted_matched = fretted_fused = 0
    for index, estimated in pairs:
        want = inlay_string(reference[index].string, len(STANDARD_TUNING))
        placement = tab[estimated]
        hit = placement is not None and placement.string == want
        fused += hit
        if reference[index].fret == 0:
            open_matched += 1
            open_fused += hit
        else:
            fretted_matched += 1
            fretted_fused += hit
        posterior = notes[estimated][3]
        if posterior is not None and int(np.argmax(posterior)) + 1 == want:
            direct += 1
    return np.array([len(pairs), len(notes), len(reference), fused, direct,
                     open_matched, open_fused, fretted_matched, fretted_fused],
                    np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path('runs/verifier/v1/best.pt'))
    parser.add_argument('--candidates', type=Path,
                        default=Path('data/verifier/candidates.npz'))
    parser.add_argument('--split', choices=['val', 'test', 'train'], default='val')
    parser.add_argument('--thresholds', default='0.80,0.85,0.90,0.93')
    parser.add_argument('--audio-weight', type=float,
                        help='Override fusion.AUDIO_STRING_WEIGHT for this run')
    parser.add_argument('--cqt', type=Path, help='CQT cache the model was trained with')
    parser.add_argument('--no-refine', action='store_true',
                        help='Keep candidate frame times instead of the predicted onset')
    parser.add_argument('--device')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))
    candidates = verifier_data.load_candidates(args.candidates)
    train, val, test = verifier_data.splits(candidates['players'])
    members = {'train': train, 'val': val, 'test': test}[args.split]
    model, channels, trained_with = verify.load(args.model, device)
    cqt = args.cqt or (Path(trained_with) if trained_with else None)
    if channels > verify.CHANNELS and cqt is None:
        raise SystemExit(f'{args.model} was trained with a CQT; pass --cqt')
    print(f'{args.model} on the {args.split} split ({len(members)} excerpts), '
          f'{channels} channels')
    patches = verifier_data.Patches(candidates['names'], cqt=cqt)

    rows_all = np.flatnonzero(np.isin(candidates['excerpt'], list(members)))
    scores = np.empty(len(rows_all), np.float32)
    deltas = np.empty(len(rows_all), np.float32)
    strings = np.empty((len(rows_all), 6), np.float32)
    with torch.no_grad():
        for start in range(0, len(rows_all), 1024):
            chunk = rows_all[start:start + 1024]
            patch = torch.from_numpy(patches.batch(
                candidates['excerpt'][chunk].astype(np.int64),
                candidates['frame'][chunk].astype(np.int64),
                candidates['pitch'][chunk].astype(np.int64))).to(device)
            pitch = torch.from_numpy(
                candidates['pitch'][chunk].astype(np.float32)).to(device)
            exist, delta, string = model(patch, pitch)
            scores[start:start + len(chunk)] = torch.sigmoid(exist).float().cpu().numpy()
            deltas[start:start + len(chunk)] = delta.float().cpu().numpy()
            strings[start:start + len(chunk)] = torch.softmax(
                string, 1).float().cpu().numpy()[:, ::-1]

    from basic_pitch.note_creation import model_frames_to_time
    found = {excerpt.name: excerpt for excerpt in guitarset.excerpts(ROOT)[0]}
    thresholds = [float(value) for value in args.thresholds.split(',')]
    totals = {threshold: np.zeros(len(COLUMNS), np.int64)
              for threshold in thresholds}
    base = np.zeros(len(COLUMNS), np.int64)

    for index in sorted(members):
        name = str(candidates['names'][index])
        excerpt = found[name]
        with np.load(CACHE/f'{name}.npz') as stored:
            note = stored['note'].astype(np.float32)
            onset = stored['onset'].astype(np.float32)
            times = model_frames_to_time(stored['contour'].shape[0])
        reference = excerpt.notes()
        mine = candidates['excerpt'][rows_all] == index
        rows = candidates['frame'][rows_all][mine].astype(np.int64)
        columns = candidates['pitch'][rows_all][mine].astype(np.int64)
        for threshold in thresholds:
            decoded = verify.decode(note, onset, times, scores[mine], deltas[mine],
                                    strings[mine], rows, columns, threshold,
                                    refine=not args.no_refine)
            totals[threshold] += score_notes(
                [(s, e, p, q) for s, e, p, _, q in decoded], reference, args.audio_weight)
        shipped = estimate({'note': note, 'onset': onset,
                            'contour': np.zeros((note.shape[0], 1), np.float32)}, CURRENT)
        base += score_notes(sorted(((s, e, p, None) for s, e, p in shipped),
                                   key=lambda row: (row[0], row[2])), reference)

    def line(name, row):
        (matched, estimated, referenced, fused, direct,
         open_matched, open_fused, fretted_matched, fretted_fused) = row
        precision = matched/estimated if estimated else 0.
        recall = matched/referenced if referenced else 0.
        f1 = 2*precision*recall/(precision + recall) if precision + recall else 0.
        tab = (2*(fused/estimated)*(fused/referenced)
               / ((fused/estimated) + (fused/referenced))) if fused else 0.
        print(f'{name:<20} {f1:6.3f} {precision:6.3f} {recall:6.3f} '
              f'{fused/matched if matched else 0:8.3f} {tab:7.3f} '
              f'{direct/matched if matched else 0:8.3f} '
              f'{fretted_fused/fretted_matched if fretted_matched else 0:8.3f} '
              f'{open_fused/open_matched if open_matched else 0:6.3f}')

    print(f'\n{"":<20} {"F1":>6} {"P":>6} {"R":>6} {"string":>8} {"tab F1":>7} '
          f'{"audio":>8} {"fretted":>8} {"open":>6}')
    line('basic-pitch', base)
    for threshold in thresholds:
        line(f'verifier >= {threshold:.2f}', totals[threshold])
    print(f'\n{base[2]} reference notes. "string" is the share of pitch-matched notes')
    print('placed on the right string; "audio" is the string head on its own.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'split': args.split, 'model': str(args.model),
            'audio_weight': args.audio_weight,
            'baseline': dict(zip(COLUMNS, base.tolist())),
            'verifier': {str(t): dict(zip(COLUMNS, totals[t].tolist()))
                         for t in thresholds}}, indent=2))
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
