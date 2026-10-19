"""Does the verifier still help on guitars it was never trained on?

    uv run python -m processing.tools.idmt --prepare
    uv run python -m processing.tools.eval_idmt

Everything else is measured on GuitarSet: six players, acoustic guitars, one
microphone. A model trained on that could have learned the room rather than
the instrument. IDMT-SMT-GUITAR is a different corpus -- a Fender
Stratocaster, a Gibson Les Paul, an Aristides and an Ibanez, across four
pickup settings, plus acoustic mic and pickup -- annotated per event with
onset, pitch, string and fret.

Nothing here is trained on. IDMT-SMT-GUITAR is CC BY-NC-ND 4.0 and is used
only to measure.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from processing.music import verify
from processing.tools import idmt
from processing.tools.eval_verifier import COLUMNS, score_notes
from processing.tools.tune_pitch import CURRENT, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=idmt.ROOT)
    parser.add_argument('--cache', type=Path, default=idmt.POSTERIORGRAMS)
    parser.add_argument('--model', type=Path,
                        default=Path('runs/verifier/v3-aug/best.pt'))
    parser.add_argument('--threshold', type=float, default=verify.SCORE_THRESHOLD)
    parser.add_argument('--audio-weight', type=float,
                        help='Override fusion.AUDIO_STRING_WEIGHT for this run')
    parser.add_argument('--device')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))
    model, channels, _ = verify.load(args.model, device)
    found, _ = idmt.recordings(args.root)
    print(f'{args.model} ({channels} channels) on {len(found)} IDMT recordings, '
          f'threshold {args.threshold}\n')

    from basic_pitch.note_creation import model_frames_to_time
    groups = {}
    mine_total = np.zeros(len(COLUMNS), np.int64)
    base_total = np.zeros(len(COLUMNS), np.int64)
    for number, recording in enumerate(found, 1):
        path = args.cache/f'{idmt.key(recording)}.npz'
        if not path.is_file():
            continue
        with np.load(path) as stored:
            note = stored['note'].astype(np.float32)
            onset = stored['onset'].astype(np.float32)
            contour = stored['contour'].astype(np.float32)
        times = model_frames_to_time(contour.shape[0])
        reference = recording.notes()
        if not reference:
            continue
        frames = min(note.shape[0], len(times))
        found_peaks = verify.peaks(onset)
        found_peaks[frames:] = False
        rows, columns = np.nonzero(found_peaks)
        notes = []
        if len(rows):
            spectrum = None
            if channels > verify.CHANNELS:
                spectrum = verify.align_cqt(verify.spectrum(recording.audio),
                                            times[:note.shape[0]], note.shape[1])
            prepared = verify.prepare(note, onset, contour, spectrum)
            rows, columns = rows.astype(np.int64), columns.astype(np.int64)
            exists, deltas, strings = verify.score(model, prepared, rows, columns,
                                                   channels, device)
            notes = [(s, e, p, q) for s, e, p, _, q in verify.decode(
                note, onset, times, exists, deltas, strings, rows, columns,
                args.threshold)]
        shipped = sorted(((s, e, p, None) for s, e, p in estimate(
            {'note': note, 'onset': onset, 'contour': contour}, CURRENT)),
            key=lambda row: (row[0], row[2]))
        mine = score_notes(notes, reference, args.audio_weight)
        base = score_notes(shipped, reference)
        mine_total += mine
        base_total += base
        bucket = groups.setdefault(recording.group,
                                   [np.zeros(len(COLUMNS), np.int64),
                                    np.zeros(len(COLUMNS), np.int64)])
        bucket[0] += base
        bucket[1] += mine
        if number % 150 == 0:
            print(f'  {number}/{len(found)}', flush=True)

    def line(name, row):
        matched, estimated, referenced, fused = row[:4]
        precision = matched/estimated if estimated else 0.
        recall = matched/referenced if referenced else 0.
        f1 = 2*precision*recall/(precision + recall) if precision + recall else 0.
        tab = (2*(fused/estimated)*(fused/referenced)
               / ((fused/estimated) + (fused/referenced))) if fused else 0.
        print(f'{name:<40} {f1:6.3f} {precision:6.3f} {recall:6.3f} '
              f'{fused/matched if matched else 0:8.3f} {tab:7.3f}')

    print(f'\n{"":<40} {"F1":>6} {"P":>6} {"R":>6} {"string":>8} {"tab F1":>7}')
    line('basic-pitch', base_total)
    line('verifier', mine_total)
    print(f'\n{base_total[2]} annotated notes over {len(found)} recordings\n')
    print('by subset:')
    for name in sorted(groups):
        base, mine = groups[name]
        line(f'  {name[:34]} basic-pitch', base)
        line(f'  {name[:34]} verifier', mine)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        keys = COLUMNS
        args.output.write_text(json.dumps({
            'model': str(args.model), 'threshold': args.threshold,
            'audio_weight': args.audio_weight,
            'baseline': dict(zip(keys, base_total.tolist())),
            'verifier': dict(zip(keys, mine_total.tolist())),
            'by_subset': {name: {'baseline': dict(zip(keys, base.tolist())),
                                 'verifier': dict(zip(keys, mine.tolist()))}
                          for name, (base, mine) in groups.items()}}, indent=2))
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
