"""Training data for the note verifier: one row per candidate attack.

The verifier is trained on GuitarSet, which annotates every note with its
onset, pitch and played string. Candidates come from the same permissive
onset peak-picking the runtime uses, so what the model learns to rank is
exactly what it will be asked to rank.

Splits hold whole players out. A player's guitar, touch and recording session
repeat across all 60 of their excerpts, so splitting any other way reports a
model's memory rather than its accuracy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from processing.music import verify
from processing.tools import guitarset

ROOT = Path('data/guitarset')
CACHE = ROOT/'posteriorgrams'
TOLERANCE = .05
TRAIN_PLAYERS = ('00', '01', '02', '03')
VAL_PLAYERS = ('04',)
TEST_PLAYERS = ('05',)


def splits(players, train=TRAIN_PLAYERS, val=VAL_PLAYERS, test=TEST_PLAYERS):
    """Excerpt indices per split, by player."""
    return ({i for i, p in enumerate(players) if p in train},
            {i for i, p in enumerate(players) if p in val},
            {i for i, p in enumerate(players) if p in test})


def load_candidates(path=Path('data/verifier/candidates.npz')):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


class Patches:
    """Every excerpt's posteriorgram, padded so a patch is a plain slice.

    All 360 excerpts fit in memory at float16, which makes patch extraction
    array indexing rather than file reads: 0.6 ms for a batch of 256, against
    roughly 100 ms for the training step itself.
    """

    def __init__(self, names, cache=CACHE, progress=False, cqt=None):
        from basic_pitch.note_creation import model_frames_to_time
        self.prepared = []
        self.channels = verify.CHANNELS + (verify.CQT_CHANNELS if cqt else 0)
        for index, name in enumerate(names):
            with np.load(Path(cache)/f'{name}.npz') as stored:
                note, onset = stored['note'], stored['onset']
                contour = stored['contour']
            spectrum = None
            if cqt:
                with np.load(Path(cqt)/f'{name}.npz') as stored:
                    spectrum = verify.align_cqt(
                        stored['cqt'].astype(np.float32),
                        model_frames_to_time(contour.shape[0])[:note.shape[0]],
                        note.shape[1])
            self.prepared.append(verify.prepare(note, onset, contour, spectrum))
            if progress and index % 60 == 59:
                print(f'  loaded {index + 1}/{len(names)} posteriorgrams', flush=True)

    def batch(self, excerpt, frame, pitch):
        """(n, channels, FRAME_SPAN, PITCH_SPAN) patches for n candidates."""
        out = np.empty((len(excerpt), self.channels, verify.FRAME_SPAN,
                        verify.PITCH_SPAN), np.float32)
        for position, (index, row, column) in enumerate(zip(excerpt, frame, pitch)):
            verify.fill(out[position], self.prepared[index], int(row), int(column))
        return out


def _one(job):
    index, name, annotation = job
    from basic_pitch.note_creation import model_frames_to_time
    with np.load(CACHE/f'{name}.npz') as stored:
        onset = stored['onset'].astype(np.float32)
        note = stored['note'].astype(np.float32)
        contour_frames = stored['contour'].shape[0]
    times = model_frames_to_time(contour_frames)
    frames = min(onset.shape[0], len(times))
    found = verify.peaks(onset)
    found[frames:] = False
    rows, columns = np.nonzero(found)
    reference = guitarset.load_notes(annotation)

    # Reference notes grouped by pitch column, so labelling is a local search.
    by_pitch = {}
    for note_event in reference:
        by_pitch.setdefault(int(round(note_event.pitch)) - verify.LOWEST_BIN,
                            []).append(note_event)

    label = np.zeros(len(rows), np.int8)
    string = np.full(len(rows), -1, np.int8)
    delta = np.zeros(len(rows), np.float32)
    duration = np.zeros(len(rows), np.float32)
    for position, (row, column) in enumerate(zip(rows, columns)):
        best, gap = None, TOLERANCE + 1
        for candidate in by_pitch.get(int(column), ()):
            distance = abs(times[row] - candidate.start)
            if distance <= TOLERANCE and distance < gap:
                best, gap = candidate, distance
        if best is not None:
            label[position] = 1
            string[position] = best.string
            delta[position] = best.start - times[row]
            duration[position] = best.end - best.start
    return dict(excerpt=np.full(len(rows), index, np.int16),
                frame=rows.astype(np.int32), pitch=columns.astype(np.int8),
                onset_value=onset[rows, columns].astype(np.float32),
                note_value=note[rows, columns].astype(np.float32),
                label=label, string=string, delta=delta, duration=duration,
                reference=np.int32(len(reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('data/verifier/candidates.npz'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    found, _ = guitarset.excerpts(args.root)
    found = sorted(found, key=lambda excerpt: excerpt.name)
    jobs = [(index, excerpt.name, excerpt.annotation)
            for index, excerpt in enumerate(found)]
    parts, references = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, part in enumerate(pool.map(_one, jobs, chunksize=2), 1):
            references += int(part.pop('reference'))
            parts.append(part)
            if number % 60 == 0:
                print(f'{number}/{len(jobs)}', flush=True)
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    merged['names'] = np.array([excerpt.name for excerpt in found])
    merged['players'] = np.array([excerpt.player for excerpt in found])
    merged['styles'] = np.array([excerpt.style for excerpt in found])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)  # type: ignore[arg-type]  # ** collides with savez's own allow_pickle keyword.
    positives = int(merged['label'].sum())
    print(f'\n{len(merged["label"])} candidates, {positives} positive, '
          f'{references} reference notes')
    print(f'  negatives per positive: '
          f'{(len(merged["label"]) - positives)/max(1, positives):.1f}')
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
