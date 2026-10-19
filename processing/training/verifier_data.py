"""Training data for the note verifier: one row per candidate attack.

The verifier is trained on GuitarSet, which annotates every note with its
onset, pitch and played string. Candidates come from the same permissive
onset peak-picking the runtime uses, so what the model learns to rank is
exactly what it will be asked to rank.

Splits hold whole players out. A player's guitar, touch and recording session
repeat across all 60 of their excerpts, so splitting any other way reports a
model's memory rather than its accuracy.

An excerpt can appear more than once, under a *domain*: the microphone as
recorded, GuitarSet's pickup mix of the same performance, or an augmented
render of either (``processing.tools.variants``). Most domains share their
excerpt's annotation unchanged -- the audio moved, the notes did not -- and the
domain a number is measured on is always named, because a model trained on a
synthesised amp and measured on one would only be describing the augmenter.

A neck-shifted domain is the exception. It was resampled to move up the neck,
which raises every pitch and shortens every interval by the same ratio, so its
annotation is scaled to match. The shift is read back from the posteriorgram
that was rendered with it rather than redrawn here.
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
# The microphone excerpts everything was measured on before domains existed.
# They keep the bare name, so the caches already on disk still resolve.
MIC = 'mic'


def variant_name(domain, name):
    """One excerpt's cache name within ``domain``.

    A domain is a directory inside each cache, so a single name resolves a
    posteriorgram and a CQT at once and ``Patches`` needs to know nothing
    about where the audio came from.
    """
    return str(name) if domain == MIC else f'{domain}/{name}'


def domain_of(name):
    """Which domain a cache name belongs to."""
    name = str(name)
    return name.rsplit('/', 1)[0] if '/' in name else MIC


def splits(players, names=None, train=TRAIN_PLAYERS, val=VAL_PLAYERS,
           test=TEST_PLAYERS, domain=MIC):
    """Excerpt indices per split: every domain to train on, one to measure on.

    Training takes whichever domains the candidate file holds, which is how
    augmentation gets in. Validation and test take ``domain`` alone, so a
    number always names the recording chain it was measured on and an
    augmented render of a held-out player cannot quietly become one.
    """
    here = ([domain_of(name) == domain for name in names] if names is not None
            else [True]*len(players))
    return ({i for i, p in enumerate(players) if p in train},
            {i for i, p in enumerate(players) if p in val and here[i]},
            {i for i, p in enumerate(players) if p in test and here[i]})


def load_candidates(path=Path('data/verifier/candidates.npz')):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


class Patches:
    """Every excerpt's posteriorgram, padded so a patch is a plain slice.

    All 360 excerpts fit in memory at float16, which makes patch extraction
    array indexing rather than file reads: 0.6 ms for a batch of 256, against
    roughly 100 ms for the training step itself.
    """

    def __init__(self, names, cache=CACHE, progress=False, cqt=None, only=None):
        from basic_pitch.note_creation import model_frames_to_time
        self.prepared = []
        self.channels = verify.CHANNELS + (verify.CQT_CHANNELS if cqt else 0)
        for index, name in enumerate(names):
            if only is not None and index not in only:
                # Indexed by excerpt, so a skipped one still needs its place.
                self.prepared.append(None)
                continue
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


def shifted(notes, semitones):
    """The same notes as played ``semitones`` higher up the same strings.

    Resampling an excerpt to raise it by a ratio shortens it by that ratio, so
    every annotated time divides by it and every pitch gains the interval. The
    string does not move, which is the whole point: this exists to show the
    string head the top of a neck GuitarSet barely visits.
    """
    if not semitones:
        return notes
    ratio = 2.**(semitones/12.)
    return [guitarset.Note(note.start/ratio, note.end/ratio,
                           note.pitch + semitones, note.string) for note in notes]


def _one(job):
    index, name, annotation, cache = job
    from basic_pitch.note_creation import model_frames_to_time
    with np.load(Path(cache)/f'{name}.npz') as stored:
        onset = stored['onset'].astype(np.float32)
        note = stored['note'].astype(np.float32)
        contour_frames = stored['contour'].shape[0]
        semitones = int(stored['shift']) if 'shift' in stored.files else 0
    times = model_frames_to_time(contour_frames)
    frames = min(onset.shape[0], len(times))
    found = verify.peaks(onset)
    found[frames:] = False
    rows, columns = np.nonzero(found)
    reference = shifted(guitarset.load_notes(annotation), semitones)

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
    parser.add_argument('--domains', nargs='*', default=[],
                        help='Extra rendered domains to add, e.g. pickup aug-v1')
    parser.add_argument('--measure-domains', nargs='*', default=[],
                        help='Domains to add for held-out players too, to measure on')
    args = parser.parse_args()

    cache = args.root/'posteriorgrams'
    found, _ = guitarset.excerpts(args.root)
    found = sorted(found, key=lambda excerpt: excerpt.name)
    # One row per (excerpt, variant). Candidates are peak-picked from each
    # render's own posteriorgram, because a frame that is a peak in the clean
    # audio need not be one once the audio has been through an amp, and a
    # patch the runtime would never centre on teaches nothing.
    rendered = [(excerpt, excerpt.name) for excerpt in found]
    for domain in dict.fromkeys(args.domains + args.measure_domains):
        for excerpt in found:
            if (excerpt.player not in TRAIN_PLAYERS
                    and domain not in args.measure_domains):
                continue
            name = variant_name(domain, excerpt.name)
            if (cache/f'{name}.npz').is_file():
                rendered.append((excerpt, name))
    jobs = [(index, name, excerpt.annotation, cache)
            for index, (excerpt, name) in enumerate(rendered)]
    parts, references = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, part in enumerate(pool.map(_one, jobs, chunksize=2), 1):
            references += int(part.pop('reference'))
            parts.append(part)
            if number % 60 == 0:
                print(f'{number}/{len(jobs)}', flush=True)
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    merged['names'] = np.array([name for _, name in rendered])
    merged['players'] = np.array([excerpt.player for excerpt, _ in rendered])
    merged['styles'] = np.array([excerpt.style for excerpt, _ in rendered])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)  # type: ignore[arg-type]  # ** collides with savez's own allow_pickle keyword.
    positives = int(merged['label'].sum())
    counts = {}
    for _, name in rendered:
        counts[domain_of(name)] = counts.get(domain_of(name), 0) + 1
    print(f'\n{len(rendered)} excerpts (' +
          ', '.join(f'{count} {domain}' for domain, count in counts.items()) + '), '
          f'{len(merged["label"])} candidates, {positives} positive, '
          f'{references} reference notes')
    print(f'  negatives per positive: '
          f'{(len(merged["label"]) - positives)/max(1, positives):.1f}')
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
