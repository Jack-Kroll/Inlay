"""IDMT-SMT-GUITAR: electric guitars, for checking generalisation off GuitarSet.

GuitarSet is six players on acoustic guitars through one microphone. Anything
tuned on it could be tuned to that one recording chain. IDMT-SMT-GUITAR is a
different group of instruments -- Fender and Ibanez solid bodies, an Aristides,
a Career SG, plus acoustic mic and pickup -- annotated per event with onset,
pitch, string and fret, so the same measurements can be repeated on it.

The dataset is CC BY-NC-ND 4.0: no derivatives and non-commercial. It is used
here to *measure*, never to train, and nothing derived from it ships.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree

STANDARD = (40, 45, 50, 55, 59, 64)  # Low E first, as IDMT writes tunings.


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: float
    string: int   # 0 is the low E, matching processing.tools.guitarset.
    fret: int


@dataclass(frozen=True)
class Recording:
    name: str
    audio: Path
    annotation: Path
    instrument: str
    group: str
    tuning: tuple

    def notes(self):
        return load_notes(self.annotation)


def _text(node, tag, default=None):
    found = node.find(tag)
    return found.text.strip() if found is not None and found.text else default


def load_notes(annotation):
    """Read one XML transcription. Strings are renumbered to start at the low E."""
    root = ElementTree.parse(annotation).getroot()
    notes = []
    for event in root.iter('event'):
        fields = [_text(event, tag) for tag in
                  ('onsetSec', 'offsetSec', 'pitch', 'stringNumber')]
        if any(value is None for value in fields):
            continue
        start, end, pitch, string = (float(value) for value in fields
                                     if value is not None)
        if end <= start:
            continue
        # IDMT numbers strings 1..6 from the low E; subtract one to match
        # GuitarSet's 0..5 from the low E.
        index = int(string) - 1
        if not 0 <= index < 6:
            continue
        fret = _text(event, 'fretNumber')
        notes.append(Note(start, end, pitch, index,
                          int(float(fret)) if fret is not None else -1))
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def _tuning(root):
    raw = _text(root.find('globalParameter'), 'instrumentTuning') if \
        root.find('globalParameter') is not None else None
    if not raw:
        return STANDARD
    # Tunings appear both space-separated and comma-separated in the corpus.
    values = tuple(int(float(v)) for v in re.split(r'[\s,]+', raw.strip()) if v)
    return values if len(values) == 6 else STANDARD


def recordings(root, standard_only=True, min_notes=1):
    """Every annotated recording, paired with its audio.

    ``standard_only`` keeps recordings in standard tuning, because inlay is not
    told the tuning and a dropped or raised string would be measuring the wrong
    thing.
    """
    root = Path(root)
    found, skipped = [], []
    for annotation in sorted(root.rglob('*.xml')):
        try:
            tree = ElementTree.parse(annotation).getroot()
        except ElementTree.ParseError:
            skipped.append((annotation.name, 'unparseable'))
            continue
        globals_ = tree.find('globalParameter')
        if globals_ is None:
            skipped.append((annotation.name, 'no globalParameter'))
            continue
        named = _text(globals_, 'audioFileName') or ''
        stem = Path(named.replace('\\', '/')).name or f'{annotation.stem}.wav'
        audio = None
        for candidate in (annotation.parent/stem,
                          annotation.parent.parent/'audio'/stem,
                          annotation.parent.parent/'audio'/f'{annotation.stem}.wav'):
            if candidate.is_file():
                audio = candidate
                break
        if audio is None:
            matches = list(root.rglob(stem))
            audio = matches[0] if len(matches) == 1 else None
        if audio is None:
            skipped.append((annotation.name, f'no audio for {stem}'))
            continue
        tuning = _tuning(tree)
        if standard_only and tuning != STANDARD:
            skipped.append((annotation.name, f'tuning {tuning}'))
            continue
        notes = load_notes(annotation)
        if len(notes) < min_notes:
            skipped.append((annotation.name, f'{len(notes)} notes'))
            continue
        found.append(Recording(
            name=annotation.stem, audio=audio, annotation=annotation,
            instrument=_text(globals_, 'instrumentModel', 'unknown') or 'unknown',
            group=annotation.parts[-3] if len(annotation.parts) >= 3 else 'unknown',
            tuning=tuning))
    return found, skipped


ROOT = Path('data/idmt-smt-guitar/extracted')
POSTERIORGRAMS = Path('data/idmt-smt-guitar/posteriorgrams')


def key(recording):
    """A filename-safe id. IDMT reuses stems across subsets and pickup folders,
    so the path has to be part of the identity."""
    import hashlib
    digest = hashlib.sha1(str(recording.audio).encode()).hexdigest()[:10]
    return f'{recording.name}.{digest}'


def _cache_one(job):
    name, audio, destination = job
    if destination.is_file():
        return name, 'cached'
    import contextlib, io
    from basic_pitch.inference import run_inference
    from basic_pitch import ICASSP_2022_MODEL_PATH
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            output = run_inference(str(audio), ICASSP_2022_MODEL_PATH)
    except Exception as error:            # noqa: BLE001 - report and keep going
        return name, f'failed: {type(error).__name__}: {error}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(destination, note=output['note'].astype(np.float16),
                        onset=output['onset'].astype(np.float16),
                        contour=output['contour'].astype(np.float16))
    return name, f'{output["note"].shape[0]} frames'


def main():
    import argparse
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepare', action='store_true',
                        help='Cache Basic Pitch output for every recording')
    parser.add_argument('--output', type=Path, default=POSTERIORGRAMS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    found, skipped = recordings(args.root)
    print(f'{len(found)} usable recordings, {len(skipped)} skipped')
    print('\nby instrument:')
    for name, count in Counter(r.instrument for r in found).most_common():
        print(f'  {name:<34} {count:>4}')
    print('\nby subset:')
    for name, count in Counter(r.group for r in found).most_common():
        print(f'  {name:<34} {count:>4}')
    print(f'\n{sum(len(r.notes()) for r in found)} annotated notes')
    if not args.prepare:
        return
    jobs = [(key(r), r.audio, args.output/f'{key(r)}.npz') for r in found]
    done, failed = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, note in pool.map(_cache_one, jobs, chunksize=1):
            done += 1
            if note.startswith('failed'):
                failed.append((name, note))
            if done % 100 == 0:
                print(f'{done}/{len(jobs)}', flush=True)
    print(f'cached {done - len(failed)}/{len(jobs)}; {len(failed)} failed')
    for name, note in failed[:10]:
        print(f'  {name}: {note}')


if __name__ == '__main__':
    main()
