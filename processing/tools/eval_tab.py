"""Measure string choice against GuitarSet, with perfect pitch and no video.

The tab stage combines three signals; this isolates the one that has never been
measured. Feeding GuitarSet's annotated notes straight into ``transcribe_notes``
with no frames removes the detector and the hand tracker from the picture, so
what is left is the spelling logic: which of a pitch's several (string, fret)
positions it picks, and how it arbitrates a string two notes both want.

This isolates fallback assignment with perfect pitch. It is not a lower bound
on pipeline accuracy: real audio errors and incorrect hand/board evidence can
make the combined result worse.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from processing.music.transcribe import transcribe_notes
from processing.music.tuning import STANDARD_TUNING
from processing.tools import guitarset


@dataclass
class Sounded:
    """The minimum ``transcribe_notes`` needs: a pitch with a lifetime."""
    pitch: int
    start: float
    end: float
    amplitude: float = 1.


def inlay_string(gs_string, strings=len(guitarset.OPEN_STRINGS)):
    """GuitarSet counts strings from the low E; inlay counts from the high E."""
    return strings - gs_string


def measure(excerpts, tuning=STANDARD_TUNING, max_fret=22):
    counts = Counter()
    flags = Counter()
    for excerpt in excerpts:
        truth = sorted(excerpt.notes(), key=lambda note: note.start)
        notes = [Sounded(round(note.pitch), note.start, note.end) for note in truth]
        assigned, _ = transcribe_notes([], notes, tuning, max_fret, .12, False)
        for reference, result in zip(truth, assigned.values()):
            flags.update(result.flags)
            # Open notes are under a tenth of the data, so an overall figure
            # hides them entirely. They are also the hardest case, because
            # nothing here can tell an open string from the same pitch fretted
            # somewhere else. Always report them separately.
            kind = 'open' if reference.fret == 0 else 'fretted'
            if result.string is None:
                counts['unassigned'] += 1
            elif result.string == inlay_string(reference.string, len(tuning)):
                counts['correct'] += 1
                counts[f'{kind}_correct'] += 1
            else:
                counts['wrong'] += 1
            counts[f'{kind}_total'] += 1
            counts['called_open'] += result.fret == 0
            counts['total'] += 1
    return counts, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path('data/guitarset'))
    parser.add_argument('--limit', type=int, default=0, help='0 uses all 360 excerpts')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    found, _ = guitarset.excerpts(args.root)
    chosen = guitarset.stratified(found, args.limit or None)
    counts, flags = measure(chosen)
    total = counts['total']
    print(f'{total} notes over {len(chosen)} excerpts, perfect pitch, no hand, no board\n')
    for name in ('correct', 'wrong', 'unassigned'):
        print(f'  string {name:<11} {counts[name]:>7}  {counts[name]/total:.3f}')
    print()
    for kind in ('open', 'fretted'):
        size = counts[f'{kind}_total']
        print(f'  truly {kind:<9} {size:>7}  {counts[f"{kind}_correct"]/size:.3f} correct')
    print(f'  called open       {counts["called_open"]:>7}  '
          f'{counts["called_open"]/max(1, counts["open_total"]):.2f}x the true rate')
    print('\n  flags: ' + ', '.join(f'{name} {count}' for name, count in flags.most_common()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'excerpts': len(chosen), 'counts': dict(counts),
                                           'flags': dict(flags)}, indent=2))
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
