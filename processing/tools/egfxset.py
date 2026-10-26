"""EGFxSet: single notes on every string and fret of one electric, for the string head.

    uv run python -m processing.tools.egfxset --prepare

EGFxSet (Zenodo record 7044411, CC BY 4.0; Pedroza, Meza and Roman, ISMIR
2022 late-breaking demo) recorded all 138 notes of a 22-fret Stratocaster in
standard tuning -- six strings, frets 0 to 22 -- through each of its five
pickup positions, clean and through twelve hardware effects, with string and
fret in every file name. Download ``Clean.zip``, ``BluesDriver.zip``,
``TubeScreamer.zip`` and ``RAT.zip`` and unzip them under
``data/egfxset/extracted``; ``processing.training.frontend_data`` uses the same
files to train the front end.

For the verifier, ``--prepare`` strings the single notes of one pickup and
pedal into thirty-second excerpts -- a new note every 2.5 seconds, each faded
out before the next, strings and frets shuffled -- and writes a GuitarSet-style
JAMS annotation for each, so the verifier's data pipeline reads them as one
more domain, ``egfx``. Labels are exact: the onset is the file's own attack
and the string is in the name. Every open string is there, on an electric, in
every pickup position; Guitar-TECHS has almost none.

Delays, reverbs and modulation are left out, as for the front end: the first
puts audible repeats where one attack is labelled, the others smear offsets.
One file, TubeScreamer/Bridge-Middle/1-23, has no clean counterpart and is
skipped.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from processing.tools import guitarset, guitar_techs
from processing.training import frontend_data

ROOT = Path('data/egfxset')
EXCERPTS = ROOT/'excerpts'
MANIFEST = ROOT/'manifest.json'
PLAYER = 'EGFX'
DOMAIN = 'egfx'
SAMPLE_RATE = 22050
SPACING = 2.5
NOTES_PER_EXCERPT = 12
FADE = .03


def build(items, rng):
    """One excerpt's audio and notes from single-note items, in order."""
    length = int(SPACING*NOTES_PER_EXCERPT*SAMPLE_RATE)
    wave = np.zeros(length, np.float32)
    notes = []
    span = int(SPACING*SAMPLE_RATE)
    fade = int(FADE*SAMPLE_RATE)
    ramp = np.linspace(1., 0., fade, dtype=np.float32)
    for slot, item in enumerate(items):
        audio = np.load(item.audio).astype(np.float32)[:span]
        audio = audio.copy()
        audio[-fade:] *= ramp[:len(audio[-fade:])]
        (begin, end, pitch), = frontend_data.load_labels(item)[0]
        offset = slot*span
        wave[offset:offset + len(audio)] += audio*rng.uniform(.5, 1.)
        string = 6 - int(item.source.stem.split('-')[0])
        notes.append(guitarset.Note(offset/SAMPLE_RATE + float(begin),
                                    offset/SAMPLE_RATE + min(float(end), SPACING - FADE),
                                    float(pitch), string))
    peak = np.abs(wave).max()
    return wave*(.9/peak if peak > .9 else 1.), notes


def prepare(root=ROOT, seed=0):
    import soundfile
    items = frontend_data.egfx_items()
    missing = [item for item in items if not item.audio.is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} EGFxSet notes not rendered; run '
                         'processing.training.frontend_data --prepare first')
    groups = {}
    for item in items:
        effect, pickup = item.source.parent.parent.name, item.source.parent.name
        groups.setdefault((effect, pickup), []).append(item)
    rng = np.random.default_rng(seed)
    rows = []
    for (effect, pickup), members in sorted(groups.items()):
        order = rng.permutation(len(members))
        for number, start in enumerate(range(0, len(order), NOTES_PER_EXCERPT)):
            chosen = [members[i] for i in order[start:start + NOTES_PER_EXCERPT]]
            wave, notes = build(chosen, rng)
            name = f'EGFX_{effect}_{pickup}_{number:03d}'
            audio = root/'excerpts'/f'{name}.wav'
            annotation = root/'excerpts'/f'{name}.jams'
            audio.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(str(audio), wave, SAMPLE_RATE, subtype='PCM_16')
            annotation.write_text(json.dumps(guitar_techs.jams(notes)))
            rows.append(dict(name=name, effect=effect, pickup=pickup, notes=len(notes),
                             audio=str(audio), annotation=str(annotation)))
    (root/MANIFEST.name).write_text(json.dumps(rows, indent=1))
    print(f'{len(rows)} excerpts, {sum(r["notes"] for r in rows)} notes; '
          f'wrote {root/MANIFEST.name}')


def excerpts(root=ROOT):
    """Prepared excerpts as ``guitarset.Excerpt`` records, empty when none are.

    The player is ``EGFX`` throughout -- one guitarist, one guitar -- and the
    effect stands where GuitarSet keeps its style."""
    manifest = Path(root)/MANIFEST.name
    if not manifest.is_file():
        return []
    rows = json.loads(manifest.read_text())
    return [guitarset.Excerpt(name=row['name'], audio=Path(row['audio']),
                              annotation=Path(row['annotation']), player=PLAYER,
                              progression=row['effect'], tempo=0, key=DOMAIN,
                              style=row['effect'])
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()


if __name__ == '__main__':
    main()
