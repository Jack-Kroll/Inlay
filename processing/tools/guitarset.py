"""GuitarSet: annotated acoustic guitar recordings used as transcription truth.

GuitarSet ships 360 thirty-second excerpts with one ``note_midi`` annotation
per string, so it supplies both the sounding pitch Basic Pitch is measured
against and the string the note was actually played on. Download
``annotation.zip`` and ``audio_mono-mic.zip`` from Zenodo record 3371780
(CC BY 4.0) and unzip both into one directory.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re

# GuitarSet numbers its ``note_midi`` annotations from the low E upwards. The
# tuning inlay uses elsewhere runs highest string first, so it reads reversed.
OPEN_STRINGS = (40, 45, 50, 55, 59, 64)
STRING_NAMES = ('E2', 'A2', 'D3', 'G3', 'B3', 'E4')
FILENAME = re.compile(r'(?P<player>\d+)_(?P<progression>[A-Za-z]+\d*)-(?P<tempo>\d+)-'
                      r'(?P<key>[^_]+)_(?P<style>comp|solo)$')


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: float  # MIDI, fractional: GuitarSet annotates the sounding pitch.
    string: int   # 0 is the low E.

    @property
    def fret(self):
        return round(self.pitch) - OPEN_STRINGS[self.string]


@dataclass(frozen=True)
class Excerpt:
    name: str
    audio: Path
    annotation: Path
    player: str
    progression: str
    tempo: int
    key: str
    style: str

    def notes(self):
        return load_notes(self.annotation)


def load_notes(annotation):
    """Read the six per-string note annotations out of one JAMS file."""
    annotation = Path(annotation)
    document = json.loads(annotation.read_text())
    notes, strings = [], set()
    for block in document.get('annotations', ()):
        if block.get('namespace') != 'note_midi':
            continue
        source = block.get('annotation_metadata', {}).get('data_source')
        string = int(source)
        if not 0 <= string < len(OPEN_STRINGS):
            raise ValueError(f'{annotation}: string index {string} out of range')
        if string in strings:
            raise ValueError(f'{annotation}: duplicate annotation for string {string}')
        strings.add(string)
        for observation in block.get('data', ()):
            start = float(observation['time'])
            duration = float(observation['duration'])
            # Zero-length observations cannot be matched against and would fail
            # mir_eval's interval validation, so they are dropped here.
            if duration <= 0:
                continue
            notes.append(Note(start, start + duration, float(observation['value']), string))
    if len(strings) != len(OPEN_STRINGS):
        raise ValueError(f'{annotation}: found {len(strings)} string annotations, expected '
                         f'{len(OPEN_STRINGS)}')
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def excerpts(root, audio_dir='audio_mono-mic', audio_suffix='_mic'):
    """Pair every JAMS file under ``root`` with its audio."""
    root = Path(root)
    annotations = sorted((root/'annotation').glob('*.jams'))
    if not annotations:
        raise FileNotFoundError(f'No JAMS files under {root/"annotation"}; unzip '
                                'annotation.zip into that directory')
    found = []
    missing = []
    for annotation in annotations:
        name = annotation.stem
        audio = root/audio_dir/f'{name}{audio_suffix}.wav'
        if not audio.is_file():
            missing.append(name)
            continue
        fields = FILENAME.match(name)
        if not fields:
            raise ValueError(f'Unrecognised GuitarSet filename: {name}')
        found.append(Excerpt(name=name, audio=audio, annotation=annotation,
                             player=fields['player'], progression=fields['progression'],
                             tempo=int(fields['tempo']), key=fields['key'],
                             style=fields['style']))
    if not found:
        raise FileNotFoundError(f'No audio matched the annotations under {root/audio_dir}')
    return found, missing


def stratified(items, limit):
    """Take ``limit`` excerpts spread evenly over player and playing style.

    A contiguous slice would weight the result towards whichever players sort
    first, and comp and solo excerpts stress the detector very differently.
    """
    if limit is None or limit >= len(items):
        return list(items)
    if limit <= 0:
        raise ValueError('limit must be positive')
    groups = {}
    for item in sorted(items, key=lambda item: item.name):
        groups.setdefault((item.style, item.player), []).append(item)
    order = [groups[key] for key in sorted(groups)]
    quota = -(-limit//len(order))
    picks = []
    for group in order:
        take = min(quota, len(group))
        # Evenly spaced within the group, so one progression cannot dominate.
        step = len(group)/take
        picks.append([group[int(index*step)] for index in range(take)])
    chosen = [group[position] for position in range(quota)
              for group in picks if position < len(group)]
    return sorted(chosen[:limit], key=lambda item: item.name)
