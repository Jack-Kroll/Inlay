"""Guitar-TECHS: electric guitars with per-string labels, for the string head.

    uv run python -m processing.tools.guitar_techs --prepare

The verifier's string head reads timbre, and it learned timbre from six
acoustic guitars through one microphone. Off that corpus it is confidently
wrong -- .638 on IDMT's electrics, .153 past fret 15 -- which is why the audio
string term ships turned off. Guitar-TECHS (Zenodo record 14963133, CC BY 4.0)
is three players on three electric guitars through three amps, five hours,
every take recorded four ways: direct input, a microphone on the amp, and two
stereo microphones in the room. Labels come from a Fishman TriplePlay pickup,
one MIDI track per string, so the string is read off hardware rather than
inferred. Its single-note set plays frets 0 to 22 on every string, which is
the part of the neck GuitarSet never visits.

``--prepare`` slices each recording into thirty-second excerpts, writes a
GuitarSet-style JAMS annotation per excerpt, and renders each recording chain
into the verifier's caches as its own domain: ``gt-di``, ``gt-amp``,
``gt-ego``, ``gt-exo``. The domains live beside GuitarSet's under
``data/guitarset`` because that is the one cache root the training data
reads; the ``gt-`` prefix says where they came from.

The MIDI is not sample-aligned to the audio. Zenodo warns of up to 100 ms, and
the sample measured here sat 12 ms early against the direct input and 46 ms
early against the room microphones -- within the 50 ms onset tolerance, but
not comfortably, and different per chain. So each (recording, chain) pair
estimates one global lag -- the median distance from a labelled note to the
nearest peak of the rendered onset head at its pitch, which is the signal the
labels are matched against -- and the annotation for that chain carries it.
That is why an excerpt's name includes its chain: the audio moved and so,
slightly, did the labels.

After that shift the direct input and amp microphone agree with their labels
as well as GuitarSet does: 99% of notes within 50 ms, 96% within 25. The two
room microphones do not -- 83% within 50 ms, 56% within 25 -- and not for
want of a better lag: the offsets scatter note to note with a 33 ms spread
inside a single excerpt, with no drift to fit. Whether that is the video
device's clock or the onset head losing precision in a room, the labels
cannot say *when* a note attacked on those chains, only *which string*
played it. So ``gt-ego`` and ``gt-exo`` are trained as string-only domains
(``train_verifier --string-only-domains``), and note F1 on the held-out
player is read off ``gt-amp``, not the room.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from processing.music import verify
from processing.tools import guitarset

ROOT = Path('data/guitar-techs')
RAW = ROOT/'raw'
MANIFEST = ROOT/'manifest.json'
# Where the rendered domains go: the one cache root verifier_data reads.
CACHE_ROOT = Path('data/guitarset')
PLAYERS = ('P1', 'P2', 'P3')
# chain -> (directory under the recording, filename prefix)
CHAINS = {'di': ('audio/directinput', 'directinput_'),
          'amp': ('audio/micamp', 'micamp_'),
          'ego': ('video/ego', 'ego_'),
          'exo': ('video/exo', 'exo_')}
# Track names in the MIDI, low E first, matching guitarset.OPEN_STRINGS.
STRING_TRACKS = ('E', 'A', 'D', 'G', 'B', 'e')
EXCERPT_SECONDS = 30.
SAMPLE_RATE = 22050
MAX_LAG = .25
HIGHEST_FRET = 24
ARCHIVE = re.compile(r'^(?P<player>P\d)_(?P<category>[a-z]+)$')


def domain(chain):
    return f'gt-{chain}'


@dataclass(frozen=True)
class Recording:
    player: str
    category: str
    stem: str
    midi: Path
    audio: dict  # chain -> Path

    @property
    def name(self):
        return f'{self.player}_{self.category}_{self.stem}'


def recordings(raw=RAW):
    """Every MIDI file under ``raw`` paired with the audio of each chain.

    The archives unzip to ``raw/P1_chords/P1_chords/{audio,midi,video}``; the
    walk tolerates the doubled directory and the ``__MACOSX`` litter.
    """
    found = []
    for midi in sorted(Path(raw).rglob('midi/*.mid')):
        if '__MACOSX' in midi.parts:
            continue
        base = midi.parent.parent
        fields = ARCHIVE.match(base.name)
        if not fields:
            continue
        stem = midi.stem[len('midi_'):] if midi.stem.startswith('midi_') else midi.stem
        audio = {}
        for chain, (directory, prefix) in CHAINS.items():
            matches = sorted((base/directory).glob(f'{prefix}{stem}.*'))
            matches = [m for m in matches if m.suffix in ('.wav', '.mp3', '.flac')]
            if matches:
                audio[chain] = matches[0]
        if audio:
            found.append(Recording(fields['player'], fields['category'], stem,
                                   midi, audio))
    return found


def load_midi(path):
    """Notes from a Guitar-TECHS MIDI file: one track per string, named."""
    import mido
    file = mido.MidiFile(str(path))
    # Tempo map from every track, in absolute ticks. The files seen so far
    # carry one set_tempo at tick 0, but the conversion costs nothing to do
    # properly.
    tempos = []
    for track in file.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == 'set_tempo':
                tempos.append((tick, message.tempo))
    tempos.sort()
    if not tempos or tempos[0][0] != 0:
        tempos.insert(0, (0, 500000))

    def seconds(tick):
        total, at, tempo = 0., 0, tempos[0][1]
        for change, next_tempo in tempos[1:]:
            if change >= tick:
                break
            total += mido.tick2second(change - at, file.ticks_per_beat, tempo)
            at, tempo = change, next_tempo
        return total + mido.tick2second(tick - at, file.ticks_per_beat, tempo)

    notes = []
    for track in file.tracks:
        if track.name not in STRING_TRACKS:
            continue
        string = STRING_TRACKS.index(track.name)
        tick, open_notes = 0, {}
        for message in track:
            tick += message.time
            if message.type == 'note_on' and message.velocity > 0:
                open_notes.setdefault(message.note, []).append(tick)
            elif message.type in ('note_off', 'note_on') and message.note in open_notes:
                started = open_notes[message.note].pop(0)
                if not open_notes[message.note]:
                    del open_notes[message.note]
                start, end = seconds(started), seconds(tick)
                # A pitch below the string's open note, or past the last
                # fret, is the pickup misreading the string: 30 notes in
                # 18,830 on the direct input. They would teach the string
                # head an impossible fingering, so they go.
                fret = message.note - guitarset.OPEN_STRINGS[string]
                if end > start and 0 <= fret <= HIGHEST_FRET:
                    notes.append(guitarset.Note(start, end, float(message.note), string))
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def peak_offsets(onset, times, notes, max_lag=MAX_LAG):
    """Per labelled note, how far the nearest onset-head peak at its own pitch
    sits from its labelled start, in seconds; notes with no peak within
    ``max_lag`` are left out.

    This is the geometry the candidate labelling uses -- a peak of the onset
    head, at the note's pitch, within tolerance of the annotated start -- so
    the median of these is the shift that makes Guitar-TECHS's labels sit
    where GuitarSet's do. (Correlating the labels against the envelope
    collapsed over pitch was tried first and read +23 to +139 ms on GuitarSet
    excerpts that this reads at -3 to -12 ms: with a chord every beat the
    broad surface finds the next beat as readily as the attack.)
    """
    found = verify.peaks(onset)
    offsets = []
    for note in notes:
        column = int(round(note.pitch)) - verify.LOWEST_BIN
        if not 0 <= column < onset.shape[1]:
            continue
        rows = np.flatnonzero(found[:len(times), column])
        if not len(rows):
            continue
        distance = times[rows] - note.start
        nearest = distance[np.argmin(np.abs(distance))]
        if abs(nearest) <= max_lag:
            offsets.append(float(nearest))
    return offsets


def lag(offsets):
    """Seconds to add to the MIDI times: the median peak offset, or 0 when
    there is nothing to go on."""
    return float(np.median(offsets)) if len(offsets) else 0.


def load_onsets(posteriorgram):
    """(onset head, frame times) of one cached posteriorgram, on the frame
    times the labelling uses."""
    from basic_pitch.note_creation import model_frames_to_time
    with np.load(posteriorgram) as stored:
        onset = stored['onset'].astype(np.float32)
        frames = stored['contour'].shape[0]
    times = model_frames_to_time(frames)[:onset.shape[0]]
    return onset[:len(times)], times


def jams(notes):
    """A minimal JAMS document ``guitarset.load_notes`` reads: six note_midi
    blocks, one per string, even when a string is silent."""
    blocks = []
    for string in range(len(guitarset.OPEN_STRINGS)):
        data = [{'time': note.start, 'duration': note.end - note.start,
                 'value': note.pitch, 'confidence': None}
                for note in notes if note.string == string]
        blocks.append({'namespace': 'note_midi',
                       'annotation_metadata': {'data_source': str(string)},
                       'data': data, 'sandbox': {}, 'time': 0, 'duration': None})
    return {'annotations': blocks, 'file_metadata': {}, 'sandbox': {}}


def slice_notes(notes, start, stop):
    """Notes attacking inside [start, stop), re-timed to the excerpt."""
    return [guitarset.Note(note.start - start, min(note.end, stop) - start,
                           note.pitch, note.string)
            for note in notes if start <= note.start < stop]


def _prepare(job):
    """Slice one (recording, chain) into excerpts, render them, then label them.

    The render comes first because the lag is read off the rendered onset
    head, and the annotations are written last because they carry it.
    """
    recording, chain, root, chains_root, augment = job
    import librosa, soundfile
    from processing.tools import variants
    wave = librosa.load(str(recording.audio[chain]), sr=SAMPLE_RATE, mono=True)[0]
    wave = np.asarray(wave, np.float32)
    duration = len(wave)/SAMPLE_RATE
    notes = load_midi(recording.midi)

    windows, offsets = [], []
    for number in range(int(np.ceil(duration/EXCERPT_SECONDS))):
        start = number*EXCERPT_SECONDS
        stop = min(start + EXCERPT_SECONDS, duration)
        if stop - start < 5.:
            continue
        name = f'{recording.name}_{chain}_{number:03d}'
        audio = root/'audio'/chain/f'{name}.wav'
        audio.parent.mkdir(parents=True, exist_ok=True)
        if not audio.is_file():
            soundfile.write(str(audio), wave[int(start*SAMPLE_RATE):int(stop*SAMPLE_RATE)],
                            SAMPLE_RATE, subtype='PCM_16')
        variants._render(variants.Job(domain=domain(chain), name=name, audio=audio,
                                      root=chains_root, augmented=augment))
        stem = f'{domain(chain)}/{name}'
        onset, times = load_onsets(chains_root/'posteriorgrams'/f'{stem}.npz')
        offsets.extend(peak_offsets(onset, times, slice_notes(notes, start, stop)))
        windows.append((number, name, audio, start, stop))

    offset = lag(offsets)
    shifted = [guitarset.Note(n.start + offset, n.end + offset, n.pitch, n.string)
               for n in notes]
    entries = []
    for number, name, audio, start, stop in windows:
        part = slice_notes(shifted, start, stop)
        if not part:
            continue
        annotation = root/'annotation'/f'{name}.jams'
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(json.dumps(jams(part)))
        entries.append(dict(name=name, player=recording.player,
                            category=recording.category, chain=chain,
                            source=str(recording.audio[chain]), start=start,
                            lag=offset, notes=len(part),
                            audio=str(audio), annotation=str(annotation)))
    return recording.name, chain, offset, entries


def _reamp(job):
    """Render one direct-input excerpt through a distorted amp into its own
    domain. The annotation is the direct input's: an amp moves no attack."""
    row, chain, chains_root = job
    from processing.tools import variants
    name = row['name'].replace('_di_', f'_{chain}_')
    variants._render(variants.Job(domain=domain(chain), name=name,
                                  audio=Path(row['audio']), root=chains_root,
                                  amped=True))
    return dict(row, name=name, chain=chain)


def reamp(variants_count, root=ROOT, chains_root=CACHE_ROOT, workers=4):
    """Re-amp every prepared direct-input excerpt ``variants_count`` times.

    Guitar-TECHS's own amps are clean or nearly so, and so is every IDMT
    subset; a distorted electric is in no labelled corpus here. But every take
    has a direct input, and a direct input through an amp *is* a distorted
    electric with the direct input's labels, exact to the sample. Held-out
    players get one variant, to measure on; training players get all of them.
    """
    manifest = Path(root)/MANIFEST.name
    rows = json.loads(manifest.read_text())
    jobs = []
    for row in rows:
        if row['chain'] != 'di':
            continue
        count = variants_count if row['player'] in ('P1', 'P2') else 1
        jobs += [(row, f'drive-v{number}', chains_root) for number in range(1, count + 1)]
    print(f'{len(jobs)} re-amp renders')
    made = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number, entry in enumerate(pool.map(_reamp, jobs, chunksize=2), 1):
            made.append(entry)
            if number % 200 == 0:
                print(f'  {number}/{len(jobs)}', flush=True)
    merged = {row['name']: row for row in rows}
    merged.update({row['name']: row for row in made})
    manifest.write_text(json.dumps(sorted(merged.values(), key=lambda r: r['name']),
                                   indent=1))
    print(f'{len(made)} re-amped excerpts, {len(merged)} in {manifest}')


def excerpts(root=ROOT):
    """Prepared excerpts as ``guitarset.Excerpt`` records.

    ``progression`` and ``style`` carry the category (chords, scales, ...) and
    ``key`` the chain, so the stratified sampler and the split logic work
    without knowing which corpus an excerpt came from. Empty when nothing has
    been prepared, so callers can mix the corpus in opportunistically.
    """
    manifest = Path(root)/MANIFEST.name
    if not manifest.is_file():
        return []
    rows = json.loads(manifest.read_text())
    return [guitarset.Excerpt(name=row['name'], audio=Path(row['audio']),
                              annotation=Path(row['annotation']),
                              player=row['player'], progression=row['category'],
                              tempo=0, key=row['chain'], style=row['category'])
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cache-root', type=Path, default=CACHE_ROOT)
    parser.add_argument('--chains', nargs='*', default=list(CHAINS))
    parser.add_argument('--categories', nargs='*',
                        help='Only these categories (chords scales singlenotes techniques music)')
    parser.add_argument('--prepare', action='store_true',
                        help='Slice, annotate and render; otherwise just list')
    parser.add_argument('--reamp', type=int, metavar='N',
                        help='Re-amp the prepared direct inputs through N distorted '
                             'amps per training excerpt (one per held-out excerpt), '
                             'as domains gt-drive-v1 .. gt-drive-vN')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    if args.reamp:
        reamp(args.reamp, args.root, args.cache_root, args.workers)
        return

    found = recordings(args.raw)
    if args.categories:
        found = [r for r in found if r.category in args.categories]
    by_player = defaultdict(list)
    for recording in found:
        by_player[recording.player].append(recording)
    for player in sorted(by_player):
        print(f'{player}: ' + ', '.join(
            f'{r.category}/{r.stem} [{" ".join(sorted(r.audio))}]'
            for r in by_player[player]))
    if not args.prepare:
        return

    jobs = [(recording, chain, args.root, args.cache_root, False)
            for recording in found for chain in args.chains
            if chain in recording.audio]
    print(f'\n{len(jobs)} (recording, chain) pairs to prepare')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, chain, offset, entries in pool.map(_prepare, jobs):
            rows.extend(entries)
            print(f'  {name} {chain}: lag {offset*1000:+.0f} ms, '
                  f'{len(entries)} excerpts, '
                  f'{sum(e["notes"] for e in entries)} notes', flush=True)
    manifest = args.root/MANIFEST.name
    existing = json.loads(manifest.read_text()) if manifest.is_file() else []
    merged = {row['name']: row for row in existing}
    merged.update({row['name']: row for row in rows})
    manifest.write_text(json.dumps(sorted(merged.values(), key=lambda r: r['name']),
                                   indent=1))
    print(f'\n{len(rows)} excerpts prepared, {len(merged)} in {manifest}')


if __name__ == '__main__':
    main()
