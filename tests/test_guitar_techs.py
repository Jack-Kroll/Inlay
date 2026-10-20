"""The Guitar-TECHS loader: MIDI tracks to notes, lag estimation, JAMS round trip."""
from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
import numpy as np

from processing.tools import guitar_techs, guitarset


def write_midi(path, per_string, tempo=1000000, ticks_per_beat=960):
    """A file shaped like Guitar-TECHS's: a tempo track, then one named track
    per string, ``per_string`` mapping track name to [(start, end, pitch)]."""
    import mido
    file = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    head = mido.MidiTrack()
    head.append(mido.MetaMessage('track_name', name='AllNotes', time=0))
    head.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    file.tracks.append(head)
    for name, notes in per_string.items():
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        events = []
        for start, end, pitch in notes:
            events.append((start, 'note_on', pitch))
            events.append((end, 'note_off', pitch))
        events.sort()
        at = 0
        for when, kind, pitch in events:
            tick = int(round(mido.second2tick(when, ticks_per_beat, tempo)))
            track.append(mido.Message(kind, note=pitch, velocity=80 if kind == 'note_on' else 0,
                                      time=tick - at))
            at = tick
        file.tracks.append(track)
    file.save(str(path))


class LoadMidiTest(unittest.TestCase):
    def test_tracks_name_the_string(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'notes.mid'
            write_midi(path, {'E': [(0., .5, 40), (1., 1.5, 45)],
                              'B': [(.25, .75, 59)], 'e': [(2., 2.5, 64)],
                              'A': [], 'D': [], 'G': []})
            notes = guitar_techs.load_midi(path)
        self.assertEqual([(n.start, n.pitch, n.string) for n in notes],
                         [(0., 40., 0), (.25, 59., 4), (1., 45., 0), (2., 64., 5)])
        self.assertAlmostEqual(notes[0].end, .5, places=3)
        # Fret follows from pitch and string, as it does for GuitarSet.
        self.assertEqual([n.fret for n in notes], [0, 0, 5, 0])

    def test_impossible_frets_are_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'notes.mid'
            # 38 on the low E is below the open string; 70 on it is fret 30.
            write_midi(path, {'E': [(0., .5, 38), (1., 1.5, 70), (2., 2.5, 52)]})
            notes = guitar_techs.load_midi(path)
        self.assertEqual([(n.pitch, n.fret) for n in notes], [(52., 12)])

    def test_tempo_is_honoured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'notes.mid'
            write_midi(path, {'G': [(3., 3.5, 55)]}, tempo=500000)
            notes = guitar_techs.load_midi(path)
        self.assertAlmostEqual(notes[0].start, 3., places=3)


class LagTest(unittest.TestCase):
    def test_recovers_a_constant_offset(self):
        from processing.music import verify
        fps = verify.FPS
        frames = int(6*fps)
        times = np.arange(frames)/fps
        onset = np.zeros((frames, 88), np.float32)
        notes = [guitarset.Note(.5, 1., 40., 0), guitarset.Note(1.5, 2., 59., 4),
                 guitarset.Note(2.5, 3., 64., 5), guitarset.Note(3.5, 4., 45., 1)]
        offset = -.046   # The attacks sit 46 ms before the labels say.
        for note in notes:
            at = int(round((note.start + offset)*fps))
            onset[at, int(note.pitch) - verify.LOWEST_BIN] = 1.
        # A loud peak at another pitch, right on a label, must not count.
        onset[int(round(1.5*fps)), 70 - verify.LOWEST_BIN] = 1.
        offsets = guitar_techs.peak_offsets(onset, times, notes)
        self.assertEqual(len(offsets), 4)
        self.assertAlmostEqual(guitar_techs.lag(offsets), offset, delta=.5/fps)

    def test_no_evidence_means_no_shift(self):
        self.assertEqual(guitar_techs.lag([]), 0.)


class JamsTest(unittest.TestCase):
    def test_round_trips_through_guitarset(self):
        notes = [guitarset.Note(0., .5, 40., 0), guitarset.Note(.2, .4, 59., 4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'x.jams'
            path.write_text(json.dumps(guitar_techs.jams(notes)))
            back = guitarset.load_notes(path)
        self.assertEqual(back, notes)

    def test_slice_retimes_and_clips(self):
        notes = [guitarset.Note(29., 31., 40., 0), guitarset.Note(31., 32., 45., 1),
                 guitarset.Note(59.9, 60.5, 50., 2)]
        part = guitar_techs.slice_notes(notes, 30., 60.)
        self.assertEqual(part, [guitarset.Note(1., 2., 45., 1),
                                guitarset.Note(29.9, 30., 50., 2)])


class RecordingsTest(unittest.TestCase):
    def test_walks_the_archive_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner = root/'P2_scales'/'P2_scales'
            (inner/'midi').mkdir(parents=True)
            (inner/'midi'/'midi_allscales.mid').write_bytes(b'')
            for chain, (sub, prefix) in guitar_techs.CHAINS.items():
                (inner/sub).mkdir(parents=True)
                ext = '.mp3' if sub.startswith('video') else '.wav'
                (inner/sub/f'{prefix}allscales{ext}').write_bytes(b'')
            (root/'__MACOSX'/'P2_scales'/'midi').mkdir(parents=True)
            (root/'__MACOSX'/'P2_scales'/'midi'/'._midi_allscales.mid').write_bytes(b'')
            found = guitar_techs.recordings(root)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].player, found[0].category, found[0].stem),
                         ('P2', 'scales', 'allscales'))
        self.assertEqual(sorted(found[0].audio), ['amp', 'di', 'ego', 'exo'])
        self.assertEqual(found[0].name, 'P2_scales_allscales')


if __name__ == '__main__':
    unittest.main()
