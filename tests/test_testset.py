"""The end-to-end test set's label format and scoring."""
from pathlib import Path
import tempfile
import unittest

from processing.tools import testset
from processing.tools.testset import Note


def label_file(directory, text, name='take.tab.txt'):
    path = Path(directory)/name
    path.write_text(text)
    return testset.parse(path)


class FormatTests(unittest.TestCase):
    def test_header_events_chords_and_pitch_only_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            label = label_file(directory, """
                # a comment
                video: clips/My Take.mov   # trailing comment
                tuning: standard
                capo: 2
                @0.50 5:0
                @1.00 3:0 2:1
                @1.50 B3
            """.replace('\n                ', '\n'))
        self.assertEqual(label.video, Path('clips/My Take.mov'))
        self.assertEqual(label.capo, 2)
        self.assertTrue(label.timed)
        self.assertFalse(label.draft)
        self.assertEqual([len(e) for e in label.events], [1, 2, 1])
        # Frets count from the capo; pitches include it.
        self.assertEqual(label.events[0][0], Note(47, .5, 5, 2))
        self.assertEqual(label.events[1][1], Note(62, 1., 2, 3))
        self.assertEqual(label.events[2][0], Note(59, 1.5))

    def test_tunings(self):
        self.assertEqual(testset.parse_tuning('drop-d'), (64, 59, 55, 50, 45, 38))
        self.assertEqual(testset.parse_tuning('E4 B3 G3 D3 A2 D2'), (64, 59, 55, 50, 45, 38))
        self.assertEqual(testset.parse_tuning('62,57,53,48,43,38'), (62, 57, 53, 48, 43, 38))
        with self.assertRaises(ValueError):
            testset.parse_tuning('E2 A2 D3 G3 B3 E4')

    def test_mistakes_are_reported_with_their_line(self):
        cases = {'5:0 5:2': 'twice', '7:0': 'no string 7', 'x': "got 'x'", '@1.0': 'no notes'}
        for line, message in cases.items():
            with self.subTest(line=line), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    label_file(directory, f'video: v.mov\n{line}\n')

    def test_untimed_when_any_line_lacks_a_time(self):
        with tempfile.TemporaryDirectory() as directory:
            label = label_file(directory, '@0.5 5:0\n5:2\n')
        self.assertFalse(label.timed)

    def test_draft_round_trip(self):
        notes = [Note(45, .5, 5, 0), Note(50, 1., 4, 0), Note(57, 1.01, 3, 2), Note(59, 1.5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'take.tab.txt'
            testset.write(path, notes, Path('clips/take.mov'), (64, 59, 55, 50, 45, 40))
            label = testset.parse(path)
        self.assertTrue(label.draft)
        self.assertEqual([[(n.pitch, n.string, n.fret) for n in e] for e in label.events],
                         [[(45, 5, 0)], [(50, 4, 0), (57, 3, 2)], [(59, None, None)]])


class ScoreTests(unittest.TestCase):
    def label(self, text):
        with tempfile.TemporaryDirectory() as directory:
            return label_file(directory, text)

    def test_timed_scoring_counts_misses_extras_and_strings(self):
        label = self.label('@0.50 5:0\n@1.00 3:0\n@1.50 2:1\n@2.00 B3\n')
        predicted = [Note(45, .52, 5, 0),     # right
                     Note(55, 1.03, 4, 5),    # right pitch, wrong string
                     Note(59, 2.01),          # the pitch-only note
                     Note(70, 3.0, 1, 6)]     # extra; 2:1 at 1.5 s is missed
        counts = testset.score(label, predicted)
        self.assertTrue(counts['timed'])
        self.assertEqual((counts['matched'], counts['positioned'], counts['right_string']),
                         (3, 2, 1))
        self.assertEqual(counts['open'], 2)
        self.assertEqual(counts['open_right'], 1)
        self.assertEqual([m['pitch'] for m in counts['missed']], ['C4'])
        self.assertEqual([e['pitch'] for e in counts['extra']], ['A#4'])
        self.assertEqual(counts['wrong_string'][0]['got'], '4:5')
        self.assertEqual(counts['onset_offset_ms'], 20.)
        summary = testset.summarise(counts)
        self.assertAlmostEqual(summary['note_f1'], .75)
        self.assertAlmostEqual(summary['string'], .5)
        # One of three judged predictions, one of three positioned label notes.
        self.assertAlmostEqual(summary['tab_f1'], 1/3)

    def test_late_onsets_miss_unless_the_tolerance_allows(self):
        label = self.label('@0.50 5:0\n')
        late = [Note(45, .58, 5, 0)]
        self.assertEqual(testset.score(label, late)['matched'], 0)
        self.assertEqual(testset.score(label, late, tolerance=.1)['matched'], 1)

    def test_untimed_labels_align_in_order_whatever_the_times(self):
        label = self.label('5:0\n3:0 2:1\n5:0\n')
        predicted = [Note(45, 10., 5, 0), Note(64, 10.5, 1, 0),   # an extra note between
                     Note(55, 11., 3, 0), Note(60, 11.02, 2, 1), Note(45, 12., 4, 0)]
        counts = testset.score(label, predicted)
        self.assertFalse(counts['timed'])
        self.assertEqual(counts['matched'], 4)
        self.assertEqual(counts['right_string'], 3)
        self.assertEqual([e['pitch'] for e in counts['extra']], ['E4'])

    def test_order_matters_for_untimed_labels(self):
        label = self.label('5:0\n4:0\n')
        swapped = [Note(50, 1., 4, 0), Note(45, 2., 5, 0)]
        self.assertEqual(testset.score(label, swapped)['matched'], 1)


if __name__ == '__main__':
    unittest.main()
