"""Pairing two tab JSONL files, without either being ground truth."""
import json
from pathlib import Path
import tempfile
import unittest

from processing.tools.compare_tab import cell, load, pair


def note(start, pitch, string=None, fret=None, support='fingered', **extra):
    return {'kind': 'note', 'startSeconds': start, 'pitch': pitch,
            'name': f'P{pitch}', 'durationSeconds': .2, 'string': string,
            'fret': fret, 'support': support, **extra}


def write(root, name, notes, summary=None):
    path = Path(root)/name
    rows = [summary or {'kind': 'summary', 'notes': len(notes)}] + notes
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    return path


class Load(unittest.TestCase):
    def test_summary_is_separated_from_the_notes(self):
        with tempfile.TemporaryDirectory() as root:
            path = write(root, 'a.jsonl', [note(0., 64)], {'kind': 'summary', 'notes': 1})
            summary, notes = load(path)
        self.assertEqual(summary['notes'], 1)
        self.assertEqual(len(notes), 1)

    def test_a_file_of_only_notes_still_loads(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'a.jsonl'
            path.write_text(json.dumps(note(0., 64)))
            summary, notes = load(path)
        self.assertEqual(summary, {})
        self.assertEqual(len(notes), 1)


class Pairing(unittest.TestCase):
    def test_same_pitch_within_tolerance_is_one_note(self):
        pairs, dropped, added = pair([note(1., 64)], [note(1.02, 64)], tolerance=.05)
        self.assertEqual(pairs, [(0, 0)])
        self.assertEqual((dropped, added), ([], []))

    def test_same_pitch_beyond_tolerance_is_a_drop_and_an_add(self):
        pairs, dropped, added = pair([note(1., 64)], [note(1.5, 64)], tolerance=.05)
        self.assertEqual(pairs, [])
        self.assertEqual((dropped, added), ([0], [0]))

    def test_a_different_pitch_never_pairs(self):
        pairs, dropped, added = pair([note(1., 64)], [note(1., 65)], tolerance=.05)
        self.assertEqual(pairs, [])
        self.assertEqual((dropped, added), ([0], [0]))

    def test_the_nearest_candidate_wins_not_the_first(self):
        # Pairing in order would marry 1.00 to 1.04 and strand the better match.
        before = [note(1., 64)]
        after = [note(1.04, 64), note(1.001, 64)]
        pairs, _, added = pair(before, after, tolerance=.05)
        self.assertEqual(pairs, [(0, 1)])
        self.assertEqual(added, [0])

    def test_one_note_cannot_pair_twice(self):
        before = [note(1., 64), note(1.01, 64)]
        after = [note(1.005, 64)]
        pairs, dropped, added = pair(before, after, tolerance=.05)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(added, [])

    def test_empty_sides_are_all_drops_or_all_adds(self):
        self.assertEqual(pair([note(1., 64)], []), ([], [0], []))
        self.assertEqual(pair([], [note(1., 64)]), ([], [], [0]))


class Cell(unittest.TestCase):
    def test_a_placed_note_reads_as_string_over_fret(self):
        self.assertEqual(cell(note(0., 64, string=2, fret=5)), '2/5')

    def test_an_unassigned_note_says_so_rather_than_showing_none(self):
        self.assertEqual(cell(note(0., 64)), 'unassigned')

    def test_fret_zero_is_not_mistaken_for_missing(self):
        self.assertEqual(cell(note(0., 64, string=1, fret=0)), '1/0')


if __name__ == '__main__':
    unittest.main()
