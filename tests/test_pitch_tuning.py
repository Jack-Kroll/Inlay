"""GuitarSet loading and Basic Pitch scoring, without the dataset on disk."""
import json
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
import numpy as np

from processing.music import pitch
from processing.music.fusion import plan_positions
from processing.tools import guitarset
from processing.tools.tune_pitch import CURRENT, Setting, Tally, counts, estimate, grid


def jams(strings):
    """A JAMS document holding one note_midi annotation per string."""
    return {'annotations': [
        {'namespace': 'note_midi',
         'annotation_metadata': {'data_source': str(index)},
         'data': [{'time': time, 'duration': duration, 'value': value, 'confidence': None}
                  for time, duration, value in observations]}
        for index, observations in enumerate(strings)
    ] + [{'namespace': 'tempo', 'annotation_metadata': {'data_source': '0'},
          'data': [{'time': 0., 'duration': 0., 'value': 120., 'confidence': None}]}]}


def write(root, name, strings):
    path = Path(root)/f'{name}.jams'
    path.write_text(json.dumps(jams(strings)))
    return path


class LoadNotesTests(unittest.TestCase):
    def test_reads_every_string_sorted_and_maps_pitch_to_fret(self):
        with tempfile.TemporaryDirectory() as tmp:
            # String 0 is the low E: 40 open, so 43.02 is the third fret.
            path = write(tmp, 'take', [[(1.5, .4, 43.02)], [(.5, .4, 45.01)],
                                       [(2.0, .3, 50.)], [(.2, .3, 55.)],
                                       [(.1, .3, 59.)], [(.9, .3, 71.98)]])
            notes = guitarset.load_notes(path)
        self.assertEqual([round(note.pitch) for note in notes], [59, 55, 45, 72, 43, 50])
        self.assertEqual([note.start for note in notes], sorted(note.start for note in notes))
        self.assertEqual(notes[4].fret, 3)
        self.assertEqual(notes[3].fret, 8)  # 72 on the high E, open 64.
        self.assertEqual([note.end - note.start > 0 for note in notes], [True]*6)

    def test_zero_length_observations_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, 'take', [[(1., 0., 43.)], [(1., .4, 45.)], [], [], [], []])
            notes = guitarset.load_notes(path)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].string, 1)

    def test_a_missing_string_annotation_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'take.jams'
            document = jams([[(1., .4, 43.)]]*6)
            del document['annotations'][3]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'expected 6'):
                guitarset.load_notes(path)


class StratifiedTests(unittest.TestCase):
    def excerpts(self, players=6, styles=('comp', 'solo'), per_group=10):
        return [guitarset.Excerpt(name=f'{player:02d}_BN1-100-C{index}_{style}',
                                  audio=Path('a.wav'), annotation=Path('a.jams'),
                                  player=f'{player:02d}', progression='BN1', tempo=100,
                                  key='C', style=style)
                for player in range(players) for style in styles
                for index in range(per_group)]

    def test_take_is_spread_evenly_over_player_and_style(self):
        chosen = guitarset.stratified(self.excerpts(), 24)
        self.assertEqual(len(chosen), 24)
        groups = {}
        for excerpt in chosen:
            groups.setdefault((excerpt.style, excerpt.player), []).append(excerpt)
        self.assertEqual(len(groups), 12)
        self.assertEqual({len(group) for group in groups.values()}, {2})

    def test_is_deterministic_and_passes_through_when_not_limiting(self):
        items = self.excerpts()
        self.assertEqual([e.name for e in guitarset.stratified(items, 17)],
                         [e.name for e in guitarset.stratified(items, 17)])
        self.assertEqual(len(guitarset.stratified(items, 500)), len(items))


@unittest.skipUnless(find_spec('mir_eval'), 'Install the transcribe extra for pitch scoring')
class CountsTests(unittest.TestCase):
    def reference(self, *notes):
        return [guitarset.Note(start, end, pitch, 0) for start, end, pitch in notes]

    def test_matches_within_onset_and_pitch_tolerance(self):
        reference = self.reference((1., 1.5, 60.), (2., 2.5, 64.))
        # 20 ms late and 10 cents sharp still matches; 200 ms late does not.
        self.assertEqual(counts(reference, [(1.02, 1.5, 60), (2.2, 2.5, 64)]), (1, 2, 2))
        self.assertEqual(counts(reference, [(1.02, 1.5, 60), (2.01, 2.6, 64)]), (2, 2, 2))

    def test_wrong_pitch_does_not_match(self):
        reference = self.reference((1., 1.5, 60.))
        self.assertEqual(counts(reference, [(1., 1.5, 61)]), (0, 1, 1))

    def test_empty_sides_are_counted_not_matched(self):
        self.assertEqual(counts(self.reference((1., 1.5, 60.)), []), (0, 0, 1))
        self.assertEqual(counts([], [(1., 1.5, 60)]), (0, 1, 0))


class TallyTests(unittest.TestCase):
    def test_micro_averages_over_excerpts(self):
        tally = Tally()
        tally.add(3, 4, 6)
        tally.add(1, 6, 4)
        self.assertAlmostEqual(tally.precision, .4)
        self.assertAlmostEqual(tally.recall, .4)
        self.assertAlmostEqual(tally.f1, .4)

    def test_no_estimates_scores_zero_rather_than_dividing_by_zero(self):
        tally = Tally()
        tally.add(0, 0, 5)
        self.assertEqual((tally.precision, tally.recall, tally.f1), (0., 0., 0.))


@unittest.skipUnless(find_spec('basic_pitch'), 'Install the transcribe extra for pitch decoding')
class EstimateTests(unittest.TestCase):
    def output(self, midi=60, frames=200, start=50, length=40):
        note = np.zeros((frames, 88), np.float32)
        onset = np.zeros((frames, 88), np.float32)
        note[start:start + length, midi - 21] = .9
        onset[start, midi - 21] = .9
        return {'note': note, 'onset': onset,
                'contour': np.zeros((frames, 264), np.float32)}

    def test_a_clear_activation_becomes_one_note_of_that_pitch(self):
        notes = estimate(self.output(), Setting())
        self.assertEqual(len(notes), 1)
        start, end, midi = notes[0]
        self.assertEqual(midi, 60)
        self.assertLess(start, end)

    def test_thresholds_above_the_activation_suppress_it(self):
        self.assertEqual(estimate(self.output(), Setting(onset_threshold=.95,
                                                         frame_threshold=.95)), [])

    def test_frequency_bounds_exclude_out_of_range_pitches(self):
        # MIDI 36 is 65 Hz, below the 70 Hz floor the guitar range sets.
        self.assertEqual(estimate(self.output(midi=36), Setting()), [])
        self.assertEqual(len(estimate(self.output(midi=36), Setting(minimum_frequency=30.))), 1)

    def test_frequency_sweep_is_independent_of_setting_order(self):
        output = self.output(midi=36)
        original = {key: value.copy() for key, value in output.items()}
        wide = Setting(minimum_frequency=30.)
        expected = estimate(output, wide)
        self.assertEqual(len(expected), 1)
        self.assertEqual(estimate(output, Setting()), [])
        self.assertEqual(estimate(output, wide), expected)
        for key in output:
            np.testing.assert_array_equal(output[key], original[key])

    def test_minimum_note_length_drops_short_activations(self):
        # 40 frames at 86 fps is about 465 ms; 4 frames is about 46 ms.
        brief = self.output(length=4)
        self.assertEqual(len(estimate(brief, Setting(min_note_ms=30.))), 1)
        self.assertEqual(estimate(brief, Setting(min_note_ms=128.)), [])


class GridTests(unittest.TestCase):
    class Args:
        onset = [.4, .5]
        frame = [.2, .3]
        min_note_ms = [70.]
        min_freq = [70.]
        max_freq = [1400.]
        melodia = [True]
        infer_onsets = [True]

    def test_covers_the_product_and_appends_the_shipped_defaults(self):
        settings = grid(self.Args())
        self.assertEqual(len(settings), 5)  # The 2x2 product, plus CURRENT.
        self.assertIn(CURRENT, settings)
        self.assertEqual(len(set(settings)), len(settings))

    def test_shipped_defaults_are_not_duplicated_when_the_grid_covers_them(self):
        args = self.Args()
        args.onset = [CURRENT.onset_threshold]
        args.frame = [CURRENT.frame_threshold]
        args.min_note_ms = [CURRENT.min_note_ms]
        self.assertEqual(grid(args), [CURRENT])


class PosteriorgramTests(unittest.TestCase):
    """The activations a visual-candidate rescue would consult."""

    def build(self, frames=10):
        note = np.zeros((frames, 88), dtype=np.float32)
        onset = np.zeros((frames, 88), dtype=np.float32)
        times = np.arange(frames, dtype=np.float64)/10
        return pitch.Posteriorgram(note, onset, times)

    def test_peak_is_located_by_the_onset_head(self):
        """The onset head is the one trained to say where an attack is."""
        board = self.build()
        column = 54 - pitch.LOWEST_BIN
        board.onset[3, column], board.onset[7, column] = .4, .9
        board.note[5, column] = .8
        found = board.evidence(54, 0., .9)
        self.assertAlmostEqual(found.time, .7)
        self.assertAlmostEqual(found.onset, .9)
        # The frame peak is the window's, not whatever sits under the onset.
        self.assertAlmostEqual(found.frame, .8)

    def test_window_restricts_what_is_seen(self):
        board = self.build()
        column = 54 - pitch.LOWEST_BIN
        board.onset[8, column] = .9
        self.assertAlmostEqual(board.evidence(54, 0., .4).onset, 0.)
        self.assertAlmostEqual(board.evidence(54, .7, .9).onset, .9)

    def test_pitch_outside_the_model_range_reports_nothing(self):
        """None must not be confused with 'the model saw nothing there'."""
        board = self.build()
        self.assertIsNone(board.evidence(pitch.LOWEST_BIN - 1, 0., .9))
        self.assertIsNone(board.evidence(pitch.LOWEST_BIN + 88, 0., .9))

    def test_window_past_the_recording_still_reports_a_frame(self):
        """A note near the end must not fall off into a zero-width slice."""
        board = self.build()
        self.assertIsNotNone(board.evidence(54, 5., 9.))


class ShippedDefaultsTests(unittest.TestCase):
    def test_setting_defaults_track_detect_notes(self):
        """A sweep compares against what inlay ships, so the two must agree."""
        self.assertEqual(Setting(), CURRENT)

    def test_current_is_read_from_the_detect_notes_signature(self):
        import inspect
        parameters = inspect.signature(pitch.detect_notes).parameters
        self.assertEqual(CURRENT.onset_threshold, parameters['onset_threshold'].default)
        self.assertEqual(CURRENT.frame_threshold, parameters['frame_threshold'].default)
        self.assertEqual(CURRENT.min_note_ms, parameters['minimum_note_length'].default)



class Sounded:
    def __init__(self, pitch):
        self.pitch = pitch


class PlanPositionsTests(unittest.TestCase):
    """The neck trajectory is chosen for the whole clip, not chord by chord."""

    def groups(self, *pitches):
        return [[Sounded(p) for p in group] for group in pitches]

    def test_a_seen_hand_pins_its_group(self):
        planned = plan_positions(self.groups([64], [64], [64]), [None, 12., None])
        self.assertEqual(planned[1], 12.)

    def test_a_gap_between_anchors_is_bridged_rather_than_jumping_home(self):
        """The blind middle group should sit between the hands seen either side."""
        planned = plan_positions(self.groups([64], [64], [64], [64], [64]),
                                 [9., None, None, None, 9.])
        self.assertEqual(planned, [9.]*5)

    def test_pitches_playable_only_high_move_the_hand_up(self):
        # 84 is only reachable at fret 20 on the high E, nowhere else by fret 22.
        planned = plan_positions(self.groups([84], [84]), [None, None])
        self.assertTrue(all(position is not None and position >= 15 for position in planned), planned)

    def test_a_pitch_with_no_spelling_at_all_does_not_steer_the_hand(self):
        """88 is past fret 22 on every string, so it is no evidence of position."""
        anchored = plan_positions(self.groups([88], [88]), [7., None])
        self.assertEqual(anchored, [7., 7.])

    def test_movement_is_paid_for_not_free(self):
        """One stray note should not drag the hand away from a settled position."""
        settled = self.groups([45], [45], [45], [64], [45], [45], [45])
        planned = plan_positions(settled, [None]*7)
        self.assertEqual(len(set(planned)), 1, planned)

    def test_no_groups_and_no_finite_option_are_both_handled(self):
        self.assertEqual(plan_positions([], []), [])
        self.assertEqual(len(plan_positions(self.groups([64]), [None])), 1)


if __name__ == '__main__':
    unittest.main()
