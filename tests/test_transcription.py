import unittest
from unittest.mock import patch
import cv2
import numpy as np

from processing.vision.tracker import Observation
from processing.music.fretboard import (
    board_transform, fret_ratio, ratio_to_fret, DEFAULT_STRING_INSET,
)
from processing.music.fusion import (
    assign_positions, finger_contacts, unexplained_contacts,
)
from processing.music.hands import FINGERTIPS, Hand, fretting_hand
from processing.music.pitch import NoteEvent
from processing.music.transcribe import FrameRecord, group_notes, nearest_record, transcribe_notes
from processing.music.tuning import STANDARD_TUNING, candidates, note_name, pitch_at, validate_tuning

# A deliberately oblique view so nothing passes by accident on a fronto-parallel board.
VIEW = np.array([[520., 40., 60.], [90., 300., 120.], [.35, .05, 1.]])


def project(points):
    return cv2.perspectiveTransform(
        np.asarray(points, np.float32).reshape(-1, 1, 2), VIEW).reshape(-1, 2)


def wire(number):
    return project([[fret_ratio(number), 0.], [fret_ratio(number), 1.]]).astype(np.float32)


def board(numbers=(1, 3, 5, 7, 9, 12), nut=True):
    return Observation(
        np.array(project([[0, 0], [.6, 0], [.6, 1], [0, 1]]), np.float32),
        # Alternating endpoint order: wire endpoints carry no string identity.
        [wire(n)[::-1] if i % 2 else wire(n) for i, n in enumerate(numbers)],
        wire(0) if nut else None, .9, list(numbers), 'nut-anchored fit')


def required_transform(*args, **kwargs):
    transform = board_transform(*args, **kwargs)
    assert transform is not None
    return transform


def hand_on(transform, placements, parked=None):
    points = np.tile(parked if parked is not None else project([[.02, 3.]])[0], (21, 1))
    for finger, (string, fret) in placements.items():
        points[FINGERTIPS[finger]] = transform.board_point(string, fret)
    return Hand(points.astype(np.float32), 'Left', .9)


class FretboardGeometryTests(unittest.TestCase):
    def test_fret_ratio_round_trip(self):
        frets = np.arange(0, 23)
        np.testing.assert_allclose(ratio_to_fret(fret_ratio(frets)), frets, atol=1e-9)
        self.assertEqual(ratio_to_fret(1.), np.inf)

    def test_every_cell_round_trips_through_perspective(self):
        transform = required_transform(board())
        assert transform is not None
        self.assertLess(transform.residual, 1e-4)
        for string in range(1, 7):
            for fret in (1, 2, 5, 8, 11, 15, 22):
                point = transform.board_point(string, fret)
                fret_value, string_value = transform.locate([point])
                self.assertEqual(int(np.ceil(fret_value[0] - 1e-9)), fret)
                self.assertAlmostEqual(string_value[0], string, places=3)

    def test_point_just_behind_a_wire_sounds_that_fret(self):
        transform = required_transform(board())
        point = transform.to_image([[fret_ratio(5) - 1e-4, .5]])
        self.assertEqual(int(np.ceil(transform.locate(point)[0][0])), 5)

    def test_flipped_order_mirrors_string_numbering(self):
        normal = required_transform(board())
        flipped = required_transform(board(), flipped=True)
        point = normal.board_point(1, 5)
        self.assertAlmostEqual(normal.locate([point])[1][0], 1, places=3)
        self.assertAlmostEqual(flipped.locate([point])[1][0], 6, places=3)

    def test_cell_tolerance_shrinks_up_the_neck(self):
        transform = required_transform(board())
        self.assertGreater(transform.cell_radius(3, 1), transform.cell_radius(3, 12))

    def test_string_identity_survives_camera_rotation(self):
        from processing.vision.tracker import transform_observation
        observation = board()
        point = required_transform(observation).board_point(2, 5)
        for angle in range(0, 360, 15):
            matrix = np.vstack([cv2.getRotationMatrix2D((300, 300), angle, 1), [0, 0, 1]])
            rotated = transform_observation(observation, matrix)
            moved = cv2.perspectiveTransform(point.reshape(1, 1, 2), matrix).reshape(1, 2)
            fret, string = required_transform(rotated).locate(moved)
            self.assertAlmostEqual(string[0], 2, places=3)
            self.assertEqual(int(np.ceil(fret[0])), 5)

    def test_string_identity_does_not_depend_on_svd_axis_sign(self):
        observation = board()
        normal = required_transform(observation)
        svd = np.linalg.svd
        def equivalent_svd(*args, **kwargs):
            u, s, vt = svd(*args, **kwargs)
            # Both sign choices are mathematically equivalent SVDs.
            u[:, 0] *= -1
            vt[0] *= -1
            return u, s, vt
        with patch('processing.music.fretboard.np.linalg.svd', side_effect=equivalent_svd):
            equivalent = required_transform(observation)
        point = normal.board_point(2, 5)
        self.assertAlmostEqual(equivalent.locate([point])[1][0], 2, places=3)

    def test_single_string_is_centered_and_rejects_offboard_fingers(self):
        transform = required_transform(board(), strings=1)
        point = transform.board_point(1, 5)
        self.assertAlmostEqual(transform.to_board([point])[0, 1], .5, places=4)
        self.assertAlmostEqual(transform.locate([point])[1][0], 1, places=4)
        contacts = finger_contacts(hand_on(transform, {}), transform)
        self.assertFalse(any(c.on_board for c in contacts))

    def test_rejects_insufficient_or_degenerate_evidence(self):
        self.assertIsNone(board_transform(board(numbers=(5,), nut=False)))
        self.assertIsNone(board_transform(board(numbers=(3, 5), nut=False)))
        unnumbered = board()
        unnumbered.numbers = [None]*len(unnumbered.frets)
        unnumbered.nut = None
        self.assertIsNone(board_transform(unnumbered))

    def test_ragged_wire_endpoints_still_recover_the_board(self):
        """Real detections stop where evidence stops, not at the board edge.

        Each wire is truncated by a different amount on each side. Only clipping
        every wire to a common fitted edge makes the across coordinate mean the
        same thing on all of them.
        """
        numbers = (1, 3, 5, 7, 9, 12)
        rng = np.random.default_rng(7)
        frets = []
        for number in numbers:
            low, high = rng.uniform(.05, .3), rng.uniform(.7, .95)
            frets.append(project([[fret_ratio(number), low],
                                  [fret_ratio(number), high]]).astype(np.float32))
        observation = Observation(
            np.array(project([[0, 0], [.6, 0], [.6, 1], [0, 1]]), np.float32),
            frets, wire(0), .9, list(numbers), 'nut-anchored fit')
        transform = required_transform(observation)
        assert transform is not None
        self.assertLess(transform.edge_residual, .01)
        for string in (1, 3, 6):
            for fret in (2, 5, 9):
                point = transform.board_point(string, fret)
                fret_value, string_value = transform.locate([point])
                self.assertEqual(int(np.ceil(fret_value[0] - 1e-9)), fret)
                self.assertAlmostEqual(string_value[0], string, places=2)

    def test_wildly_inconsistent_wires_are_rejected(self):
        numbers = (1, 3, 5, 7, 9, 12)
        rng = np.random.default_rng(3)
        frets = []
        for number in numbers:
            # Scatter the wires off their equal-tempered positions entirely.
            ratio = fret_ratio(number) + rng.uniform(-.2, .2)
            frets.append(project([[ratio, 0.], [ratio, 1.]]).astype(np.float32))
        observation = Observation(
            np.array(project([[0, 0], [.9, 0], [.9, 1], [0, 1]]), np.float32),
            frets, wire(0), .9, list(numbers), 'nut-anchored fit')
        self.assertIsNone(board_transform(observation))

    def test_estimates_do_not_count_as_anchors(self):
        from processing.vision.tracker import EstimatedFret
        observation = board(numbers=(3, 5), nut=False)
        observation.estimates = [EstimatedFret(wire(9), 9), EstimatedFret(wire(12), 12)]
        self.assertIsNone(board_transform(observation))


class TuningTests(unittest.TestCase):
    def test_candidates_cover_every_string_and_order_by_fret(self):
        self.assertEqual(candidates(64)[0], (1, 0))
        self.assertEqual(candidates(40), [(6, 0)])
        self.assertEqual([f for _, f in candidates(64)], sorted(f for _, f in candidates(64)))
        self.assertEqual(candidates(64, max_fret=4), [(1, 0)])

    def test_tuning_must_be_descending(self):
        self.assertEqual(validate_tuning(STANDARD_TUNING), STANDARD_TUNING)
        with self.assertRaisesRegex(ValueError, 'descending'):
            validate_tuning([40, 45, 50, 55, 59, 64])
        with self.assertRaises(ValueError):
            validate_tuning([200])

    def test_pitch_at_matches_candidates(self):
        self.assertIn((3, 5), candidates(pitch_at(3, 5)))


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.transform = required_transform(board())

    def contacts(self, placements):
        return finger_contacts(hand_on(self.transform, placements), self.transform)

    def assign(self, pitches, placements):
        notes = [NoteEvent(0., 1., p, .8) for p in pitches]
        return assign_positions(notes, self.contacts(placements)), self.contacts(placements)

    def test_fingertip_evidence_beats_an_unoccupied_open_string(self):
        # E4 is open string 1 and also string 2 fret 5; the finger decides it.
        result, _ = self.assign([64], {'index': (2, 5)})
        self.assertEqual((result[0].string, result[0].fret), (2, 5))
        self.assertEqual(result[0].support, 'fingered')
        self.assertEqual(result[0].finger, 'index')

    def test_open_string_chosen_when_no_finger_is_on_it(self):
        result, _ = self.assign([64], {})
        self.assertEqual((result[0].string, result[0].fret), (1, 0))
        self.assertEqual(result[0].support, 'open')

    def test_open_string_is_rejected_when_a_finger_sits_on_that_string(self):
        # A finger on string 1 must stop string 1 being called open.
        occupied = self.assign([64], {'index': (1, 4)})[0][0]
        self.assertNotEqual((occupied.string, occupied.fret), (1, 0))

    def test_two_notes_never_share_a_string(self):
        result, _ = self.assign([pitch_at(2, 5), pitch_at(3, 5)],
                                {'index': (2, 5), 'middle': (3, 5)})
        self.assertEqual({(n.string, n.fret) for n in result}, {(2, 5), (3, 5)})
        self.assertTrue(all(n.support == 'fingered' for n in result))
        self.assertTrue(all(not n.flags for n in result))

    def test_displaced_note_is_flagged_not_silently_respelled(self):
        result, _ = self.assign([64, 65], {})
        self.assertEqual(len({n.string for n in result}), 2)
        self.assertIn('string-taken', [flag for n in result for flag in n.flags])

    def test_note_far_from_the_hand_is_flagged_unsupported(self):
        result, _ = self.assign([pitch_at(1, 17)], {'index': (2, 3)})
        self.assertEqual(result[0].support, 'position-only')
        self.assertIn('unsupported-placement', result[0].flags)

    def test_pitch_outside_the_instrument_is_reported_not_guessed(self):
        result, _ = self.assign([20], {})
        self.assertIsNone(result[0].string)
        self.assertIn('out-of-range', result[0].flags)

    def test_damping_finger_is_reported_as_unexplained(self):
        result, contacts = self.assign([pitch_at(2, 5)], {'index': (2, 5), 'pinky': (5, 7)})
        stray = unexplained_contacts(contacts, result)
        self.assertEqual([c.finger for c in stray], ['pinky'])

    def test_missing_board_and_hand_are_distinguished(self):
        notes = [NoteEvent(0., 1., 64, .8)]
        self.assertEqual(assign_positions(notes, [], board=False)[0].support, 'no-board')
        self.assertEqual(assign_positions(notes, [], board=True, hand=False)[0].support, 'no-hand')

    def test_an_empty_frame_is_not_evidence_of_an_open_string(self):
        """With no fingertips located, an open spelling gets no free credit.

        Pitch 64 is open on string 1 or fret 5 on string 2. Told the hand sits
        at fret 5, the fretted spelling must win; the open string can only win
        on evidence, and a blank frame is not evidence.
        """
        notes = [NoteEvent(0, 1, 64, .8)]
        self.assertEqual(assign_positions(notes, [], position=5.)[0].fret, 5)
        self.assertEqual(assign_positions(notes, [], position=None)[0].fret, 0)

    def test_offboard_fingertips_are_kept_but_marked(self):
        contacts = self.contacts({})
        self.assertTrue(contacts)
        self.assertFalse(any(c.on_board for c in contacts))


class HandSelectionTests(unittest.TestCase):
    def test_fretting_hand_is_the_one_on_the_neck(self):
        observation = board()
        transform = required_transform(observation)
        on_neck = hand_on(transform, {name: (3, 5) for name in FINGERTIPS},
                          parked=transform.board_point(3, 5))
        away = Hand(np.tile(np.array([5., 5.], np.float32), (21, 1)), 'Right', .9)
        self.assertIs(fretting_hand([away, on_neck], observation), on_neck)
        self.assertIsNone(fretting_hand([], observation))
        self.assertIsNone(fretting_hand([away], None))


class TimelineTests(unittest.TestCase):
    def test_chord_onsets_group_and_melody_does_not(self):
        chord = [NoteEvent(1., 2., p, .5) for p in (40, 47, 52)]
        later = [NoteEvent(1.6, 2., 55, .5)]
        groups = group_notes(chord+later)
        self.assertEqual([len(g) for g in groups], [3, 1])

    def test_nearest_record_prefers_a_frame_that_saw_the_board(self):
        records = [FrameRecord(0, .0, 'lost'), FrameRecord(1, .05, 'detected', board()),
                   FrameRecord(2, .10, 'lost')]
        timestamps = [r.timestamp for r in records]
        record = nearest_record(records, timestamps, .0)
        assert record is not None
        self.assertEqual(record.index, 1)
        self.assertIsNone(nearest_record([], [], .0))

    def test_far_onset_does_not_borrow_stale_board_evidence(self):
        records = [FrameRecord(0, .0, 'detected', board())]
        self.assertIsNone(nearest_record(records, [0.], 9.))

    def test_high_fps_search_uses_time_not_fixed_frame_count(self):
        records = [FrameRecord(i, i/120, 'lost') for i in range(40)]
        records[2].observation = board()
        record = nearest_record(records, [r.timestamp for r in records], .1)
        assert record is not None
        self.assertEqual(record.index, 2)

    def test_a_later_onset_repluck_stops_the_note_already_on_that_string(self):
        """E2 and F2 both need the low E, so the E2 must have stopped.

        Blocking the F2 instead would report a note that was plainly played as
        unplayable. GuitarSet has 3 overlapping same-string pairs in 60,670.
        """
        notes = [NoteEvent(0, 1, 40, .8), NoteEvent(.2, .8, 41, .8),
                 NoteEvent(1, 2, 41, .8)]
        assigned, _ = transcribe_notes([], notes, STANDARD_TUNING, 22, .12, False)
        first, overlap, after = assigned.values()
        self.assertEqual([first.string, overlap.string, after.string], [6, 6, 6])
        self.assertEqual(first.end, .2)
        self.assertIn('stopped-by-repluck', first.flags)
        self.assertNotIn('no-free-string', overlap.flags)

    def test_notes_sounded_together_still_cannot_share_a_string(self):
        """A re-pluck ends a ringing note; simultaneous notes are a real clash.

        E2 and F2 both need the low E, so exactly one can be placed. Which one
        wins is a scoring detail; that only one does is the rule.
        """
        notes = [NoteEvent(0, 1, 40, .8), NoteEvent(0, 1, 41, .8)]
        assigned, _ = transcribe_notes([], notes, STANDARD_TUNING, 22, .12, False)
        placed = [n for n in assigned.values() if n.string is not None]
        dropped = [n for n in assigned.values() if n.string is None]
        self.assertEqual([n.string for n in placed], [6])
        self.assertEqual(len(dropped), 1)
        self.assertIn('no-free-string', dropped[0].flags)
        # Nothing re-plucked the string, so the placed note keeps its length.
        self.assertEqual(placed[0].end, 1)
        self.assertNotIn('stopped-by-repluck', placed[0].flags)

    def test_nonoverlapping_notes_within_chord_window_can_reuse_string(self):
        notes = [NoteEvent(0, .02, 40, .8), NoteEvent(.03, .1, 41, .8)]
        assigned, _ = transcribe_notes([], notes, STANDARD_TUNING, 22, .12, False)
        self.assertEqual([n.string for n in assigned.values()], [6, 6])

    def test_duplicate_pitch_onsets_are_not_lost(self):
        notes = [NoteEvent(0, 1, 64, .8), NoteEvent(0, 2, 64, .7)]
        assigned, _ = transcribe_notes([], notes, STANDARD_TUNING, 22, .12, False)
        self.assertEqual(len(assigned), 2)
        self.assertEqual([n.end for n in assigned.values()], [1, 2])

    def test_neck_position_is_planned_from_the_notes_around_a_blind_chord(self):
        """The same pitch is spelled differently by the company it keeps.

        Pitch 64 is open on string 1 or fret 5 on string 2. Surrounded by notes
        that can only be played high on the neck it should take the fretted
        spelling, and surrounded by low ones the open string.
        """
        def spell(context):
            notes = [NoteEvent(t, t + .2, pitch, .8) for t, pitch in context]
            assigned, _ = transcribe_notes([], notes, STANDARD_TUNING, 22, .12, False)
            return next(n for n in assigned.values() if n.pitch == 64)

        high = spell(((0., 68), (.3, 69), (.6, 64), (.9, 70), (1.2, 68)))
        higher = spell(((0., 76), (.3, 77), (.6, 64), (.9, 78), (1.2, 76)))
        self.assertEqual((high.string, high.fret), (2, 5))
        self.assertEqual((higher.string, higher.fret), (4, 14))

    def test_fret_zero_is_judged_on_its_own_terms_not_on_reach(self):
        """An open string needs no hand, so distance from the fingers cannot count.

        E4 is open on string 1 or fret 5 on string 2. With the hand at the nut
        the open string wins; move the hand to fret 5 and the fretted spelling
        does. What must not happen is fret 0 being penalised for sitting far
        from the fingers, which is what made the stage refuse to say "open".
        """
        notes = [NoteEvent(0, 1, 64, .8)]
        self.assertEqual(assign_positions(notes, [], position=0.)[0].fret, 0)
        self.assertEqual(assign_positions(notes, [], position=5.)[0].fret, 5)
        # The spelling tracks the hand up the neck...
        self.assertEqual([assign_positions(notes, [], position=p)[0].fret
                          for p in (10., 14., 18.)], [9, 14, 19])
        # ...until nothing fretted is near enough, and the open string wins again.
        self.assertEqual(assign_positions(notes, [], position=22.)[0].fret, 0)

    def test_a_later_chord_can_decide_an_earlier_one(self):
        """Planning the whole clip lets lookahead place a note, not just memory."""
        notes = [NoteEvent(t, t + .2, pitch, .8)
                 for t, pitch in ((0., 64), (.3, 69), (.6, 70), (.9, 68))]
        assigned, _ = transcribe_notes([], notes, STANDARD_TUNING, 22, .12, False)
        first = list(assigned.values())[0]
        self.assertEqual((first.string, first.fret), (2, 5))

    def test_equal_geometry_scores_favor_louder_note_on_contested_string(self):
        notes = [NoteEvent(0, 1, 41, .2), NoteEvent(0, 1, 42, .9)]
        assigned = assign_positions(notes, [])
        self.assertIsNone(assigned[0].string)
        self.assertEqual(assigned[1].string, 6)


class OverlayTests(unittest.TestCase):
    def test_tab_strip_supports_custom_string_counts_and_tunings(self):
        from processing.music.transcribe import draw_tab_strip
        for tuning in ((40,), (64, 59, 55, 50, 45, 38), (64, 59, 55, 50, 45, 40, 35)):
            frame = np.zeros((360, 640, 3), np.uint8)
            with patch('processing.music.transcribe.draw_text') as text:
                draw_tab_strip(frame, [], [], 0, tuning=tuning)
            self.assertEqual([c.args[1] for c in text.call_args_list[:len(tuning)]],
                             [note_name(p) for p in tuning])
    def test_tab_strip_marks_the_assigned_string_and_fret(self):
        from processing.music.transcribe import draw_tab_strip
        from processing.music.fusion import TabNote
        frame = np.zeros((360, 640, 3), np.uint8)
        note = TabNote(64, 2., 3., .8, string=1, fret=0, support='open', confidence=.7)
        draw_tab_strip(frame, [note], [note], 2.5)
        self.assertTrue((frame[frame.shape[0]-130:] > 0).any())

    def test_mirrored_strings_match_the_unmirrored_assignment(self):
        """Mirroring is a display transform; it must not renumber the strings."""
        from processing.music.transcribe import draw_strings, mirror_points
        transform = required_transform(board())
        plain = np.zeros((360, 640, 3), np.uint8)
        mirrored = np.zeros((360, 640, 3), np.uint8)
        draw_strings(plain, transform, 12, mirror=False)
        draw_strings(mirrored, transform, 12, mirror=True)
        # Anti-aliased rasters will not match pixel for pixel; compare where the
        # ink actually landed.
        drawn_plain = np.argwhere(plain.any(axis=2))
        drawn_mirrored = np.argwhere(mirrored.any(axis=2))
        self.assertTrue(len(drawn_plain) and len(drawn_mirrored))
        # A couple of pixels of slack: strings running past the frame edge are
        # clipped by different amounts on each side.
        self.assertAlmostEqual(drawn_plain[:, 1].mean(),
                               plain.shape[1]-1-drawn_mirrored[:, 1].mean(), delta=2.)
        self.assertAlmostEqual(drawn_plain[:, 0].mean(), drawn_mirrored[:, 0].mean(), delta=2.)
        point = np.array([10., 20.])
        np.testing.assert_allclose(mirror_points(point, 640, True), [629., 20.])
        np.testing.assert_allclose(mirror_points(point, 640, False), point)

    def test_hand_drawing_accepts_contacts_without_crashing(self):
        from processing.music.transcribe import draw_hand
        transform = required_transform(board())
        contacts = finger_contacts(hand_on(transform, {'index': (2, 5)}), transform)
        frame = np.zeros((360, 640, 3), np.uint8)
        draw_hand(frame, np.tile(np.array([100., 100.], np.float32), (21, 1)), contacts)
        self.assertTrue((frame > 0).any())


if __name__ == '__main__':
    unittest.main()
