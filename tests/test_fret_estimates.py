import unittest
from unittest.mock import patch
import cv2
import numpy as np
from processing.vision.tracker import (
    Observation, FretboardTracker, assign_numbers, transform_observation,
)
from processing.vision.preview import render_preview


def position(n):
    r = 1-2.**(-n/12)
    return 800*r/(1+.4*r)


def wire(n):
    x = 25+position(n)
    return np.array([[x, 30], [x, 100]], np.float32)


def board(numbers=(1, 2, 4, 5, 8, 12, 15)):
    return Observation(
        np.array([[20, 25], [600, 25], [600, 110], [20, 110]], np.float32),
        [wire(n)[::-1] if i % 2 else wire(n) for i, n in enumerate(numbers)],
        wire(0), .95,
    )


class FretEstimateTests(unittest.TestCase):
    def test_numbering_expires_during_motion_after_recent_board_detection(self):
        tracker = FretboardTracker(max_gap=.5)
        frame = np.zeros((140, 640, 3), np.uint8)
        initial = board(); assign_numbers(initial)
        tracker.update(frame, initial, 0.)
        with patch.object(tracker, '_motion', return_value=initial):
            partial = board(); partial.nut = None; assign_numbers(partial)
            tracker.update(frame, partial, .4)
        self.assertEqual(partial.numbering, 'short-term tracked')
        with patch.object(tracker, '_motion', return_value=partial):
            expired = tracker.update(frame, None, .6)
        self.assertEqual(tracker.state, 'tracked')
        assert expired is not None
        self.assertEqual(expired.numbering, 'unknown')
        self.assertTrue(all(n is None for n in expired.numbers))
        self.assertEqual(expired.estimates, [])

    def test_fills_only_bracketed_gaps_and_keeps_measurements_separate(self):
        observation = board()
        assign_numbers(observation)
        self.assertEqual(observation.numbers, [1, 2, 4, 5, 8, 12, 15])
        self.assertEqual([e.number for e in observation.estimates], [3, 6, 7, 9, 10, 11, 13, 14])
        self.assertEqual(len(observation.frets), 7)
        for estimate in observation.estimates:
            np.testing.assert_allclose(estimate.line[:, 0], wire(estimate.number)[:, 0], atol=.01)
            np.testing.assert_allclose(np.sort(estimate.line[:, 1]), [30, 100], atol=.01)
            self.assertEqual(estimate.method, 'spacing')

    def test_can_fill_first_frets_between_nut_and_fret_five(self):
        observation = board((5, 6, 8, 10, 12, 15))
        assign_numbers(observation)
        self.assertEqual(observation.numbers, [5, 6, 8, 10, 12, 15])
        self.assertTrue({1, 2, 3, 4}.issubset({e.number for e in observation.estimates}))

    def test_wide_gap_is_not_filled(self):
        observation = board((1, 2, 3, 4, 12, 15))
        assign_numbers(observation)
        self.assertFalse(set(range(5, 12)) & {e.number for e in observation.estimates})

    def test_no_nut_or_insufficient_evidence_clears_estimates(self):
        observation = board()
        assign_numbers(observation)
        self.assertTrue(observation.estimates)
        observation.nut = None
        assign_numbers(observation)
        self.assertEqual(observation.estimates, [])
        self.assertEqual(observation.numbering, 'unknown')
        observation = board((1, 3, 5))
        assign_numbers(observation)
        self.assertEqual(observation.estimates, [])

    def test_rotation_and_mirroring_preserve_estimates_and_metadata(self):
        observation = board()
        assign_numbers(observation)
        matrix = np.array([[0., -1., 160.], [1., 0., 5.], [0., 0., 1.]])
        rotated = transform_observation(observation, matrix)
        assign_numbers(rotated)
        self.assertEqual([e.number for e in rotated.estimates], [e.number for e in observation.estimates])
        for original, result in zip(observation.estimates, rotated.estimates):
            expected = cv2.perspectiveTransform(original.line[:, None], matrix)[:, 0]
            np.testing.assert_allclose(result.line, expected, atol=.05)
        mirrored = transform_observation(observation, np.array([[-1., 0., 639.], [0., 1., 0.], [0., 0., 1.]]))
        np.testing.assert_allclose(mirrored.estimates[0].line[:, 0], 639-observation.estimates[0].line[:, 0])
        self.assertIsNot(mirrored.estimates[0].line, observation.estimates[0].line)

    def test_estimates_are_orange_and_measurements_remain_blue(self):
        observation = board()
        assign_numbers(observation)
        frame = np.zeros((140, 640, 3), np.uint8)
        with patch('processing.vision.preview.draw_text'):
            shown = render_preview(frame, observation, 'detected', False)
        x = int(observation.estimates[0].line[0, 0])
        np.testing.assert_array_equal(shown[50, x], [0, 165, 255])
        x = int(observation.frets[0][0, 0])
        np.testing.assert_array_equal(shown[50, x], [255, 220, 0])

    def test_motion_estimates_expire_without_refreshing_detection_time(self):
        tracker = FretboardTracker(max_gap=.5)
        frame = np.zeros((140, 640, 3), np.uint8)
        initial = board(); assign_numbers(initial)
        tracker.update(frame, initial, 0.)
        # Nut and fret 2 disappear, but other detected wires still match.
        with patch.object(tracker, '_motion', return_value=initial):
            partial = board((1, 4, 5, 8, 12, 15)); partial.nut = None
            assign_numbers(partial)
            result = tracker.update(frame, partial, .1)
            assert result is not None
            self.assertEqual(result.numbering, 'short-term tracked')
            self.assertIn(2, [e.number for e in result.estimates])
            self.assertTrue(all(e.method == 'tracked' for e in result.estimates))
            self.assertEqual(tracker.last_detection, 0.)
            expired = board((1, 4, 5, 8, 12, 15)); expired.nut = None
            assign_numbers(expired)
            tracker.update(frame, expired, .55)
            self.assertEqual(expired.estimates, [])


if __name__ == '__main__':
    unittest.main()
