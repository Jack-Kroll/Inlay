import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from processing.vision.preview import (
    LatestCameraFrame, display_observation, render_preview, main,
)
from processing.vision.tracker import Observation


def observation():
    return Observation(
        np.array([[2, 3], [14, 3], [14, 12], [2, 12]], np.float32),
        [np.array([[7, 3], [7, 12]], np.float32)],
        np.array([[2, 3], [2, 12]], np.float32),
        .9, [5], 'nut-anchored fit',
    )


class PreviewDisplayTests(unittest.TestCase):
    def test_mirror_reflects_every_geometry_element_without_mutation(self):
        original = observation()
        result = display_observation(original, 32, True)
        np.testing.assert_allclose(result.neck[:, 0], 31-original.neck[:, 0])
        np.testing.assert_allclose(result.frets[0][:, 0], [24, 24])
        assert result.nut is not None
        np.testing.assert_allclose(result.nut[:, 0], [29, 29])
        np.testing.assert_array_equal(original.frets[0][:, 0], [7, 7])
        self.assertEqual(result.numbers, [5])
        self.assertEqual(result.numbering, original.numbering)
        result.numbers[0] = 6
        self.assertEqual(original.numbers, [5])

    def test_camera_is_flipped_before_text_and_text_remains_readable(self):
        frame = np.zeros((20, 32, 3), np.uint8)
        frame[:, 0] = (25, 50, 75)
        with patch('processing.vision.preview.draw_text') as text:
            shown = render_preview(frame, observation(), 'detected', True)
        np.testing.assert_array_equal(shown[0, -1], frame[0, 0])
        self.assertEqual(text.call_args.args[1], '5')
        np.testing.assert_allclose(text.call_args.args[2], [24, 7.5])
        np.testing.assert_array_equal(frame[:, -1], 0)

    def test_no_mirror_keeps_camera_orientation(self):
        frame = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        shown = render_preview(frame, None, 'lost', False)
        np.testing.assert_array_equal(shown, frame)
        self.assertIsNot(shown, frame)

    def test_live_capture_discards_queued_frames_and_releases_once(self):
        capture = MagicMock()
        capture.read.side_effect = [(True, np.full((2, 2, 3), i, np.uint8)) for i in range(5)] + [(False, None)]
        reader = LatestCameraFrame(capture, time.monotonic())
        reader.worker.join(timeout=1)
        self.assertFalse(reader.worker.is_alive())
        packet = reader.read()
        assert packet is not None
        index, timestamp, frame = packet
        self.assertEqual(index, 4)
        self.assertGreaterEqual(timestamp, 0)
        np.testing.assert_array_equal(frame, 4)
        self.assertIsNone(reader.read())
        reader.close()
        capture.release.assert_called_once()

    def test_capture_errors_are_reported_and_release_the_camera(self):
        capture = MagicMock()
        capture.read.side_effect = RuntimeError('device failure')
        reader = LatestCameraFrame(capture, time.monotonic())
        reader.worker.join(timeout=1)
        with self.assertRaisesRegex(RuntimeError, 'Camera capture failed'):
            reader.read()
        reader.close()
        capture.release.assert_called_once()

    def test_video_keeps_frames_timestamps_and_unmirrored_json(self):
        frames = [np.zeros((20, 32, 3), np.uint8) for _ in range(3)]
        capture = MagicMock()
        capture.get.return_value = 10.
        capture.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'preview.jsonl'
            with patch('sys.argv', ['preview', '--model', 'unused.pt', '--source', 'clip.avi', '--device', 'cpu', '--headless', '--jsonl', str(output)]), \
                    patch('processing.vision.preview.open_capture', return_value=capture), \
                    patch('processing.vision.model.DenseDetector') as detector, \
                    patch('processing.vision.preview.FretboardTracker') as tracker, \
                    patch('processing.vision.preview.decode_maps', return_value=observation()), \
                    patch('processing.vision.preview.LatestCameraFrame') as camera:
                detector.return_value.predict.return_value = np.zeros((20, 32, 3), np.float32)
                tracker.return_value.update.side_effect = lambda frame, obs, timestamp: obs
                tracker.return_value.state = 'detected'
                tracker.return_value.flow_inliers = 0
                main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([r['timestamp'] for r in rows], [0., .1, .2])
        self.assertEqual([r['capture_frame'] for r in rows], [0, 1, 2])
        self.assertEqual(rows[0]['frets'][0]['endpoints'], [[7., 3.], [7., 12.]])
        camera.assert_not_called()
        capture.release.assert_called_once()

