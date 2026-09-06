import unittest
from unittest.mock import patch
from processing.vision.preview import open_capture


class CameraSourceTests(unittest.TestCase):
    @patch("processing.vision.preview.sys.platform", "darwin")
    @patch("processing.vision.preview.cv2.VideoCapture")
    def test_denied_camera_reports_permission_steps_and_releases(self, factory):
        factory.return_value.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Privacy & Security > Camera"):
            open_capture(0)
        factory.return_value.release.assert_called_once()
        self.assertEqual(len(factory.call_args.args), 2)

    @patch("processing.vision.preview.sys.platform", "darwin")
    @patch("processing.vision.preview.cv2.VideoCapture")
    def test_video_file_does_not_use_camera_backend(self, factory):
        factory.return_value.isOpened.return_value = True
        self.assertIs(open_capture("clip.mp4"), factory.return_value)
        factory.assert_called_once_with("clip.mp4")

    @patch("processing.vision.preview.cv2.VideoCapture")
    def test_bad_video_reports_file_error(self, factory):
        factory.return_value.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "file path"):
            open_capture("missing.mp4")
        factory.return_value.release.assert_called_once()
