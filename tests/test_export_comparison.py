import unittest
import numpy as np
from processing.tools.verify_coreml import compare


class ExportComparisonTests(unittest.TestCase):
    def setUp(self):
        self.box = np.array([[0, 0], [20, 0], [20, 2], [0, 2]])

    def test_corner_order_does_not_change_comparison(self):
        self.assertEqual(compare([(0, .9, self.box)], [(0, .9, np.roll(self.box[::-1], 2, axis=0))]), 0)

    def test_rejects_shifted_box(self):
        with self.assertRaises(AssertionError):
            compare([(0, .9, self.box)], [(0, .9, self.box + 10)])

    def test_rejects_missing_and_empty_detections(self):
        for ref, out in [([], []), ([(0, .9, self.box)], [])]:
            with self.assertRaises(AssertionError):
                compare(ref, out)
