import unittest
import numpy as np
from processing.vision.geometry import (
    fret_ratio, invert_projective, fit_from_assigned, obb_to_segment, select_anchors,
)


class GeometryTests(unittest.TestCase):
    def test_octave_frets(self):
        self.assertAlmostEqual(fret_ratio(0), 0)
        self.assertAlmostEqual(fret_ratio(12), 0.5)
        self.assertAlmostEqual(fret_ratio(24), 0.75)

    def test_recovers_perspective_with_missing_frets(self):
        a, c = 900.0, 0.6
        anchors = [{"s": a * fret_ratio(n) / (1 + c * fret_ratio(n)),
                    "conf": 0.9, "fret_num": n} for n in (1, 3, 7, 12)]
        fit = fit_from_assigned(anchors, 24)
        assert fit is not None
        self.assertAlmostEqual(fit["a"], a, places=6)
        self.assertAlmostEqual(fit["c"], c, places=6)
        for anchor in anchors:
            self.assertAlmostEqual(invert_projective(anchor["s"], fit["a"], fit["c"]), anchor["fret_num"])

    def test_rejects_unusable_projection(self):
        self.assertIsNone(invert_projective(0, 900, 0))
        self.assertIsNone(invert_projective(1000, 900, 0))
        self.assertIsNone(fit_from_assigned([], 24))
        self.assertIsNone(fit_from_assigned([
            {"s": 50, "conf": .9, "fret_num": 1},
            {"s": 50, "conf": .9, "fret_num": 1},
        ], 24))

    def test_segment_independent_of_corner_order(self):
        corners = np.array([[10, 20], [70, 20], [70, 24], [10, 24]])
        ref = obb_to_segment(corners)
        for order in (corners, corners[::-1]):
            for shift in range(4):
                segment = obb_to_segment(np.roll(order, shift, axis=0))
                np.testing.assert_allclose(segment["p1"], ref["p1"])
                np.testing.assert_allclose(segment["p2"], ref["p2"])

    def test_anchor_filtering_and_duplicate_confidence(self):
        segments = [{"center": np.array([s, 0]), "conf": conf}
                    for s, conf in [(-10, .9), (20, .4), (22, .9), (60, .1), (90, .8)]]
        anchors = select_anchors(segments, np.zeros(2), np.array([1, 0]), .3, 5, 10)
        self.assertEqual([a["s"] for a in anchors], [22, 90])
