import unittest
from unittest.mock import patch
import cv2
import numpy as np
from processing.vision.tracker import Observation, FretboardTracker, decode_maps, lattice_fit


def board():
    return Observation(np.array([[10,20],[210,20],[210,60],[10,60]],np.float32),
                       [np.array([[x,20],[x,60]],np.float32) for x in (25,45,65,85)],
                       None,.9,[None]*4)


class DenseTrackerTests(unittest.TestCase):
    def test_lattice_missing_nonconsecutive_frets(self):
        numbers=np.array([1,2,4,5,8,12,15])
        r=1-2.**(-numbers/12);distances=800*r/(1+.4*r)
        fit=lattice_fit(distances)
        assert fit is not None
        self.assertEqual(fit['numbers'],numbers.tolist())

    def test_lattice_rejects_insufficient_evidence(self):
        self.assertIsNone(lattice_fit([10,20,30]))
        self.assertIsNone(lattice_fit([10,float('nan'),30,40]))

    def test_decode_board_without_nut_or_markers(self):
        maps=np.zeros((160,320,3),np.float32)
        maps[50:101,20:300,0]=.95
        for x in (40,80,120,160,200,240,280):maps[50:101,x-1:x+2,1]=.95
        observation=decode_maps(maps)
        assert observation is not None
        self.assertGreaterEqual(len(observation.frets),6)
        self.assertIsNone(observation.nut)
        self.assertEqual(observation.numbering,'unknown')
        rotated=decode_maps(np.rot90(maps).copy())
        assert rotated is not None
        self.assertGreaterEqual(len(rotated.frets),6)

    def test_short_gap_requires_motion_and_expires(self):
        tracker=FretboardTracker(max_gap=.5);frame=np.zeros((100,250,3),np.uint8)
        tracker.update(frame,board(),0)
        with patch.object(tracker,'_motion',return_value=board()):
            self.assertIsNotNone(tracker.update(frame,None,.1))
            self.assertEqual(tracker.state,'tracked')
            self.assertIsNotNone(tracker.update(frame,None,.4))
            self.assertIsNone(tracker.update(frame,None,.6))
            self.assertEqual(tracker.state,'lost')

    def test_no_motion_support_does_not_freeze_overlay(self):
        tracker=FretboardTracker();frame=np.zeros((100,250,3),np.uint8)
        tracker.update(frame,board(),0)
        self.assertIsNone(tracker.update(frame,None,.1))

    def test_timestamp_rewind_and_frame_resize_reset_history(self):
        tracker=FretboardTracker();frame=np.zeros((100,250,3),np.uint8)
        tracker.update(frame,board(),1)
        self.assertIsNone(tracker.update(frame,None,.5))
        tracker.update(frame,board(),2)
        self.assertIsNone(tracker.update(frame[:50],None,2.1))

    def test_tightly_spaced_wires_do_not_become_longitudinal_lines(self):
        maps=np.zeros((150,320,3),np.float32);maps[30:111,20:301,0]=1
        xs=list(range(40,281,6))
        for x in xs:maps[30:111,x,1]=1
        observation=decode_maps(maps)
        assert observation is not None
        self.assertEqual(len(observation.frets),len(xs))
        centers=sorted(float(line[:,0].mean()) for line in observation.frets)
        np.testing.assert_allclose(centers,xs,atol=1)
