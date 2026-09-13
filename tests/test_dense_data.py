import unittest
import numpy as np
from processing.training.data import render_targets, letterbox


class DenseDataTests(unittest.TestCase):
    def test_wire_on_image_boundary_is_supervised_inside_image(self):
        record = {'supervised': ['fret'], 'annotations': [
            {'head': 'fret', 'kind': 'line', 'points': [[1, 0], [1, 1]]}]}
        target, valid = render_targets(record, (100, 50, 3), 128)
        self.assertEqual(target[:, 95, 1].sum(), 128)
        self.assertEqual((target * (1-valid)).sum(), 0)

    def test_letterbox_extreme_aspect_ratio_keeps_a_pixel(self):
        image, box = letterbox(np.zeros((1, 1000, 3), np.uint8), 64)
        self.assertEqual(image.shape, (64, 64, 3))
        self.assertEqual(box[3], 1)

    def test_letterbox_and_missing_head_mask(self):
        image=np.zeros((50,100,3),np.uint8)
        output,box=letterbox(image,128)
        self.assertEqual(box,(0,32,128,64))
        r={'supervised':['neck'],'annotations':[{'head':'neck','kind':'polygon','points':[[0,0],[1,0],[1,1],[0,1]]}]}
        target,valid=render_targets(r,image.shape,128)
        self.assertEqual(valid[...,1:].sum(),0)
        self.assertEqual(valid[:32].sum(),0)
        self.assertGreater(target[...,0].sum(),0)
