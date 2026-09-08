import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from processing.training.data import read_manifest, render_targets, letterbox
from processing.training.prepare import prepare


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

    def test_session_holdouts_preserve_original_image_sampling_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'train/images').mkdir(parents=True)
            (root/'train/labels').mkdir()
            (root/'data.yaml').write_text('names: [neck]\n')
            rng = np.random.default_rng(2)
            names = ('a.rf.copy1', 'a.rf.copy2', 'b.rf.copy1')
            for name in names:
                cv2.imwrite(str(root/f'train/images/{name}.png'),
                            rng.integers(0, 255, (32, 64, 3), dtype=np.uint8))
                (root/f'train/labels/{name}.txt').write_text('')
            spec = {'name': 'fixture', 'root': str(root), 'format': 'obb',
                    'mapping': {'neck': 'neck'}, 'supervised': ['neck'],
                    'attribution': 'test', 'license': 'test'}
            prepare([spec], root/'manifest.json',
                    groups={'fixture:a': 'session1', 'fixture:b': 'session1'})
            records = read_manifest(root/'manifest.json')['records']
            self.assertEqual(len({r['group'] for r in records}), 1)
            self.assertEqual([r['original_group'] for r in records],
                             ['fixture:a', 'fixture:a', 'fixture:b'])

    def test_letterbox_and_missing_head_mask(self):
        image=np.zeros((50,100,3),np.uint8)
        output,box=letterbox(image,128)
        self.assertEqual(box,(0,32,128,64))
        r={'supervised':['neck'],'annotations':[{'head':'neck','kind':'polygon','points':[[0,0],[1,0],[1,1],[0,1]]}]}
        target,valid=render_targets(r,image.shape,128)
        self.assertEqual(valid[...,1:].sum(),0)
        self.assertEqual(valid[:32].sum(),0)
        self.assertGreater(target[...,0].sum(),0)

    def test_merge_preserves_holdout_and_masks_unknown_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'data.yaml').write_text('names: [fret, neck, nut]\n')
            pixels=np.random.default_rng(0).integers(0,255,(32,64,3),dtype=np.uint8)
            for split in ('train','valid','test'):
                (source/split/'images').mkdir(parents=True);(source/split/'labels').mkdir()
                cv2.imwrite(str(source/split/'images'/f'{split}.png'),pixels)
                (source/split/'labels'/f'{split}.txt').write_text('1 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n')
            spec={'name':'fixture','root':str(source),'format':'obb','mapping':{'neck':'neck'},'supervised':['neck'],'attribution':'test','license':'test'}
            report=prepare([spec],root/'manifest.json')
            doc=read_manifest(root/'manifest.json')
            self.assertEqual(report['quarantined_images'],2)
            self.assertEqual([r['split'] for r in doc['records']],['test'])
            self.assertEqual(doc['records'][0]['supervised'],['neck'])
            leaked=dict(doc['records'][0]);leaked['split']='train';doc['records'].append(leaked)
            (root/'manifest.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'leakage'):read_manifest(root/'manifest.json')

    def test_obb_clipping_does_not_turn_wire_into_wrong_diagonal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train/images').mkdir(parents=True);(root/'train/labels').mkdir()
            (root/'data.yaml').write_text('names: [fret, neck, nut]\n')
            cv2.imwrite(str(root/'train/images/a.png'),np.zeros((30,30,3),np.uint8))
            (root/'train/labels/a.txt').write_text('0 0.4 -0.1 0.42 -0.1 0.42 0.8 0.4 0.8\n')
            spec={'name':'fixture','root':str(root),'format':'obb','mapping':{'fret':'fret'},'supervised':['neck','fret'],'mask_missing_neck':True,'attribution':'test','license':'test'}
            prepare([spec],root/'manifest.json')
            annotation=read_manifest(root/'manifest.json')['records'][0]['annotations'][0]
            np.testing.assert_allclose(annotation['points'],[[.41,0],[.41,.8]],atol=1e-5)
            self.assertTrue(annotation['clipped'])
            self.assertEqual(read_manifest(root/'manifest.json')['records'][0]['supervised'], ['fret'])
