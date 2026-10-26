"""Regression checks for defects found during the visual accuracy audit."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np

from processing.music.fusion import Sounded, TabNote, FingerContact, plan_positions, assign_positions
from processing.music import verify
from processing.music.transcribe import FrameRecord, nearest_record
from processing.music.pitch import notes_from_media
from processing.tools.eval_verifier import score_notes
from processing.tools.guitarset import Note
from processing.tools.eval_visual_contacts import entries, stabilize


class AccuracyRegressions(unittest.TestCase):
    def test_verifier_obeys_the_same_exclusive_frequency_ceiling_as_basic_pitch(self):
        with patch('processing.music.pitch.raw_activations', return_value=object()), \
                patch('processing.music.pitch.verified_notes', return_value=[]) as decode:
            notes_from_media('unused.wav', verifier='unused.pt',
                             maximum_frequency=440 * 2**((87 - 69)/12))
        self.assertEqual(decode.call_args.kwargs['maximum_pitch'], 86)

    def test_nearby_hand_recovers_a_single_frame_dropout(self):
        records = [FrameRecord(0, 1., 'detected', object()),
                   FrameRecord(1, 1.03, 'detected', object(), np.zeros((21, 2)))]
        self.assertIs(nearest_record(records, [r.timestamp for r in records], 1.), records[1])
        records[1].timestamp = 1.1
        self.assertIs(nearest_record(records, [r.timestamp for r in records], 1.), records[0])
        records[0].hand_points = np.ones((21, 2))
        records[1].timestamp = 1.03
        self.assertIs(nearest_record(records, [r.timestamp for r in records], 1.), records[0])

    def test_bad_anchor_does_not_erase_valid_visual_context(self):
        groups = [[Sounded(64, i, i + .2)] for i in range(3)]
        for bad in (-1., 23., float('nan'), float('inf')):
            expected = plan_positions(groups, [12., None, None])
            self.assertEqual(plan_positions(groups, [12., bad, None]), expected)

    def test_position_planner_requires_one_observation_per_group(self):
        with self.assertRaisesRegex(ValueError, 'one observed'):
            plan_positions([[Sounded(64, 0., 1.)]], [])

    def test_evaluator_preserves_unison_identity_and_input_order(self):
        # Input is out of order and has two distinct same-onset E4 notes.
        notes = [(1., 2., 65, None), (0., .5, 64, None), (0., .5, 64, None)]
        reference = [Note(1., 2., 65, 5), Note(0., .5, 64, 5),
                     Note(0., .5, 64, 4)]
        assigned = {0: TabNote(64, 0., .5, 1., string=1, fret=0),
                    1: TabNote(64, 0., .5, 1., string=2, fret=5),
                    2: TabNote(65, 1., 2., 1., string=1, fret=1)}
        with patch('processing.tools.eval_verifier.transcribe_notes',
                   return_value=(assigned, [])), \
             patch('mir_eval.transcription.match_notes',
                   return_value=[(0, 0), (1, 1), (2, 2)]):
            counts = score_notes(notes, reference)
        self.assertEqual(counts[3], 3)

    def test_fret_only_evidence_does_not_claim_fingertip_support(self):
        contact = FingerContact('index', np.zeros(2), 5, 5, 4.5, 5., True)
        # E4 at B-string fret 5 fits the fret, but the measured string is far away.
        note = assign_positions([Sounded(64, 0., 1.)], [contact], fret_weight=1.)[0]
        self.assertEqual((note.string, note.fret), (2, 5))
        self.assertNotEqual(note.support, 'fingered')
        self.assertIsNone(note.finger)

    def test_fret_weight_cannot_change_audio_only_placements(self):
        notes = [Sounded(64, 0., 1., string_posterior=np.array([.1, .8, .1, 0, 0, 0]))]
        for position in (None, 0., 5., 12.):
            a = assign_positions(notes, [], position=position, fret_weight=0.)[0]
            b = assign_positions(notes, [], position=position, fret_weight=1.)[0]
            self.assertEqual((a.string, a.fret, a.confidence), (b.string, b.fret, b.confidence))

    def test_explicit_suppression_preserves_fast_repeat_when_requested(self):
        note = np.full((50, 88), .8, np.float32)
        args = (note, note, np.arange(50)/verify.FPS, np.array([.9, .8]),
                np.zeros(2), np.ones((2, 6))/6, np.array([10, 13]), np.array([40, 40]))
        self.assertEqual(len(verify.decode(*args)), 1)
        self.assertEqual(len(verify.decode(*args, suppression_seconds=.025)), 2)

    def test_dataset_join_keeps_frame_ids_when_an_image_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'tablature_labels').mkdir()
            (root / 'tablature_frames').mkdir()
            labels = np.zeros((3, 4, 3), np.int32)
            labels[2, 0] = [1, 8, 3]
            np.save(root / 'tablature_labels/0.npy', labels)
            for index in (0, 2):
                (root / f'tablature_frames/0_{index}.png').touch()
            (root / 'timestamps.csv').write_text(
                'frame,audio,timestamp\n0_0.png,0.wav,0\n0_1.png,0.wav,.1\n0_2.png,0.wav,.2\n')
            rows, audit = entries(root)
            self.assertEqual(rows[1]['frame'], '0_2.png')
            self.assertEqual(rows[1]['label'][0], [1, 8, 3])
            self.assertEqual(audit['missing_images'], ['0_1.png'])

    def test_temporal_pooling_never_bridges_a_shift_or_recording(self):
        def row(t, fret, recording=0):
            c = dict(finger='index', point=[0., 0.], fret=round(fret), string=3,
                     fret_float=fret, string_float=3., on_board=True)
            return dict(timestamp=t, recording=recording, contacts=[c] if fret else [])
        stable = [row(0., 4.5), row(.1, 0), row(.2, 4.6)]
        self.assertEqual(len(stabilize(stable)[1]['contacts']), 1)
        self.assertFalse(stable[1]['contacts'])  # Input cache remains immutable.
        shifted = [row(0., 4.5), row(.1, 0), row(.2, 7.5)]
        self.assertFalse(stabilize(shifted)[1]['contacts'])
        boundary = [row(0., 4.5), row(.1, 0), row(.2, 4.5, recording=1)]
        self.assertFalse(stabilize(boundary)[1]['contacts'])
