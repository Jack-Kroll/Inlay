"""Stable visual evidence and candidate-recovery safety properties."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from typing import Any
import numpy as np

from processing.music.fusion import FingerContact, Sounded, assign_positions
from processing.music.visual_evidence import VisualContext
from processing.music import verify
from processing.tools.diagnose_missed_notes import recover_scores, diagnose
from processing.music.fusion import TabNote
from processing.tools.eval_confidence_clips import supported_recovery
from processing.music.pitch import Posteriorgram, verified_notes
from unittest.mock import patch


def contacts():
    return [FingerContact('index', np.zeros(2), 5, 1, 4.5, 1.2, True),
            FingerContact('ring', np.zeros(2), 10, 2, 9.5, 2.45, True)]


def samples() -> list[dict[str, Any]]:
    return [dict(timestamp=i/30., contacts=contacts(),
                 quality=dict(residual=.03, edge_residual=.001, anchors=10, state='detected'))
            for i in range(5)]


class VisualPriorityTests(unittest.TestCase):
    def test_separate_raw_peaks_refining_to_one_attack_are_deduplicated(self):
        note = np.full((60, 88), .8, np.float32)
        args = (note, note, np.arange(60)/verify.FPS, np.array([.99, .97]),
                np.array([2., -2.]), np.ones((2, 6))/6, np.array([10, 15]), np.array([29, 29]))
        self.assertEqual(len(verify.decode(*args, suppression_mode='raw')), 2)
        fixed = verify.decode(*args)
        self.assertEqual(len(fixed), 1)
        self.assertAlmostEqual(fixed[0][0], 12/verify.FPS)
        self.assertAlmostEqual(fixed[0][3], .99)

    def test_runtime_recovery_retains_verifier_confidence_and_raw_context(self):
        note, onset = np.zeros((60, 88), np.float32), np.zeros((60, 88), np.float32)
        note[10:20, 29], onset[10, 29] = .6, .9
        original = note.copy()
        activations = Posteriorgram(note, onset, np.arange(60)/verify.FPS,
                                   np.zeros((60, 264), np.float32))
        scored = (np.array([.6]), np.array([0.]), np.ones((1, 6))/6)
        with patch('processing.music.verify.load', return_value=(object(), 5, None)), \
                patch('processing.music.verify.score', return_value=scored):
            self.assertEqual(verified_notes(activations, 'unused.pt'), [])
            recovered = verified_notes(activations, 'unused.pt', recover_attacks=True)
        self.assertEqual(len(recovered), 1)
        self.assertAlmostEqual(recovered[0].amplitude, .6)
        np.testing.assert_array_equal(note, original)

    def test_recovered_candidates_keep_original_score_order_for_suppression(self):
        note = np.full((60, 88), .8, np.float32)
        out = verify.decode(note, note, np.arange(60)/verify.FPS, np.array([.51, .7]),
                            np.zeros(2), np.ones((2, 6))/6,
                            np.array([10, 12]), np.array([29, 29]), accepted=np.ones(2, bool))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 12/verify.FPS)
        self.assertAlmostEqual(out[0][3], .7)

    def test_recovered_note_must_keep_its_confirmed_cell_after_chord_assignment(self):
        placement = TabNote(63, 0., .2, .75, string=3, fret=8, support='position-only',
                            evidence={'visual_priority': {'string': 2, 'fret': 4}})
        self.assertFalse(supported_recovery(placement))
        placement.string, placement.fret, placement.support = 2, 4, 'fingered'
        self.assertTrue(supported_recovery(placement))
    def test_stable_specific_cell_can_outweigh_wrong_audio_string(self):
        proof = VisualContext(samples()).placement(2/30., 69)
        self.assertIsNotNone(proof)
        note = Sounded(69, 2/30., .3, string_posterior=np.array([0., 1., 0, 0, 0, 0]))
        baseline = assign_positions([note], contacts())[0]
        priority = assign_positions([note], contacts(), visual_evidence=[proof])[0]
        self.assertEqual(baseline.string, 2)
        self.assertEqual(priority.string, 1)
        self.assertIn('visual_priority', priority.evidence)
        self.assertEqual((priority.pitch, priority.start, priority.amplitude),
                         (baseline.pitch, baseline.start, baseline.amplitude))

    def test_moving_hand_and_poor_fit_do_not_get_visual_priority(self):
        bad = samples()
        for row in bad:
            row['quality']['residual'] = .08
        self.assertIsNone(VisualContext(bad).placement(2/30., 69))
        moving = samples()
        moving[2]['contacts'][0].string_float = 1.6
        moving[3]['contacts'][0].string_float = 1.7
        self.assertIsNone(VisualContext(moving).placement(2/30., 69))

    def test_stable_hovering_finger_cannot_override_open_string_policy(self):
        self.assertIsNone(VisualContext(samples()).placement(2/30., 64))

    def test_one_sided_or_nonmonotonic_samples_are_rejected(self):
        self.assertIsNone(VisualContext(samples()).placement(0., 69))
        repeated = samples()
        repeated[1]['timestamp'] = repeated[0]['timestamp']
        with self.assertRaisesRegex(ValueError, 'strictly increase'):
            VisualContext(repeated)

    def test_neighbor_proof_cannot_overrule_disagreeing_current_frame(self):
        proof = VisualContext(samples()).placement(2/30., 69)
        current = contacts()[1:]
        note = Sounded(69, .1, .3, string_posterior=np.array([0., 1., 0, 0, 0, 0]))
        placed = assign_positions([note], current, visual_evidence=[proof])[0]
        self.assertEqual(placed.string, 2)
        self.assertNotIn('visual_priority', placed.evidence)

    def test_recovery_uses_audio_evidence_and_preserves_original_scores(self):
        s = dict(scores=np.array([.6, .6, .4, .9]), frame=np.array([0, 1, 2, 3]),
                 pitch=np.zeros(4, np.int64), onset=np.array([[.8], [.3], [.9], [.9]]),
                 note=np.full((4, 1), .9))
        original = deepcopy(s)
        updated = recover_scores(s, 'strong_attack')
        np.testing.assert_array_equal(s['scores'], original['scores'])
        np.testing.assert_array_equal(updated, [.75, .6, .4, .9])

    def test_miss_diagnosis_partitions_threshold_and_refinement_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            frames = np.array([10, 30], np.int64)
            times = np.arange(60)/verify.FPS
            starts = times[frames]
            np.savez(Path(temporary)/'fixture.npz', note=np.full((60, 88), .9),
                     onset=np.full((60, 88), .9), times=times, frame=frames,
                     pitch=np.array([29, 29]), scores=np.array([.7, .9]),
                     deltas=np.array([0., 6.]), strings=np.ones((2, 6))/6,
                     reference=np.array([[t, t+.1, 50., 3.] for t in starts]))
            result = diagnose(Path(temporary))
        self.assertEqual(result['baseline_misses'], 2)
        self.assertEqual(result['miss_causes'], {'verifier_below_threshold': 1,
                                               'refinement_outside_tolerance': 1})
        self.assertEqual(result['variants']['strong_attack']['counts']['matched'], 1)
