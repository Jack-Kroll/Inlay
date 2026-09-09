import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from processing.music.pitch import Posteriorgram
from processing.music.fusion import TabNote, FingerContact
from processing.music.rescue import rescue_notes, weak_candidates


class VisualRescueTests(unittest.TestCase):
    def run_rescue(self, contact_pitch=64, existing=(), state='detected', onset=.5,
                   energy=.4, contact_frames=4):
        times = np.arange(100)*.01
        frames = np.zeros((100, 88))
        onsets = frames.copy()
        frames[50:70, 64-21] = energy
        onsets[50, 64-21] = onset
        activations = Posteriorgram(frames, onsets, times)
        records = [SimpleNamespace(timestamp=t, state=state, observation=object(),
                                   hand_points=np.zeros((21, 2)), hand_label='Left')
                   for t in (.45, .48, .51, .54)]
        contact = FingerContact('index', np.zeros(2), contact_pitch-59, 2,
                                contact_pitch-59-.4, 2., True)
        with patch('processing.music.rescue.board_transform', return_value=object()), \
                patch('processing.music.rescue.finger_contacts',
                      side_effect=[[contact] if i < contact_frames else [] for i in range(4)]):
            added, report = rescue_notes(activations, records, existing,
                                         (64,59,55,50,45,40),22,.09,False,1.)
        return added, report, activations

    def test_weak_audio_and_stable_exact_cell_add_flagged_note(self):
        added, report, _ = self.run_rescue()
        self.assertEqual(len(added), 1)
        note = added[0]
        self.assertEqual((note.pitch, note.string, note.fret), (64,2,5))
        self.assertAlmostEqual(note.start,.5)
        self.assertAlmostEqual(note.end,.7)
        self.assertEqual(note.support,'rescued')
        self.assertIn('visual-audio-rescue',note.flags)
        self.assertEqual(report['accepted'],1)

    def test_finger_alone_or_wrong_pitch_or_stale_tracking_cannot_rescue(self):
        for kwargs in ({'onset':.1}, {'energy':.1}, {'contact_pitch':65},
                       {'state':'tracked'}, {'contact_frames':1}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.run_rescue(**kwargs)[0],[])

    def test_no_duplicates_or_displacement_of_baseline_notes(self):
        for pitch in (64,65):
            baseline = TabNote(pitch,.48,.8,.8,string=2,fret=pitch-59)
            added, _, _ = self.run_rescue(existing=[baseline])
            self.assertEqual(added,[])
            self.assertEqual((baseline.start,baseline.end),(.48,.8))

    def test_audio_arrays_are_unchanged(self):
        _, _, activations = self.run_rescue()
        self.assertEqual(np.count_nonzero(activations.note),20)
        self.assertEqual(np.count_nonzero(activations.onset),1)

    def test_short_energy_and_energy_far_from_attack_are_rejected(self):
        for first, last in ((50, 53), (65, 80)):
            times = np.arange(100)*.01
            frames = np.zeros((100,88)); onsets = frames.copy()
            onsets[50,43] = .5
            frames[first:last,43] = .4
            candidates = list(weak_candidates(Posteriorgram(frames,onsets,times),1.))
            self.assertEqual(candidates,[])

    def test_candidates_cannot_extend_past_analyzed_video(self):
        times = np.arange(100)*.01
        frames = np.zeros((100,88)); onsets = frames.copy()
        frames[50:90,43] = .4; onsets[50,43] = .5
        candidates = list(weak_candidates(Posteriorgram(frames,onsets,times),.65))
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0][1],.65)
