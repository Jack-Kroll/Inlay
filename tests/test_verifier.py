"""The note verifier: peak-picking, patch geometry, decoding, and the fusion
term the string head feeds. No trained checkpoint and no dataset on disk."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from processing.music import fusion, verify
from processing.music.fusion import Sounded, AUDIO_STRING_WEIGHT, assign_positions
from processing.music.tuning import STANDARD_TUNING
from processing.tools import idmt


def blank(frames=60, pitches=88):
    return np.zeros((frames, pitches), np.float32)


class Peaks(unittest.TestCase):
    def test_finds_a_local_maximum_not_a_crossing(self):
        onset = blank()
        onset[20:25, 30] = [.2, .5, .9, .5, .2]
        found = verify.peaks(onset, threshold=.15)
        # One attack, at the top of the rise, not one per frame over the line.
        self.assertEqual(list(zip(*np.nonzero(found))), [(22, 30)])

    def test_a_plateau_yields_its_first_frame(self):
        onset = blank()
        onset[10:14, 5] = .8
        found = verify.peaks(onset, threshold=.15)
        self.assertEqual(list(zip(*np.nonzero(found))), [(10, 5)])

    def test_below_threshold_is_not_an_attack(self):
        onset = blank()
        onset[20:23, 30] = [.05, .1, .05]
        self.assertFalse(verify.peaks(onset, threshold=.15).any())

    def test_two_attacks_on_one_pitch_are_both_found(self):
        onset = blank()
        onset[10:13, 7] = [.3, .9, .3]
        onset[30:33, 7] = [.3, .7, .3]
        found = np.nonzero(verify.peaks(onset, threshold=.15))
        self.assertEqual(list(found[0]), [11, 31])


class Patches(unittest.TestCase):
    def test_patch_has_the_declared_shape_and_channels(self):
        note, onset = blank(), blank()
        contour = np.zeros((60, 264), np.float32)
        patch = verify.cut(verify.prepare(note, onset, contour), [30], [40],
                           verify.CHANNELS)
        self.assertEqual(patch.shape, (1, verify.CHANNELS, verify.FRAME_SPAN,
                                       verify.PITCH_SPAN))

    def test_a_patch_at_the_edge_is_zero_padded_not_clipped(self):
        note, onset = blank(), blank()
        note[0, 0] = 1.
        contour = np.zeros((60, 264), np.float32)
        patch = verify.cut(verify.prepare(note, onset, contour), [0], [0],
                           verify.CHANNELS)
        self.assertEqual(patch.shape[2:], (verify.FRAME_SPAN, verify.PITCH_SPAN))
        # The candidate sits at the centre of its own patch.
        self.assertEqual(patch[0, 0, verify.FRAME_RADIUS, verify.PITCH_RADIUS], 1.)
        self.assertEqual(patch[0, 0, 0, 0], 0.)

    def test_cqt_channels_are_stacked_on(self):
        note, onset = blank(), blank()
        contour = np.zeros((60, 264), np.float32)
        spectrum = np.zeros((60, 88, 3), np.float32)
        prepared = verify.prepare(note, onset, contour, spectrum)
        patch = verify.cut(prepared, [30], [40], verify.CHANNELS + verify.CQT_CHANNELS)
        self.assertEqual(patch.shape[1], verify.CHANNELS + verify.CQT_CHANNELS)

    def test_cqt_is_aligned_by_time_not_by_frame_number(self):
        # Basic Pitch's frame times drift against a plain hop grid, so a CQT
        # row must be chosen by when it happened.
        spectrum = np.arange(300*264, dtype=np.float32).reshape(300, 264)
        times = np.array([0., 1., 2.])
        aligned = verify.align_cqt(spectrum, times, 88)
        self.assertEqual(aligned.shape, (3, 88, 3))
        self.assertTrue(np.array_equal(aligned[0, 0], spectrum[0, :3]))
        # One second in is round(1 * 86.13) = 86 frames, not frame 1.
        self.assertTrue(np.array_equal(aligned[1, 0], spectrum[86, :3]))

    def test_alignment_clamps_past_the_end_of_the_spectrum(self):
        spectrum = np.ones((10, 264), np.float32)
        aligned = verify.align_cqt(spectrum, np.array([99.]), 88)
        self.assertEqual(aligned.shape, (1, 88, 3))


class Suppression(unittest.TestCase):
    def test_the_stronger_of_two_close_attacks_wins(self):
        frame = np.array([10, 12]); pitch = np.array([5, 5])
        score = np.array([.6, .9], np.float32)
        kept = verify.suppress(frame, pitch, score, window=4)
        self.assertEqual(list(kept), [1])

    def test_attacks_far_apart_both_survive(self):
        frame = np.array([10, 40]); pitch = np.array([5, 5])
        score = np.array([.6, .9], np.float32)
        self.assertEqual(sorted(verify.suppress(frame, pitch, score, 4)), [0, 1])

    def test_different_pitches_do_not_suppress_each_other(self):
        frame = np.array([10, 11]); pitch = np.array([5, 6])
        score = np.array([.6, .9], np.float32)
        self.assertEqual(sorted(verify.suppress(frame, pitch, score, 4)), [0, 1])


class Decode(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(60)/verify.FPS

    def decode(self, note, exists, deltas, frame, pitch, **kwargs):
        strings = np.tile(np.eye(6, dtype=np.float32)[0], (len(frame), 1))
        return verify.decode(note, blank(), self.times, exists, deltas, strings,
                             frame, pitch, **kwargs)

    def test_a_score_below_threshold_produces_no_note(self):
        exists = np.array([.5], np.float32)
        out = self.decode(blank(), exists, np.zeros(1, np.float32),
                          np.array([10]), np.array([30]), threshold=.85)
        self.assertEqual(out, [])

    def test_offset_follows_the_frame_head_down(self):
        note = blank()
        note[10:20, 30] = .9
        out = self.decode(note, np.array([.99], np.float32), np.zeros(1, np.float32),
                          np.array([10]), np.array([30]), threshold=.85)
        self.assertEqual(len(out), 1)
        start, end, pitch, confidence, posterior = out[0]
        self.assertEqual(pitch, 30 + verify.LOWEST_BIN)
        self.assertAlmostEqual(confidence, .99, places=5)
        self.assertEqual(len(posterior), 6)
        # The note ends where the frame head falls away, near frame 20.
        self.assertGreater(end - start, 8/verify.FPS)
        self.assertLess(end - start, 12/verify.FPS)

    def test_a_repluck_ends_the_note_already_on_that_pitch(self):
        note = blank()
        note[10:40, 30] = .9
        exists = np.array([.99, .99], np.float32)
        out = self.decode(note, exists, np.zeros(2, np.float32),
                          np.array([10, 25]), np.array([30, 30]), threshold=.85)
        self.assertEqual(len(out), 2)
        # The first note stops at the second attack rather than ringing through it.
        self.assertLessEqual(out[0][1], out[1][0] + 1e-6)

    def test_refinement_moves_the_onset_by_the_predicted_offset(self):
        note = blank()
        note[10:20, 30] = .9
        deltas = np.array([2.], np.float32)
        refined = self.decode(note, np.array([.99], np.float32), deltas,
                              np.array([10]), np.array([30]), threshold=.85)
        raw = self.decode(note, np.array([.99], np.float32), deltas,
                          np.array([10]), np.array([30]), threshold=.85, refine=False)
        self.assertAlmostEqual(refined[0][0] - raw[0][0], 2/verify.FPS, places=5)

    def test_an_onset_is_never_refined_before_zero(self):
        note = blank()
        note[0:5, 30] = .9
        out = self.decode(note, np.array([.99], np.float32),
                          np.array([-50.], np.float32),
                          np.array([0]), np.array([30]), threshold=.85)
        self.assertGreaterEqual(out[0][0], 0.)


class AudioStringTerm(unittest.TestCase):
    def test_a_confident_posterior_moves_the_choice(self):
        # E4 is open on string 1 and fret 5 on string 2. With no hand and no
        # board, geometry cannot separate them; audio can.
        for index, expected in ((0, 1), (1, 2)):
            posterior = np.full(6, .01, np.float32)
            posterior[index] = .95
            notes = [Sounded(64, 0., .5, string_posterior=posterior)]
            with mock.patch.object(fusion, 'OPEN_AUDIO_NEUTRAL', 0.):
                placed = assign_positions(notes, [], STANDARD_TUNING, 22,
                                          audio_weight=.9)
            self.assertEqual(placed[0].string, expected)

    def test_audio_cannot_argue_a_note_off_an_open_string(self):
        # E4: open on string 1, or fret 5 on string 2. Audio is sure of string 2.
        posterior = np.full(6, .01, np.float32)
        posterior[1] = .95
        notes = [Sounded(64, 0., .5, string_posterior=posterior)]
        with mock.patch.object(fusion, 'OPEN_AUDIO_NEUTRAL', 0.):
            moved = assign_positions(notes, [], STANDARD_TUNING, 22, audio_weight=.25)
        with mock.patch.object(fusion, 'OPEN_AUDIO_NEUTRAL', 1.):
            kept = assign_positions(notes, [], STANDARD_TUNING, 22, audio_weight=.25)
            # It can still choose between two fretted spellings: B3 at 2/0 is
            # open, so ask about G#3, 3/1 or 4/6.
            posterior = np.full(6, .01, np.float32)
            posterior[3] = .95
            fretted = assign_positions([Sounded(56, 0., .5, string_posterior=posterior)],
                                       [], STANDARD_TUNING, 22, audio_weight=.25)
        without = assign_positions([Sounded(64, 0., .5)], [], STANDARD_TUNING, 22,
                                   audio_weight=0.)
        self.assertEqual((moved[0].string, moved[0].fret), (2, 5))
        self.assertEqual((kept[0].string, kept[0].fret),
                         (without[0].string, without[0].fret))
        self.assertEqual(fretted[0].string, 4)

    def test_the_open_rule_does_not_favour_open_either(self):
        # Neutral means open-versus-fretted differs by geometry alone, scaled
        # as the term scales it -- not open whenever audio is unsure, which is
        # what crediting the open spelling with the posterior's maximum did.
        posterior = np.array([.10, .30, .25, .15, .10, .10], np.float32)
        contacts = [fusion.FingerContact('index', np.zeros(2), 5, 2, 4.5, 2., True)]
        def gap(weight, rival):
            with mock.patch.object(fusion, 'OPEN_AUDIO_NEUTRAL', 1.):
                opened = fusion._score(1, 0, contacts, 5., posterior, weight, rival)[0]
                fretted = fusion._score(2, 5, contacts, 5., posterior, weight, rival)[0]
            return opened - fretted
        self.assertAlmostEqual(gap(.25, .30), .75*gap(0., .30), places=6)
        self.assertGreater(gap(.25, float(posterior.max()) + .2), .75*gap(0., .30))

    def test_zero_weight_ignores_the_posterior(self):
        loud = np.zeros(6, np.float32); loud[3] = 1.
        notes = [Sounded(64, 0., .5, string_posterior=loud)]
        with_audio = assign_positions(notes, [], STANDARD_TUNING, 22, audio_weight=0.)
        without = assign_positions([Sounded(64, 0., .5)], [], STANDARD_TUNING, 22)
        self.assertEqual(with_audio[0].string, without[0].string)

    def test_a_note_with_no_posterior_still_places(self):
        placed = assign_positions([Sounded(64, 0., .5)], [], STANDARD_TUNING, 22,
                                  audio_weight=AUDIO_STRING_WEIGHT)
        self.assertIsNotNone(placed[0].string)

    def test_the_posterior_is_recorded_as_evidence(self):
        posterior = np.full(6, .1, np.float32); posterior[2] = .5
        placed = assign_positions([Sounded(64, 0., .5, string_posterior=posterior)], [],
                                  STANDARD_TUNING, 22, audio_weight=.25)
        self.assertIn('audio_string', placed[0].evidence)
        self.assertEqual(len(placed[0].evidence['audio_string']), 6)

    def test_one_confident_note_does_not_respell_the_chord_around_it(self):
        """A wrong-but-confident note has to outscore the whole onset.

        Both notes can take string 1 or string 2. Note by note the F4 wins
        string 1 outright and the E4 is left with what remains; between them
        the F4 loses almost nothing by moving and the E4 loses most of its
        confidence, so the pair is better off the other way round.
        """
        loud = np.zeros(6, np.float32); loud[0], loud[1] = .6, .4
        quiet = np.zeros(6, np.float32); quiet[0], quiet[1] = .55, .05
        notes = [Sounded(65, 0., .5, string_posterior=loud), Sounded(64, 0., .5, string_posterior=quiet)]
        # E4 on string 1 is open; this is about the onset, not the open rule.
        with mock.patch.object(fusion, 'OPEN_AUDIO_NEUTRAL', 0.):
            placed = assign_positions(notes, [], STANDARD_TUNING, 22, audio_weight=.9)
        self.assertEqual([note.string for note in placed], [2, 1])
        self.assertIn('string-taken', placed[0].flags)

    def test_two_notes_still_cannot_share_a_string(self):
        # Both posteriors insist on the same string; one note has to move.
        posterior = np.zeros(6, np.float32); posterior[0] = 1.
        notes = [Sounded(64, 0., .5, string_posterior=posterior.copy()),
                 Sounded(59, 0., .5, string_posterior=posterior.copy())]
        placed = assign_positions(notes, [], STANDARD_TUNING, 22, audio_weight=.9)
        strings = [note.string for note in placed if note.string is not None]
        self.assertEqual(len(strings), len(set(strings)))


XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<instrumentRecording>
  <globalParameter>
    <audioFileName>\\\\take.wav</audioFileName>
    <instrumentModel>Test Guitar</instrumentModel>
    <instrumentTuning>{tuning}</instrumentTuning>
  </globalParameter>
  <transcription>
    <event><onsetSec>1.0</onsetSec><pitch>45</pitch><offsetSec>2.0</offsetSec>
      <fretNumber>0</fretNumber><stringNumber>2</stringNumber></event>
    <event><onsetSec>3.0</onsetSec><pitch>52</pitch><offsetSec>3.0</offsetSec>
      <fretNumber>2</fretNumber><stringNumber>4</stringNumber></event>
  </transcription>
</instrumentRecording>
"""


class Idmt(unittest.TestCase):
    def test_strings_are_renumbered_from_the_low_e(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'take.xml'
            path.write_text(XML.format(tuning='40  45  50  55  59  64'))
            notes = idmt.load_notes(path)
        # IDMT's string 2 is the A string, which is index 1 counting from low E.
        self.assertEqual([note.string for note in notes], [1])
        self.assertEqual(notes[0].fret, 0)

    def test_a_zero_length_event_is_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'take.xml'
            path.write_text(XML.format(tuning='40,45,50,55,59,64'))
            self.assertEqual(len(idmt.load_notes(path)), 1)

    def test_a_non_standard_tuning_recording_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'take.xml').write_text(XML.format(tuning='38  45  50  55  59  64'))
            (root/'take.wav').write_bytes(b'')
            found, skipped = idmt.recordings(root)
        self.assertEqual(found, [])
        self.assertEqual(len(skipped), 1)

    def test_a_standard_tuning_recording_is_paired_with_its_audio(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'take.xml').write_text(XML.format(tuning='40  45  50  55  59  64'))
            (root/'take.wav').write_bytes(b'')
            found, _ = idmt.recordings(root)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].audio.name, 'take.wav')
        self.assertEqual(found[0].instrument, 'Test Guitar')


if __name__ == '__main__':
    unittest.main()
