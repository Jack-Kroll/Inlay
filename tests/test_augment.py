"""Training augmentation: the filters do what they claim, and none of them
moves a label. No dataset on disk and no ffmpeg required."""
import os
import shutil
import subprocess
import sys
import unittest
import numpy as np

from processing.music import verify
from processing.tools.guitarset import OPEN_STRINGS, Note
from processing.training import augment, verifier_data

RATE = augment.SAMPLE_RATE


def response(stage, hz, length=8192):
    """The gain a stage applies at ``hz``, in dB, measured off its impulse."""
    impulse = np.zeros(length, np.float32)
    impulse[0] = 1.
    spectrum = np.fft.rfft(stage(impulse))
    frequencies = np.fft.rfftfreq(length, 1/RATE)
    return 20*np.log10(abs(spectrum[np.argmin(abs(frequencies - hz))]))


def pluck(offset=4000, length=22050, hz=220.):
    """One decaying note starting at a known sample, on silence."""
    wave = np.zeros(length, np.float32)
    span = np.arange(length - offset)
    wave[offset:] = np.sin(2*np.pi*hz*span/RATE)*np.exp(-span/(RATE/8))
    return wave.astype(np.float32)


class Filters(unittest.TestCase):
    def test_a_peaking_filter_reaches_its_gain_at_its_centre(self):
        self.assertAlmostEqual(
            response(lambda x: augment.peaking(x, 1000., 9., 2.), 1000.), 9., places=1)

    def test_a_peaking_filter_leaves_distant_frequencies_alone(self):
        self.assertAlmostEqual(
            response(lambda x: augment.peaking(x, 1000., 9., 2.), 50.), 0., places=1)

    def test_a_low_shelf_moves_everything_below_its_corner(self):
        stage = lambda x: augment.shelf(x, 300., 6.)
        self.assertAlmostEqual(response(stage, 40.), 6., places=1)
        self.assertAlmostEqual(response(stage, 8000.), 0., places=1)

    def test_a_high_shelf_moves_everything_above_its_corner(self):
        stage = lambda x: augment.shelf(x, 2000., -6., high=True)
        self.assertAlmostEqual(response(stage, 10000.), -6., places=1)
        self.assertAlmostEqual(response(stage, 100.), 0., places=1)

    def test_tilt_trades_lows_against_highs(self):
        stage = lambda x: augment.tilt(x, 6.)
        self.assertAlmostEqual(response(stage, 40.), -6., places=1)
        self.assertAlmostEqual(response(stage, 10000.), 6., places=1)

    def test_a_shelf_is_stable(self):
        # The low shelf's a0 sign is the one coefficient that puts a pole on
        # the unit circle if it is wrong, and the output then overflows rather
        # than merely sounding odd.
        for gain in (-12., -6., 6., 12.):
            for high in (False, True):
                out = augment.shelf(pluck(), 300., gain, high=high)
                self.assertTrue(np.isfinite(out).all(), f'{gain} high={high}')
                self.assertLess(np.abs(out).max(), 100.)


class Causality(unittest.TestCase):
    """Nothing may put energy before the attack: a label says when a note
    started, and the verifier is scored against it to within 50 ms."""

    def quiet_before(self, stage, offset=4000):
        wave = stage(pluck(offset))
        before = float(np.abs(wave[:offset - 1]).max())
        after = float(np.abs(wave[offset:offset + 2000]).max())
        self.assertLess(before, .01*after, f'{before:.2e} before the attack')

    def test_filters_do_not_precede_the_attack(self):
        self.quiet_before(lambda x: augment.tilt(x, -6.))
        self.quiet_before(lambda x: augment.peaking(x, 400., 9., 3.))

    def test_saturation_does_not_precede_the_attack(self):
        self.quiet_before(lambda x: augment.saturate(x, 10.))

    def test_a_cabinet_impulse_is_direct_dominant(self):
        for seed in range(8):
            impulse = augment.cabinet_impulse(np.random.default_rng(seed))
            self.assertEqual(int(np.argmax(np.abs(impulse))), 0)
            self.assertLessEqual(len(impulse), int(RATE*.04))

    def test_convolution_does_not_precede_the_attack(self):
        for seed in range(4):
            impulse = augment.cabinet_impulse(np.random.default_rng(seed))
            self.quiet_before(lambda x: augment.convolve(x, impulse))


class Stages(unittest.TestCase):
    def test_saturation_keeps_the_level_it_was_given(self):
        wave = pluck()
        out = augment.saturate(wave, 12.)
        self.assertAlmostEqual(float(np.abs(out).max()),
                               float(np.abs(wave).max()), places=5)

    def test_saturation_adds_harmonics(self):
        wave = (np.sin(2*np.pi*500*np.arange(RATE)/RATE)*.5).astype(np.float32)
        def energy(signal, hz):
            spectrum = abs(np.fft.rfft(signal))
            return float(spectrum[np.argmin(abs(np.fft.rfftfreq(RATE, 1/RATE) - hz))])
        self.assertGreater(energy(augment.saturate(wave, 12.), 1500.),
                           50*energy(wave, 1500.))

    def test_hiss_lands_on_the_requested_ratio(self):
        wave = pluck()
        rng = np.random.default_rng(0)
        for snr in (20., 35.):
            noise = augment.hiss(wave, snr, rng=rng) - wave
            measured = 10*np.log10(np.mean(wave**2)/np.mean(noise**2))
            self.assertAlmostEqual(measured, snr, places=1)

    def test_pink_hiss_is_darker_than_white(self):
        wave = pluck()
        rng = np.random.default_rng(0)
        def tilt_of(pink):
            noise = augment.hiss(wave, 20., pink=pink, rng=rng) - wave
            spectrum = abs(np.fft.rfft(noise))
            half = len(spectrum)//2
            return float(spectrum[:half].mean()/spectrum[half:].mean())
        self.assertGreater(tilt_of(True), 2*tilt_of(False))

    def test_realignment_undoes_a_known_delay(self):
        wave = pluck()
        for delay in (0, 37, 1105):
            delayed = np.concatenate([np.zeros(delay, np.float32), wave])
            recovered = augment._realign(wave, delayed)
            self.assertLess(float(np.abs(recovered - wave).max()), 1e-6, f'{delay}')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg not installed')
    def test_a_codec_round_trip_stays_on_the_original_timeline(self):
        wave = pluck(offset=8000, length=4*RATE)
        out = augment.codec(wave, 96)
        self.assertEqual(len(out), len(wave))
        # An MP3 encoder prepends about 1,100 samples of its own, which is
        # twice the verifier's onset tolerance if it is not taken back off.
        self.assertGreater(float(np.corrcoef(out, wave)[0, 1]), .99)


class Chain(unittest.TestCase):
    def test_a_chain_is_finite_bounded_and_the_length_it_started(self):
        wave = pluck(length=2*RATE)
        for seed in range(25):
            out, description = augment.chain(wave.copy(), np.random.default_rng(seed))
            self.assertEqual(len(out), len(wave), description)
            self.assertTrue(np.isfinite(out).all(), description)
            self.assertLessEqual(float(np.abs(out).max()), 1., description)
            self.assertGreater(float(np.abs(out).max()), 0., description)

    def test_a_seed_reproduces_its_chain(self):
        wave = pluck()
        first, description = augment.chain(wave.copy(), np.random.default_rng(7))
        again, repeated = augment.chain(wave.copy(), np.random.default_rng(7))
        self.assertEqual(description, repeated)
        self.assertTrue(np.array_equal(first, again))

    def test_different_seeds_give_different_chains(self):
        wave = pluck()
        chains = {augment.chain(wave.copy(), np.random.default_rng(seed))[1]
                  for seed in range(10)}
        self.assertGreater(len(chains), 8)

    def test_turning_every_stage_off_leaves_only_level(self):
        wave = pluck()
        out, description = augment.chain(
            wave.copy(), np.random.default_rng(3),
            stages={key: 0. for key in augment.PROBABILITY})
        self.assertEqual(len(description.split()), 1)
        # Level alone: the shape is untouched, so the two still correlate exactly.
        self.assertAlmostEqual(float(np.corrcoef(out, wave)[0, 1]), 1., places=5)


class NeckShift(unittest.TestCase):
    """Resampling an excerpt up the neck has to move the labels with it."""

    def shifted_peak(self, semitones, hz=220.):
        wave = (np.sin(2*np.pi*hz*np.arange(RATE)/RATE)*.3).astype(np.float32)
        out = augment.neck_shift(wave, semitones)
        spectrum = abs(np.fft.rfft(out))
        return out, float(np.fft.rfftfreq(len(out), 1/RATE)[np.argmax(spectrum)])

    def test_pitch_rises_by_the_interval_asked_for(self):
        for semitones in (3, 5, 8):
            _, peak = self.shifted_peak(semitones)
            self.assertAlmostEqual(peak, 220.*2**(semitones/12), delta=1.)

    def test_duration_shortens_by_the_same_ratio(self):
        # The two have to move together, or the annotation cannot be rescaled
        # by one number.
        for semitones in (3, 5, 8):
            out, _ = self.shifted_peak(semitones)
            self.assertAlmostEqual(len(out), RATE/2**(semitones/12), delta=2)

    def test_no_shift_is_the_identity(self):
        wave = pluck()
        self.assertIs(augment.neck_shift(wave, 0), wave)

    def test_the_attack_lands_where_the_rescaled_label_says(self):
        offset = 8000
        wave = augment.neck_shift(pluck(offset, length=RATE), 5)
        ratio = 2**(5/12)
        envelope = np.abs(wave)
        attack = int(np.argmax(envelope > .2*envelope.max()))
        # Within a frame of the verifier's own grid, which is what the
        # candidate has to land inside.
        self.assertLess(abs(attack - offset/ratio), RATE/verify.FPS)

    def test_headroom_stops_at_the_end_of_the_neck(self):
        note = lambda fret, string=0: Note(0., 1., float(OPEN_STRINGS[string] + fret),
                                           string)
        self.assertEqual(augment.shift_headroom([note(5)]), augment.HIGHEST_FRET - 5)
        self.assertEqual(augment.shift_headroom([note(5), note(20)]),
                         augment.HIGHEST_FRET - 20)
        self.assertEqual(augment.shift_headroom([note(augment.HIGHEST_FRET)]), 0)
        self.assertEqual(augment.shift_headroom([]), 0)


class ShiftedLabels(unittest.TestCase):
    def test_a_note_moves_up_its_own_string(self):
        before = Note(1., 2., float(OPEN_STRINGS[2] + 4), 2)
        after, = verifier_data.shifted([before], 5)
        self.assertEqual(after.string, before.string)
        self.assertEqual(after.fret, before.fret + 5)
        self.assertAlmostEqual(after.pitch, before.pitch + 5)

    def test_times_divide_by_the_pitch_ratio(self):
        before = Note(1., 2., 55., 0)
        after, = verifier_data.shifted([before], 5)
        ratio = 2**(5/12)
        self.assertAlmostEqual(after.start, 1./ratio)
        self.assertAlmostEqual(after.end, 2./ratio)

    def test_no_shift_leaves_the_notes_alone(self):
        notes = [Note(1., 2., 55., 0)]
        self.assertIs(verifier_data.shifted(notes, 0), notes)


class Domains(unittest.TestCase):
    def test_a_name_round_trips_through_its_domain(self):
        self.assertEqual(verifier_data.variant_name('mic', '00_x'), '00_x')
        self.assertEqual(verifier_data.variant_name('aug-v1', '00_x'), 'aug-v1/00_x')
        self.assertEqual(verifier_data.domain_of('aug-v1/00_x'), 'aug-v1')
        self.assertEqual(verifier_data.domain_of('00_x'), 'mic')

    def test_validation_and_test_take_one_domain_only(self):
        players = ['00', '00', '04', '04', '05', '05']
        names = ['a', 'aug-v1/a', 'b', 'pickup/b', 'c', 'pickup/c']
        train, val, test = verifier_data.splits(players, names)
        # Every domain trains; only the microphone is measured.
        self.assertEqual(train, {0, 1})
        self.assertEqual((val, test), ({2}, {4}))

    def test_a_measured_domain_can_be_asked_for_by_name(self):
        players = ['00', '04', '04']
        names = ['a', 'b', 'pickup/b']
        _, val, _ = verifier_data.splits(players, names, domain='pickup')
        self.assertEqual(val, {2})

    def test_an_augmented_render_cannot_become_a_test_number(self):
        # The whole point of the split: a synthesised amp may only ever train.
        _, val, test = verifier_data.splits(['04', '05'], ['aug-v1/b', 'aug-v1/c'])
        self.assertEqual((val, test), (set(), set()))

    def test_a_render_seed_is_stable_across_processes(self):
        from processing.tools import variants
        # Python salts hash() of a string per process, so a re-render built on
        # it would draw a different chain from the cache it was adding to.
        name = '00_BN1-129-Eb_comp'
        elsewhere = subprocess.run(
            [sys.executable, '-c',
             'from processing.tools import variants;'
             f'print(variants.seed("aug-v1", "{name}"))'],
            capture_output=True, text=True, check=True,
            env={**os.environ, 'PYTHONHASHSEED': '1'})
        self.assertEqual(int(elsewhere.stdout), variants.seed('aug-v1', name))


if __name__ == '__main__':
    unittest.main()
