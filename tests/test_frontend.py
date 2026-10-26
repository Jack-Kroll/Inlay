"""The PyTorch Basic Pitch and the targets it is fine-tuned against."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from processing.music import frontend, verify
from processing.training import frontend_data


class FrontEndTests(unittest.TestCase):
    def test_heads_have_basic_pitch_shapes(self):
        model = frontend.BasicPitch().eval()
        with torch.no_grad():
            heads = model(torch.zeros(2, frontend.WINDOW))
        self.assertEqual(heads['note'].shape, (2, frontend.FRAMES, frontend.NOTE_BINS))
        self.assertEqual(heads['onset'].shape, (2, frontend.FRAMES, frontend.NOTE_BINS))
        self.assertEqual(heads['contour'].shape, (2, frontend.FRAMES, frontend.CONTOUR_BINS))

    def test_harmonic_stack_lines_partials_up_with_their_fundamental(self):
        spectrum = torch.zeros(1, 1, 1, frontend.CQT_BINS)
        fundamental = 30
        for harmonic in (1, 2, 3):
            shift = int(np.round(36*np.log2(harmonic)))
            spectrum[..., fundamental + shift] = 1.
        stacked = frontend.harmonic_stack(spectrum)
        self.assertEqual(stacked.shape, (1, 8, 1, frontend.CONTOUR_BINS))
        # Channels 1..3 are harmonics 1, 2, 3: each puts its partial on the fundamental.
        for channel in (1, 2, 3):
            self.assertEqual(float(stacked[0, channel, 0, fundamental]), 1.)
        # The sub-harmonic channel looks an octave down, so nothing lines up there.
        self.assertEqual(float(stacked[0, 0, 0, fundamental]), 0.)

    def test_tensorflow_same_padding(self):
        self.assertEqual(frontend._same(7, 3, 264), (2, 2))
        self.assertEqual(frontend._same(5, 3, 264), (1, 1))

    def test_windows_and_stitch_cover_the_audio_like_run_inference(self):
        wave = np.random.default_rng(0).standard_normal(frontend.SAMPLE_RATE*5).astype(np.float32)
        chunks = frontend.windows(wave)
        self.assertEqual(chunks.shape[1], frontend.WINDOW)
        # Each window contributes FRAMES minus the overlap once stitched.
        fake = np.zeros((len(chunks), frontend.FRAMES, 3), np.float32)
        stitched = frontend.stitch(fake, len(wave))
        self.assertEqual(len(stitched), int(np.floor(len(wave)*frontend.FPS/frontend.SAMPLE_RATE)))

    def test_renormalize_restores_unit_kernels(self):
        model = frontend.BasicPitch()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(3.)
        model.renormalize()
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                norms = module.weight.flatten(1).norm(dim=1)
                self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_keras_weights_map_onto_every_layer(self):
        model = frontend.BasicPitch()
        variables = {}
        shapes = {1: (3, 39, 8, 8), 3: (5, 5, 8, 1), 4: (7, 7, 1, 32), 5: (5, 5, 8, 32),
                  7: (7, 3, 32, 1), 8: (3, 3, 33, 1)}
        for n, shape in shapes.items():
            variables[f'layer_with_weights-{n}/kernel/.ATTRIBUTES/VARIABLE_VALUE'] = np.ones(shape, np.float32)
            variables[f'layer_with_weights-{n}/bias/.ATTRIBUTES/VARIABLE_VALUE'] = np.zeros(shape[-1], np.float32)
        for n, size in ((0, 1), (2, 8), (6, 32)):
            for name in ('gamma', 'beta', 'moving_mean', 'moving_variance'):
                variables[f'layer_with_weights-{n}/{name}/.ATTRIBUTES/VARIABLE_VALUE'] = np.ones(size, np.float32)
        missing, unexpected = model.load_state_dict(frontend.keras_state_dict(variables), strict=False)
        self.assertEqual(unexpected, [])
        self.assertEqual([k for k in missing if not k.startswith('cqt.')
                          and not k.endswith('num_batches_tracked')], [])

    def test_cache_directories_never_collide_with_basic_pitch(self):
        root = Path('data/guitarset')
        self.assertEqual(frontend.cache_dir(root), root/'posteriorgrams')
        self.assertEqual(frontend.cache_dir(root, Path('runs/frontend/ft1/best.pt')),
                         root/'posteriorgrams@ft1')
        self.assertEqual(frontend.tag(Path('runs/frontend/ft1/last.pt')), 'ft1-last')

    def test_verifier_names_its_front_end(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory)/'old.pt', Path(directory)/'new.pt'
            torch.save({'model': {}}, old)
            torch.save({'model': {}, 'front_end': 'runs/frontend/ft1/best.pt'}, new)
            self.assertIsNone(verify.front_end(old))
            self.assertEqual(verify.front_end(new), Path('runs/frontend/ft1/best.pt'))
            self.assertIsNone(verify.front_end(Path(directory)/'missing.pt'))

    @unittest.skipUnless(frontend.SHIPPED.is_file(), 'converted Basic Pitch weights not present')
    def test_port_matches_the_released_core_ml_model(self):
        import contextlib, io, soundfile
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import run_inference
        rate = frontend.SAMPLE_RATE
        t = np.arange(rate*3)/rate
        wave = np.zeros_like(t)
        for f, start in ((110., 0.), (164.8, .5), (329.6, 1.2)):
            wave += np.sin(2*np.pi*f*t)*np.exp(-3*((t - start) % 1.))
        wave = (wave*.2).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tone.wav'
            soundfile.write(str(path), wave, rate)
            with contextlib.redirect_stdout(io.StringIO()):
                reference = run_inference(str(path), ICASSP_2022_MODEL_PATH)
            ported = frontend.infer(frontend.load(), frontend.load_audio(path), 'cpu')
        for key in ('note', 'onset', 'contour'):
            self.assertLess(float(np.abs(np.asarray(reference[key]) - ported[key]).max()), 1e-3, key)


class TargetTests(unittest.TestCase):
    def test_frames_onsets_and_contour_bins(self):
        hop, rate = frontend.HOP, frontend.SAMPLE_RATE
        start = 1000
        # A4 from frame 10 to frame 20 of the window.
        begin, end = (start + 10*hop)/rate, (start + 20*hop)/rate
        contour = np.array([[(start + 12*hop)/rate, 69.]], np.float32)
        note, onset, target = frontend_data.targets(np.array([[begin, end, 69.]]), contour, start)
        column = 69 - frontend_data.LOWEST_PITCH
        self.assertEqual(np.flatnonzero(note[:, column]).tolist(), list(range(10, 20)))
        self.assertEqual(np.argwhere(onset).tolist(), [[10, column]])
        # The middle of A4's three contour sub-bins, where the released model puts it.
        self.assertEqual(np.argwhere(target).tolist(), [[12, column*3 + 1]])

    def test_untrusted_onsets_and_missing_contours_stay_empty(self):
        note, onset, target = frontend_data.targets(
            np.array([[.1, .5, 60.]]), np.array([[.2, 60.]], np.float32), 0,
            onsets=False, contours=False)
        self.assertTrue(note.any())
        self.assertFalse(onset.any())
        self.assertFalse(target.any())

    def test_notes_starting_before_the_window_have_no_onset(self):
        rate = frontend.SAMPLE_RATE
        note, onset, _ = frontend_data.targets(np.array([[0., 2., 50.]]),
                                               np.zeros((0, 2), np.float32), rate//2)
        self.assertTrue(note[0].any())
        self.assertFalse(onset.any())

    def test_single_note_onset_and_decay(self):
        rate = frontend.SAMPLE_RATE
        t = np.arange(rate*2)/rate
        wave = np.where(t >= .25, np.sin(2*np.pi*220*t)*np.exp(-4*(t - .25)), 0.).astype(np.float32)
        start, end, pitch = frontend_data.single_note(wave, 5, 0)
        self.assertAlmostEqual(start, .25, delta=.011)
        self.assertGreater(end, 1.2)
        self.assertEqual(pitch, 45.)   # EGFxSet's string 5 is the A string.


if __name__ == '__main__':
    unittest.main()


class EgfxsetTests(unittest.TestCase):
    def test_excerpts_space_single_notes_and_keep_their_strings(self):
        from processing.tools import egfxset
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            for name, pitch in (('6-0', 40.), ('1-12', 76.)):
                audio = root/f'{name}.npy'
                labels = root/f'{name}.labels.npz'
                np.save(audio, np.full(egfxset.SAMPLE_RATE*5, .1, np.float16))
                frontend_data.write_labels(labels, [(0., 4.9, pitch)])
                items.append(frontend_data.Item(
                    name=name, group='egfx', player='egfx',
                    source=root/'Clean'/'Neck'/f'{name}.wav', audio=audio, labels=labels,
                    render='single'))
            wave, notes = egfxset.build(items, rng)
        self.assertEqual(len(wave), int(egfxset.SPACING*egfxset.NOTES_PER_EXCERPT*egfxset.SAMPLE_RATE))
        self.assertEqual([(n.start, n.pitch, n.string) for n in notes],
                         [(0., 40., 0), (egfxset.SPACING, 76., 5)])
        # Each note is cut before the next attack.
        self.assertLess(notes[0].end, egfxset.SPACING)
