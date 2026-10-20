"""Prepared excerpts round-trip through the raw, memory-mappable layout."""
from __future__ import annotations
import tempfile, unittest
from pathlib import Path
import numpy as np

from processing.music import verify
from processing.training import verifier_data


class PlanesTest(unittest.TestCase):
    def prepared(self, cqt):
        rng = np.random.default_rng(1)
        frames, semitones = 40, 88
        note = rng.random((frames, semitones)).astype(np.float32)
        onset = rng.random((frames, semitones)).astype(np.float32)
        contour = rng.random((frames, 3*semitones)).astype(np.float32)
        spectrum = rng.random((frames, semitones, 3)).astype(np.float32) if cqt else None
        return verify.prepare(note, onset, contour, spectrum)

    def test_patches_match_with_and_without_cqt(self):
        for cqt in (False, True):
            prepared = self.prepared(cqt)
            channels = verify.CHANNELS + (verify.CQT_CHANNELS if cqt else 0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'x.npy'
                np.save(path, verifier_data.stack(prepared))
                mapped = verifier_data.unstack(np.load(path, mmap_mode='r'))
                self.assertEqual(len(mapped), len(prepared))
                direct = verify.cut(prepared, [3, 17], [10, 40], channels)
                via_disk = verify.cut(mapped, [3, 17], [10, 40], channels)
            np.testing.assert_array_equal(direct, via_disk)
            self.assertEqual(via_disk.shape[1], channels)


if __name__ == '__main__':
    unittest.main()
