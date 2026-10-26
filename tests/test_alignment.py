"""Audio kept on the video's clock, and fingertips moved back onto the pad."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np

from processing.music import fusion
from processing.music.fretboard import fret_ratio
from processing.music.pitch import extract_audio, stream_offset


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg is required')
class AudioOffsetTests(unittest.TestCase):
    def render(self, directory, delay):
        """One second of black video and a tone starting ``delay`` seconds in."""
        path = Path(directory)/'take.mov'
        subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
                        '-f', 'lavfi', '-i', 'color=c=black:s=64x48:d=1:r=30',
                        '-itsoffset', str(delay), '-f', 'lavfi',
                        '-i', 'sine=frequency=440:duration=0.8:sample_rate=44100',
                        '-map', '0:v', '-map', '1:a', '-c:v', 'mpeg4', '-c:a', 'pcm_s16le',
                        str(path)], check=True, capture_output=True)
        return path

    def test_late_soundtrack_is_padded_onto_the_video_clock(self):
        import soundfile
        with tempfile.TemporaryDirectory() as directory:
            take = self.render(directory, .1)
            offset, rate = stream_offset(take)
            self.assertAlmostEqual(offset, .1, places=3)
            self.assertEqual(rate, 44100)
            wave, sample_rate = soundfile.read(str(extract_audio(take, Path(directory)/'a.wav')))
        first = int(np.flatnonzero(np.abs(wave) > .01)[0])
        self.assertAlmostEqual(first/sample_rate, .1, delta=.002)

    def test_audio_only_media_is_left_alone(self):
        self.assertEqual(stream_offset(Path('no-such-file.wav')), (0., None))


class FingertipSetbackTests(unittest.TestCase):
    def test_setback_moves_the_point_towards_the_nut_by_scale_length(self):
        contact = fusion.place_contact('index', (0., 0.), 5.2, 2., setback=.01)
        self.assertAlmostEqual(float(fret_ratio(5.2) - fret_ratio(contact.fret_float)), .01)
        self.assertEqual(contact.tip_fret, 5.2)
        self.assertEqual((contact.fret, contact.string), (5, 2))
        # The same distance is more of a fret high on the neck.
        high = fusion.place_contact('index', (0., 0.), 17.2, 2., setback=.01)
        self.assertGreater(17.2 - high.fret_float, 5.2 - contact.fret_float)

    def test_cached_contacts_without_a_tip_read_fret_float_as_the_tip(self):
        old = {'finger': 'ring', 'point': [1., 2.], 'fret': 6, 'string': 3,
               'fret_float': 5.1, 'string_float': 3.1, 'on_board': True}
        rebuilt = fusion.contact_from_cache(old, setback=.01)
        self.assertEqual(rebuilt.tip_fret, 5.1)
        self.assertEqual(rebuilt.fret, 5)
        unchanged = fusion.contact_from_cache(old, setback=0.)
        self.assertEqual(unchanged.fret, 6)
        again = fusion.contact_from_cache(vars(rebuilt) | {'point': [1., 2.]}, setback=.01)
        self.assertAlmostEqual(again.fret_float, rebuilt.fret_float)


if __name__ == '__main__':
    unittest.main()
