"""CLI integration checks without downloading models or requiring a GPU."""
import json
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave
import cv2
import numpy as np

from processing.music.pitch import NoteEvent, Posteriorgram
from processing.music.transcribe import FrameRecord, main, mux_audio


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for audio mux integration')
class AudioMuxTests(unittest.TestCase):
    def test_short_audio_preserves_all_rendered_frames_and_selects_overlay_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, size in [('overlay', (64, 48)), ('audio_video', (128, 96))]:
                writer = cv2.VideoWriter(str(root/f'{name}.mp4'),
                                         cv2.VideoWriter.fourcc(*'mp4v'), 10, size)
                self.assertTrue(writer.isOpened())
                for _ in range(10):
                    writer.write(np.zeros((size[1], size[0], 3), np.uint8))
                writer.release()
            with wave.open(str(root/'short.wav'), 'wb') as audio:
                audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(22050)
                audio.writeframes(b'\x00\x00' * 4410)  # Only 0.2s of audio for a 1s video.
            subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
                            '-i', str(root/'audio_video.mp4'), '-i', str(root/'short.wav'),
                            '-c:v', 'copy', '-c:a', 'aac', str(root/'soundtrack.mp4')],
                           check=True, capture_output=True)
            for audio in (root/'short.wav', root/'soundtrack.mp4'):
                output = root/'result.mp4'
                self.assertEqual(mux_audio(root/'overlay.mp4', audio, output), output)
                capture = cv2.VideoCapture(str(output))
                count = 0
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        self.assertEqual(frame.shape[:2], (48, 64))
                        count += 1
                finally:
                    capture.release()
                self.assertEqual(count, 10)


class TranscriptionCliTests(unittest.TestCase):
    def run_cli(self, root, records, max_fret=22):
        argv = ['transcribe', '--model', 'unused.pt', '--video', 'take.mov',
                '--output', str(root), '--device', 'cpu', '--no-video',
                '--tuning', '64,59,55,50,45,40,35', '--max-frames', '2',
                '--max-fret', str(max_fret)]
        notes = [NoteEvent(.05, 1, 40, .8), NoteEvent(.2, 1, 45, .8)]
        activations = Posteriorgram(np.zeros((4, 88), dtype=np.float32),
                                    np.zeros((4, 88), dtype=np.float32),
                                    np.linspace(0, .3, 4))
        with patch('sys.argv', argv), patch('builtins.print'), \
                patch('processing.music.transcribe.notes_from_media',
                      return_value=(notes, activations, root/'sound.wav')) as audio, \
                patch('processing.music.transcribe.analyse', return_value=(records, 10.)):
            main()
        return audio.call_args.kwargs

    def test_truncated_clip_limits_notes_and_uses_configured_pitch_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self.run_cli(root, [FrameRecord(0, 0, 'lost'), FrameRecord(1, .1, 'lost')])
            rows = [json.loads(line) for line in (root/'take.tab.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['notes'], 1)
        self.assertEqual(rows[0]['support_counts'], {'no-board': 1})
        self.assertEqual(set(rows[0]['string_order_scores']), {'normal', 'flipped'})
        self.assertEqual(rows[1]['durationSeconds'], .15)
        # Low B on a seven-string must not be removed by the old fixed 70 Hz cutoff.
        self.assertAlmostEqual(kwargs['minimum_frequency'], 440 * 2 ** ((35-69)/12))

    @unittest.skipUnless(find_spec('basic_pitch'), 'Install the transcribe extra for pitch decoding')
    def test_frequency_cutoff_keeps_top_note_for_odd_and_even_fret_limits(self):
        from basic_pitch.note_creation import constrain_frequency
        for max_fret in (21, 22):
            with self.subTest(max_fret=max_fret), tempfile.TemporaryDirectory() as tmp:
                kwargs = self.run_cli(Path(tmp), [FrameRecord(0, 0, 'lost')], max_fret)
                onset, frame = constrain_frequency(np.ones((2, 88)), np.ones((2, 88)),
                                                   kwargs['maximum_frequency'],
                                                   kwargs['minimum_frequency'])
                top = 64 + max_fret - 21
                self.assertEqual(frame[0, top], 1)
                self.assertEqual(frame[0, top + 1], 0)
                self.assertEqual(onset[0, 35 - 21], 1)
                self.assertEqual(onset[0, 35 - 22], 0)

    def test_empty_video_fails_before_writing_misleading_transcription(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'No video frames'):
                self.run_cli(root, [])
            self.assertFalse((root/'take.tab.jsonl').exists())

    def test_invalid_tuning_is_reported_as_cli_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['transcribe', '--model', 'unused.pt', '--video', 'take.mov',
                    '--output', tmp, '--tuning', 'E,A,D,G,B,E']
            with patch('sys.argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
