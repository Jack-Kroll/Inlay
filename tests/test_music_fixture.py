import json
from pathlib import Path
import unittest


class MusicFixtureTests(unittest.TestCase):
    def test_example_has_consistent_string_pitch_and_timing(self):
        """Semantic fixture check; not a replacement for JSON Schema validation."""
        path = Path(__file__).parent / "fixtures" / "performance.json"
        score = json.loads(path.read_text())
        self.assertEqual(score["version"], 1)
        for note in score["notes"]:
            self.assertGreaterEqual(note["startSeconds"], 0)
            self.assertGreater(note["durationSeconds"], 0)
            self.assertEqual("string" in note, "fret" in note)
            if "string" in note:
                self.assertGreaterEqual(note["string"], 1)
                self.assertLessEqual(note["string"], len(score["tuning"]))
                self.assertEqual(note["pitch"], score["tuning"][note["string"] - 1] + note["fret"])
