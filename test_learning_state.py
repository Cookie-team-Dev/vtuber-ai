import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import learning


class LearningStateTests(unittest.TestCase):
    def setUp(self):
        self.state = copy.deepcopy(learning._state)
        self.corpus_keys = set(learning._corpus_keys)

    def tearDown(self):
        learning._state.clear()
        learning._state.update(self.state)
        learning._corpus_keys.clear()
        learning._corpus_keys.update(self.corpus_keys)

    def test_empty_state_load_does_not_raise_or_emit_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning.json"
            path.write_text("", encoding="utf-8")

            with self.assertLogs(level="WARNING") as captured:
                learning.load(path)

        self.assertEqual(len(captured.output), 1)
        self.assertIn("empty or invalid JSON", captured.output[0])
        self.assertNotIn("Traceback", captured.output[0])

    def test_save_atomically_replaces_an_empty_state_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning.json"
            path.write_text("", encoding="utf-8")

            self.assertTrue(learning.save(path))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["facts"], [])
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_failed_atomic_replace_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning.json"
            path.write_text('{"existing": true}', encoding="utf-8")

            with mock.patch.object(learning.os, "replace", side_effect=OSError("read-only")):
                self.assertFalse(learning.save(path))

            self.assertEqual(path.read_text(encoding="utf-8"), '{"existing": true}')
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
