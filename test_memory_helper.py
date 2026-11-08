"""Tests for the memory helper surface.

memory.py writes to the project directory, so these tests point the paths at a
temp dir and restore them afterwards. They never touch the real memory file.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import memory_helper


class MemoryHelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._paths = [
            str(Path(self._tmp.name) / "memory.json"),
            str(Path(self._tmp.name) / "memory_journal.log"),
        ]
        self._patcher = mock.patch.object(
            memory_helper._memory, "_memory_paths", return_value=tuple(self._paths)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _write_raw(self, entries):
        with open(self._paths[0], "w", encoding="utf-8") as f:
            json.dump(entries, f)

    def test_remember_appends_and_returns_true(self):
        self.assertTrue(memory_helper.remember("I like pizza"))
        self.assertTrue(memory_helper.remember("Cookie is my Mommy"))
        hits = memory_helper.search("pizza")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["text"], "I like pizza")
        self.assertGreater(hits[0]["time"], 0.0)

    def test_remember_refuses_slurs(self):
        self.assertFalse(memory_helper.remember("that was a bitch move"))
        self.assertEqual(memory_helper.summary()["count"], 0)

    def test_search_is_case_insensitive_and_newest_first(self):
        self._write_raw([
            {"text": "old note about cats", "time": 1.0},
            {"text": "newer note about Cats", "time": 2.0},
        ])
        hits = memory_helper.search("cats")
        self.assertEqual([h["text"] for h in hits],
                         ["newer note about Cats", "old note about cats"])

    def test_search_respects_limit(self):
        entries = [{"text": f"note {i} about pizza", "time": float(i)} for i in range(5)]
        self._write_raw(entries)
        hits = memory_helper.search("pizza", limit=2)
        self.assertEqual(len(hits), 2)

    def test_forget_removes_matching_notes(self):
        self._write_raw([
            {"text": "keep this one", "time": 1.0},
            {"text": "drop this one", "time": 2.0},
        ])
        removed = memory_helper.forget("drop")
        self.assertEqual(removed, 1)
        self.assertEqual([h["text"] for h in memory_helper.recent()], ["keep this one"])

    def test_forget_returns_zero_when_nothing_matches(self):
        self._write_raw([{"text": "nothing here", "time": 1.0}])
        self.assertEqual(memory_helper.forget("zzz"), 0)

    def test_recent_returns_newest_first(self):
        self._write_raw([
            {"text": "first", "time": 1.0},
            {"text": "second", "time": 2.0},
            {"text": "third", "time": 3.0},
        ])
        texts = [h["text"] for h in memory_helper.recent(limit=2)]
        self.assertEqual(texts, ["third", "second"])

    def test_summary_reports_count_and_last_ago(self):
        self._write_raw([
            {"text": "a", "time": 1_000_000.0},
            {"text": "b", "time": 2_000_000.0},
        ])
        snap = memory_helper.summary()
        self.assertEqual(snap["count"], 2)
        self.assertEqual(snap["last_at"], 2_000_000.0)
        self.assertTrue(snap["last_ago"])

    def test_summary_empty_store(self):
        snap = memory_helper.summary()
        self.assertEqual(snap["count"], 0)
        self.assertEqual(snap["last_ago"], "never")

    def test_recall_context_returns_empty_string_when_no_match(self):
        self.assertEqual(memory_helper.recall_context("zzz"), "")

    def test_recall_context_formats_hits_for_prompts(self):
        self._write_raw([{"text": "I love pizza on Fridays", "time": __import__("time").time()}])
        ctx = memory_helper.recall_context("pizza")
        self.assertIn("I love pizza on Fridays", ctx)
        self.assertIn("(just now)", ctx)


if __name__ == "__main__":
    unittest.main()