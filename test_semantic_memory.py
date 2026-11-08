"""Tests for the optional semantic memory adapter."""

import os
import unittest
from unittest import mock

import semantic_memory


class _Stamp:
    """Helper to make the rebuild deterministic."""

    def __init__(self):
        self.values = [1000.0, 1000.0, 2000.0, 2000.0]

    def __call__(self, *args, **kwargs):
        return self.values.pop(0)


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self):
        self._load_patcher = mock.patch("semantic_memory._memory.load_memory", return_value=[])
        self._load_patcher.start()
        self.addCleanup(self._load_patcher.stop)
        self._stamp_patcher = mock.patch(
            "semantic_memory._memory._memory_stamp", side_effect=_Stamp()
        )
        self._stamp_patcher.start()
        self.addCleanup(self._stamp_patcher.stop)
        self._paths_patcher = mock.patch(
            "semantic_memory._memory._memory_paths", return_value=("m", "j")
        )
        self._paths_patcher.start()
        self.addCleanup(self._paths_patcher.stop)
        self.adapter = semantic_memory.SemanticMemory()

    def test_empty_store_returns_nothing(self):
        self.assertEqual(self.adapter.search("anything"), [])

    def test_search_finds_similar_text(self):
        with mock.patch(
            "semantic_memory._memory.load_memory",
            return_value=[
                {"text": "I love pizza on Friday", "time": 1000.0},
                {"text": "The weather is nice today", "time": 1000.0},
            ],
        ):
            hits = self.adapter.search("pizza", limit=5)
        self.assertTrue(any("pizza" in h["text"] for h in hits))
        self.assertEqual(len(hits), 1)

    def test_recall_context_empty_when_no_match(self):
        with mock.patch("semantic_memory._memory.load_memory", return_value=[]):
            self.assertEqual(self.adapter.recall_context("zzz"), "")

    def test_status_reports_indexed_count(self):
        with mock.patch("semantic_memory._memory.load_memory", return_value=[]):
            status = self.adapter.status()
        self.assertEqual(status["indexed"], 0)

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KOOKIE_SEMANTIC_MEMORY", None)
            self.assertFalse(semantic_memory.is_enabled())
            self.assertIsNone(semantic_memory.get_semantic_memory())


if __name__ == "__main__":
    unittest.main()