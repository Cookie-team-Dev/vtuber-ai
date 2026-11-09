"""The probed model list survives a process restart on disk.

The live-model probe costs 8.5s and the in-memory cache does not survive a
restart, so a Kookie that had just come up paid it again on the first reply.
fastai now writes the probed list to a state file and reads it back on the first
discovery call, so a restart skips the probe and the first reply is quick
instead of cold. These tests pin the read/write contract in isolation, without
touching a provider.
"""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

import fastai


class ModelCacheDiskTests(unittest.TestCase):
    def setUp(self):
        fastai._MODEL_CACHE.clear()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "fastai_models.json")

    def tearDown(self):
        fastai._MODEL_CACHE.clear()
        if os.path.exists(self.path):
            os.remove(self.path)
        os.rmdir(self.tmp)

    def test_save_writes_only_fresh_entries(self):
        fastai._MODEL_CACHE["GEMINI_API_KEY"] = (("a", "b"), 10**9)
        # An expired entry must not be written: it would only make the next
        # process probe again, and writing it would mask that a probe failed.
        fastai._MODEL_CACHE["STALE"] = (("x",), time.monotonic() - 1)
        with mock.patch.object(fastai, "_MODEL_CACHE_FILE", self.path):
            fastai._save_model_cache()
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertIn("GEMINI_API_KEY", data)
        self.assertNotIn("STALE", data)
        self.assertEqual(data["GEMINI_API_KEY"]["models"], ["a", "b"])

    def test_load_restores_fresh_entries(self):
        payload = {
            "GEMINI_API_KEY": {
                "models": ["gemini-3.5-flash", "gemini-3.5-flash-lite"],
                "expires": 10**9,
            },
            "STALE": {"models": ["x"], "expires": 1.0},
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with mock.patch.object(fastai, "_MODEL_CACHE_FILE", self.path):
            fastai._load_model_cache()
        self.assertEqual(
            fastai._MODEL_CACHE["GEMINI_API_KEY"][0],
            ("gemini-3.5-flash", "gemini-3.5-flash-lite"),
        )
        self.assertNotIn("STALE", fastai._MODEL_CACHE)

    def test_load_ignores_corrupt_file(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("not json at all {{{")
        with mock.patch.object(fastai, "_MODEL_CACHE_FILE", self.path):
            fastai._load_model_cache()
        self.assertEqual(fastai._MODEL_CACHE, {})

    def test_load_ignores_missing_file(self):
        with mock.patch.object(fastai, "_MODEL_CACHE_FILE", self.path):
            fastai._load_model_cache()
        self.assertEqual(fastai._MODEL_CACHE, {})

    def test_load_ignores_wrong_shape(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"GEMINI_API_KEY": ["not", "a", "dict"]}, handle)
        with mock.patch.object(fastai, "_MODEL_CACHE_FILE", self.path):
            fastai._load_model_cache()
        self.assertEqual(fastai._MODEL_CACHE, {})


if __name__ == "__main__":
    unittest.main()