"""fastai reads cloud keys out of .env, so a launch that never exports them
still answers in seconds instead of falling through to the local model.

The fifteen Gemini keys live in .env, which is gitignored: an operator
who sets them up once and then starts the bot with ./Kookie expects them to
be found, and nothing in the launch path exports them. fastai._key_file_values
reads the file lazily on first use, which is why this works - these tests
exercise that reader directly rather than the whole chat path.
"""

import os
import unittest
from unittest import mock

import fastai


class FastAIDotenvTests(unittest.TestCase):
    def setUp(self):
        # _key_file_values caches by file stamp, so a test that changes the
        # file needs the cache cleared or it reads the old contents.
        fastai._KEY_FILE_VALUES = None
        fastai._KEY_FILE_STAMP = None

    def test_key_for_falls_back_to_dotenv(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            # The real .env has fifteen keys; assert the reader found them
            # rather than asserting on a specific value, so a key rotation
            # does not silently break this test.
            keys = fastai.keys_for("GEMINI_API_KEY")
            self.assertGreater(len(keys), 0)
            self.assertTrue(all(k for k in keys))

    def test_keys_for_splits_commas_and_spaces(self):
        with mock.patch.object(
            fastai, "_key_file_values",
            return_value={"GROQ_API_KEY": "a, b c\nd"},
        ):
            self.assertEqual(
                fastai.keys_for("GROQ_API_KEY"), ["a", "b", "c", "d"]
            )

    def test_keys_for_dedupes(self):
        with mock.patch.object(
            fastai, "_key_file_values",
            return_value={"OPENROUTER_API_KEY": "k, k, k"},
        ):
            self.assertEqual(fastai.keys_for("OPENROUTER_API_KEY"), ["k"])

    def test_keys_for_empty_when_unset(self):
        with mock.patch.object(
            fastai, "_key_file_values", return_value={}
        ):
            self.assertEqual(fastai.keys_for("MISSING_API_KEY"), [])

    def test_env_beats_dotenv(self):
        with mock.patch.dict(
            os.environ, {"GEMINI_API_KEY": "exported-key"}, clear=False
        ):
            with mock.patch.object(
                fastai, "_key_file_values",
                return_value={"GEMINI_API_KEY": "file-key"},
            ):
                self.assertEqual(fastai._key_for("GEMINI_API_KEY"), "exported-key")


if __name__ == "__main__":
    unittest.main()