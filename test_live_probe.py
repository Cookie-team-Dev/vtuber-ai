"""The discovered model list is probed and trimmed to models that answer.

fastai lists every model a provider sells and then asks all of them on every
reply. Measured here that cost 68s: gemini-2.5-flash and gemini-2.5-pro both
404 "no longer available to new users", gemini-flash-latest 429s, and
gemini-3.8-flash hangs 25s before a read timeout, while gemini-3.5-flash
answers the same 1-token probe in 0.8s. _probe_live_models is what drops the
dead ones, and KOOKIE_PROBE_LIVE=0 is what lets the rest of the codebase import
this module without hitting a provider.
"""

import os
import unittest
from unittest import mock

import fastai


class LiveProbeTests(unittest.TestCase):
    def setUp(self):
        fastai._MODEL_CACHE.clear()

    def test_probe_disabled_returns_list_unchanged(self):
        with mock.patch.dict(os.environ, {"KOOKIE_PROBE_LIVE": "0"}):
            models = ("a", "b", "c")
            self.assertEqual(
                fastai._probe_live_models("https://example.com", "k", models, "P"),
                models,
            )

    def test_probe_disabled_accepts_false_and_no(self):
        for value in ("false", "no", "0"):
            with mock.patch.dict(os.environ, {"KOOKIE_PROBE_LIVE": value}):
                self.assertEqual(
                    fastai._probe_live_models("https://example.com", "k", ("a",), "P"),
                    ("a",),
                )

    def test_probe_enabled_by_default(self):
        # Default is to probe. A stubbed session that answers 200 for one model
        # and 404 for the other should return only the live one.
        def fake_post(url, headers=None, json=None, timeout=None):
            class R:
                # The probe URL is base_url/chat/completions, so the model name
                # is in the request body - that is what decides the answer.
                status_code = 200 if "good" in json["model"] else 404
            return R()
        with mock.patch.object(fastai._SESSION, "post", side_effect=fake_post):
            result = fastai._probe_live_models(
                "https://example.com", "k", ("good-1", "bad-1"), "P"
            )
        self.assertEqual(list(result), ["good-1"])

    def test_probe_keeps_list_when_nothing_answers(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            class R:
                status_code = 500
            return R()
        with mock.patch.object(fastai._SESSION, "post", side_effect=fake_post):
            models = ("a", "b")
            # All fail fast, so the budget is not the binding constraint here.
            result = fastai._probe_live_models(
                "https://example.com", "k", models, "P"
            )
        self.assertEqual(list(result), list(models))

    def test_probe_does_not_run_for_pinned_model(self):
        # A pinned model skips discovery entirely, so no probe runs.
        with mock.patch.object(fastai, "_key_for", return_value="gemini-3.5-flash"):
            with mock.patch.object(
                fastai, "_probe_live_models", side_effect=AssertionError
            ) as probe:
                result = fastai._discover_models(
                    "GEMINI_API_KEY",
                    "https://generativelanguage.googleapis.com/v1beta/openai",
                    ("gemini-3.5-flash", "gemini-2.5-flash"),
                    "Gemini",
                )
        probe.assert_not_called()
        # Discovery is skipped for a pinned model, so the full static list is
        # returned unchanged - the caller narrows it to the pinned model.
        self.assertEqual(result, ("gemini-3.5-flash", "gemini-2.5-flash"))


if __name__ == "__main__":
    unittest.main()