import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import vision_helper


class VisionHelperTests(unittest.TestCase):
    def setUp(self):
        self.breaker = vision_helper._FAST_PROVIDER_BREAKER
        self.breaker._down_until = 0.0
        self.breaker._failures = 0
        self.breaker.cooldown_seconds = 60.0
        self.breaker.attempts = 1
        self.breaker.delay_seconds = 0.0
        self.breaker.threshold = 1
        self.breaker.jitter = 0.0

    def test_describe_passes_path_bytes_to_fast_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.bin"
            payload = b"image bytes"
            path.write_bytes(payload)
            with mock.patch("fastai.describe_image", return_value="a cat") as describe:
                self.assertEqual(vision_helper.describe(path), "a cat")
        describe.assert_called_once()
        encoded = describe.call_args.args[0]
        self.assertEqual(base64.b64decode(encoded), payload)

    def test_falls_back_to_openamika_when_fast_provider_is_unavailable(self):
        with mock.patch(
            "vision_helper._describe_with_fast_provider",
            side_effect=RuntimeError("fast provider down"),
        ), mock.patch(
            "vision_helper._describe_with_openamika",
            return_value="local description",
        ) as openamika:
            result = vision_helper.describe(b"image bytes")
        self.assertEqual(result, "local description")
        openamika.assert_called_once()

    def test_fast_provider_retries_before_falling_back(self):
        self.breaker.attempts = 2
        with mock.patch(
            "vision_helper._describe_with_fast_provider",
            side_effect=[RuntimeError("first"), "fast description"],
        ) as fast, mock.patch("vision_helper.time.sleep") as sleep, mock.patch(
            "vision_helper._describe_with_openamika",
            return_value="local description",
        ) as openamika:
            result = vision_helper.describe(b"image bytes")

        self.assertEqual(result, "fast description")
        self.assertEqual(fast.call_count, 2)
        sleep.assert_not_called()
        openamika.assert_not_called()

    def test_fast_provider_failure_starts_cooldown(self):
        self.breaker.threshold = 1
        with mock.patch.object(self.breaker, "cooldown_seconds", 60), mock.patch(
            "kookie_resilience.time.time", return_value=100
        ), mock.patch(
            "vision_helper._describe_with_fast_provider",
            side_effect=RuntimeError("fast provider down"),
        ) as fast, mock.patch(
            "vision_helper._describe_with_openamika",
            return_value="local description",
        ) as openamika:
            first = vision_helper.describe(b"image bytes")
            second = vision_helper.describe(b"image bytes")
        self.assertEqual(first, "local description")
        self.assertEqual(second, "local description")
        self.assertEqual(fast.call_count, 1)
        self.assertEqual(openamika.call_count, 2)
        self.assertEqual(self.breaker._down_until, 160)

    def test_fast_provider_retry_backoff_then_openamika_fallback(self):
        self.breaker.attempts = 3
        self.breaker.delay_seconds = 0.2
        with mock.patch(
            "vision_helper._describe_with_fast_provider",
            side_effect=RuntimeError("fast provider down"),
        ) as fast, mock.patch("vision_helper.time.sleep") as sleep, mock.patch(
            "vision_helper._describe_with_openamika",
            return_value="local description",
        ) as openamika:
            result = vision_helper.describe(b"image bytes")

        self.assertEqual(result, "local description")
        self.assertEqual(fast.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_has_calls([mock.call(0.2), mock.call(0.4)])
        openamika.assert_called_once()

    def test_fast_provider_failure_threshold_requires_multiple_calls(self):
        self.breaker.threshold = 2
        self.breaker.cooldown_seconds = 60.0
        with mock.patch("vision_helper.time.time", return_value=100), mock.patch(
            "vision_helper._describe_with_fast_provider",
            side_effect=RuntimeError("fast provider down"),
        ) as fast, mock.patch(
            "vision_helper._describe_with_openamika",
            return_value="local description",
        ) as openamika:
            first = vision_helper.describe(b"image bytes")
            second = vision_helper.describe(b"image bytes")
            third = vision_helper.describe(b"image bytes")

        self.assertEqual(first, "local description")
        self.assertEqual(second, "local description")
        self.assertEqual(third, "local description")
        self.assertEqual(fast.call_count, 2)
        self.assertEqual(openamika.call_count, 3)
        self.assertEqual(self.breaker._down_until, 160)

    def test_fast_provider_is_retried_after_cooldown(self):
        self.breaker.threshold = 1
        self.breaker.cooldown_seconds = 60.0
        # Each describe() call reads the clock in allow() and again in record()
        # when the failure trips the breaker, so the sequence covers both
        # reads for both calls.
        with mock.patch("kookie_resilience.time.time", side_effect=[100, 100, 161, 161]), mock.patch(
            "vision_helper._describe_with_fast_provider",
            side_effect=[RuntimeError("fast provider down"), "fast description"],
        ) as fast, mock.patch(
            "vision_helper._describe_with_openamika",
            return_value="local description",
        ) as openamika:
            first = vision_helper.describe(b"image bytes")
            second = vision_helper.describe(b"image bytes")
        self.assertEqual(first, "local description")
        self.assertEqual(second, "fast description")
        self.assertEqual(fast.call_count, 2)
        openamika.assert_called_once()

    def test_oversized_input_is_rejected(self):
        with mock.patch.object(vision_helper, "_MAX_IMAGE_BYTES", 3):
            result = vision_helper.describe(b"1234")
        self.assertIn("too large", result)

    def test_invalid_limit_environment_falls_back_to_default(self):
        with mock.patch.dict("os.environ", {"KOOKIE_MAX_VISION_IMAGE_MB": "invalid"}):
            self.assertEqual(vision_helper._image_limit_bytes(), 16 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
