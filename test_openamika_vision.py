import json
import tempfile
import unittest
from unittest import mock

import video_perception


def _png_bytes():
    from PIL import Image
    import io
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class OpenAmikaVisionTests(unittest.TestCase):
    def setUp(self):
        self.breaker = video_perception._OPENAMIKA_VISION_BREAKER
        self.breaker._down_until = 0.0
        self.breaker._failures = 0
        self.breaker.cooldown_seconds = 20.0
        self.breaker.attempts = 1
        self.breaker.delay_seconds = 0.0
        self.breaker.threshold = 1
        self.breaker.jitter = 0.0

    def test_vision_posts_image_and_returns_description(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(_png_bytes())
            image_file.flush()
            with mock.patch.object(
                video_perception.urllib.request, "urlopen",
                return_value=_Response({"description": "a cat", "ok": True}),
            ) as urlopen, mock.patch.object(
                video_perception, "describe_images", return_value="local fallback"
            ):
                result = video_perception.describe_image_file(image_file.name)

            self.assertEqual(result, "a cat")
            self.assertEqual(urlopen.call_count, 1)

    def test_vision_falls_back_to_local_stack(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(_png_bytes())
            image_file.flush()
            with mock.patch.object(
                video_perception.urllib.request, "urlopen",
                side_effect=OSError("vision offline"),
            ) as urlopen, mock.patch.object(
                video_perception, "describe_images", return_value="local fallback"
            ) as local:
                result = video_perception.describe_image_file(image_file.name)

            self.assertEqual(result, "local fallback")
            local.assert_called_once()

    def test_vision_retry_backoff_then_cooldown(self):
        self.breaker.attempts = 3
        self.breaker.delay_seconds = 0.2
        self.breaker.cooldown_seconds = 60.0
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(_png_bytes())
            image_file.flush()
            with mock.patch("kookie_resilience.time.time", return_value=100), mock.patch.object(
                video_perception.urllib.request, "urlopen",
                side_effect=OSError("vision offline"),
            ) as urlopen, mock.patch.object(
                video_perception.time, "sleep"
            ) as sleep, mock.patch.object(
                video_perception, "describe_images", return_value="local fallback"
            ):
                result = video_perception.describe_image_file(image_file.name)

            self.assertEqual(result, "local fallback")
            self.assertEqual(urlopen.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            sleep.assert_has_calls([mock.call(0.2), mock.call(0.4)])
            self.assertEqual(self.breaker._down_until, 160)

    def test_vision_cooldown_skips_repeated_failing_requests(self):
        self.breaker.cooldown_seconds = 60.0
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(_png_bytes())
            image_file.flush()
            with mock.patch("kookie_resilience.time.time", return_value=100), mock.patch.object(
                video_perception.urllib.request, "urlopen",
                side_effect=OSError("vision offline"),
            ) as urlopen, mock.patch.object(
                video_perception, "describe_images", return_value="local fallback"
            ):
                video_perception.describe_image_file(image_file.name)
                video_perception.describe_image_file(image_file.name)

            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(self.breaker._down_until, 160)

    def test_vision_is_retried_after_cooldown(self):
        self.breaker.threshold = 1
        self.breaker.cooldown_seconds = 60.0
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(_png_bytes())
            image_file.flush()
            # allow() reads the clock on every call and record() reads it
            # again when the failure trips the breaker, so the sequence
            # covers both reads for both calls.
            with mock.patch("kookie_resilience.time.time", side_effect=[100, 100, 161, 161]), mock.patch.object(
                video_perception.urllib.request, "urlopen",
                side_effect=[OSError("vision offline"), _Response({"description": "recovered", "ok": True})],
            ) as urlopen, mock.patch.object(
                video_perception, "describe_images", return_value="local fallback"
            ):
                first = video_perception.describe_image_file(image_file.name)
                second = video_perception.describe_image_file(image_file.name)

            self.assertEqual(first, "local fallback")
            self.assertEqual(second, "recovered")
            self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
