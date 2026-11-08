import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import video_perception


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class OpenAmikaAsrTests(unittest.TestCase):
    def setUp(self):
        self.breaker = video_perception._OPENAMIKA_ASR_BREAKER
        self.breaker._down_until = 0.0
        self.breaker._failures = 0
        self.breaker.cooldown_seconds = 20.0
        self.breaker.attempts = 1
        self.breaker.delay_seconds = 0.0
        self.breaker.threshold = 1
        self.breaker.jitter = 0.0

    def test_posts_audio_to_openamika_and_returns_clean_transcription(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "audio.wav"
            wav_path.write_bytes(b"test wav bytes")
            with mock.patch.object(
                video_perception.urllib.request,
                "urlopen",
                return_value=_Response(
                    {"result": [{"text": "raw", "clean_text": "hello there"}]}
                ),
            ) as urlopen:
                text = video_perception._transcribe_openamika(wav_path)

        request = urlopen.call_args.args[0]
        self.assertEqual(text, "hello there")
        self.assertIn(b"test wav bytes", request.data)
        self.assertIn(b'name="lang"\r\n\r\nauto', request.data)
        self.assertEqual(request.full_url, video_perception._OPENAMIKA_ASR_URL)

    def test_transcribe_prefers_openamika_without_loading_whisper(self):
        with (
            mock.patch.object(
                video_perception, "_transcribe_openamika", return_value="recognized words"
            ),
            mock.patch.object(video_perception, "_whisper_model") as whisper_model,
        ):
            result = video_perception.transcribe("audio.wav")

        self.assertEqual(result, "recognized words")
        whisper_model.assert_not_called()

    def test_unavailable_openamika_asr_returns_none_for_local_fallback(self):
        with (
            tempfile.NamedTemporaryFile() as audio_file,
            mock.patch.object(
                video_perception.urllib.request,
                "urlopen",
                side_effect=OSError("service is offline"),
            ),
        ):
            audio_file.write(b"wav")
            audio_file.flush()
            self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))

    def test_asr_retries_before_falling_back(self):
        self.breaker.attempts = 2
        with tempfile.NamedTemporaryFile() as audio_file:
            audio_file.write(b"wav")
            audio_file.flush()
            with (
                mock.patch.object(
                    video_perception.urllib.request,
                    "urlopen",
                    side_effect=[
                        OSError("service is offline"),
                        _Response({"result": [{"clean_text": "recovered"}]}),
                    ],
                ) as urlopen,
                mock.patch.object(video_perception.time, "sleep") as sleep,
            ):
                self.assertEqual(video_perception._transcribe_openamika(audio_file.name), "recovered")

        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_not_called()

    def test_asr_retry_backoff_then_cooldown(self):
        self.breaker.attempts = 3
        self.breaker.delay_seconds = 0.2
        self.breaker.cooldown_seconds = 60.0
        with tempfile.NamedTemporaryFile() as audio_file:
            audio_file.write(b"wav")
            audio_file.flush()
            with (
                mock.patch("kookie_resilience.time.time", return_value=100),
                mock.patch.object(
                    video_perception.urllib.request,
                    "urlopen",
                    side_effect=OSError("service is offline"),
                ) as urlopen,
                mock.patch.object(video_perception.time, "sleep") as sleep,
            ):
                self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))

        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_has_calls([mock.call(0.2), mock.call(0.4)])
        self.assertEqual(self.breaker._down_until, 160)

    def test_asr_failure_threshold_requires_multiple_failures(self):
        self.breaker.threshold = 2
        self.breaker.cooldown_seconds = 60.0
        with (
            tempfile.NamedTemporaryFile() as audio_file,
            mock.patch("kookie_resilience.time.time", return_value=100),
            mock.patch.object(
                video_perception.urllib.request,
                "urlopen",
                side_effect=OSError("service is offline"),
            ) as urlopen,
        ):
            audio_file.write(b"wav")
            audio_file.flush()
            self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))
            self.assertEqual(self.breaker._down_until, 0.0)
            self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))
            self.assertEqual(self.breaker._down_until, 160)
            self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))

        self.assertEqual(urlopen.call_count, 2)

    def test_asr_cooldown_skips_repeated_failing_requests(self):
        self.breaker.cooldown_seconds = 60.0
        with (
            tempfile.NamedTemporaryFile() as audio_file,
            mock.patch("kookie_resilience.time.time", return_value=100),
            mock.patch.object(
                video_perception.urllib.request,
                "urlopen",
                side_effect=OSError("service is offline"),
            ) as urlopen,
        ):
            audio_file.write(b"wav")
            audio_file.flush()
            self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))
            self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))
        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(self.breaker._down_until, 160)

    def test_asr_is_retried_after_cooldown(self):
        self.breaker.threshold = 1
        self.breaker.cooldown_seconds = 60.0
        with tempfile.NamedTemporaryFile() as audio_file:
            audio_file.write(b"wav")
            audio_file.flush()
            # allow() reads the clock on every call, and record() reads it
            # again when the failure trips the breaker, so the sequence has
            # to cover both reads for both calls.
            with (
                mock.patch("kookie_resilience.time.time", side_effect=[100, 100, 161, 161]),
                mock.patch.object(
                    video_perception.urllib.request,
                    "urlopen",
                    side_effect=[
                        OSError("service is offline"),
                        _Response({"result": [{"clean_text": "recovered"}]}),
                    ],
                ) as urlopen,
            ):
                self.assertIsNone(video_perception._transcribe_openamika(audio_file.name))
                self.assertEqual(video_perception._transcribe_openamika(audio_file.name), "recovered")
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
