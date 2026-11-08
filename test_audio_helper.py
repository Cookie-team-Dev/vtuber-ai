"""Tests for the audio helper surface.

audio_helper reaches for voice.py and controller lazily, so these tests mock
the two entry points rather than spawning subprocesses or loading the Piper
model.
"""

import unittest
from unittest import mock

import audio_helper


class AudioHelperTests(unittest.TestCase):
    def setUp(self):
        # Reset the module-level volume between tests.
        audio_helper._VOLUME = 80

    def test_play_returns_player_command_for_existing_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        try:
            with mock.patch.object(audio_helper._voice, "audio_player",
                                   return_value=["paplay", path]) as player:
                result = audio_helper.play(path)
            self.assertEqual(result, ["paplay", path])
            player.assert_called_once_with(path)
        finally:
            import os
            os.unlink(path)

    def test_play_rejects_missing_path(self):
        with mock.patch.object(audio_helper._voice, "audio_player") as player:
            result = audio_helper.play("/tmp/does-not-exist-xyz.wav")
        self.assertTrue(result.startswith("Error:"))
        player.assert_not_called()

    def test_play_rejects_non_string(self):
        self.assertTrue(audio_helper.play(None).startswith("Error:"))

    def test_stop_asks_voice_to_halt(self):
        with mock.patch.object(audio_helper._voice, "stop_speaking") as stop:
            self.assertTrue(audio_helper.stop())
        stop.assert_called_once()

    def test_stop_returns_false_when_voice_raises(self):
        with mock.patch.object(audio_helper._voice, "stop_speaking",
                               side_effect=RuntimeError("boom")):
            self.assertFalse(audio_helper.stop())

    def test_volume_defaults_to_80(self):
        self.assertEqual(audio_helper.volume(), 80)

    def test_set_volume_clamps_low(self):
        self.assertEqual(audio_helper.set_volume(-50), 0)
        self.assertEqual(audio_helper.volume(), 0)

    def test_set_volume_clamps_high(self):
        self.assertEqual(audio_helper.set_volume(9999), 100)
        self.assertEqual(audio_helper.volume(), 100)

    def test_set_volume_accepts_string(self):
        self.assertEqual(audio_helper.set_volume("42"), 42)

    def test_set_volume_falls_back_on_bad_input(self):
        self.assertEqual(audio_helper.set_volume("not a number"), 80)

    def test_is_enabled_delegates_to_voice_enabled(self):
        with mock.patch.object(audio_helper, "_voice_enabled", return_value=True):
            self.assertTrue(audio_helper.is_enabled())
        with mock.patch.object(audio_helper, "_voice_enabled", return_value=False):
            self.assertFalse(audio_helper.is_enabled())

    def test_voice_enabled_falls_back_to_voice_flag(self):
        # controller.voice_enabled raises -> fall back to voice.VOICE_ENABLED.
        with mock.patch("controller.voice_enabled", side_effect=ImportError), \
             mock.patch.object(audio_helper._voice, "VOICE_ENABLED", True):
            self.assertTrue(audio_helper._voice_enabled())

    def test_voice_enabled_returns_false_when_everything_raises(self):
        with mock.patch("controller.voice_enabled", side_effect=ImportError), \
             mock.patch.object(audio_helper._voice, "VOICE_ENABLED", False):
            self.assertFalse(audio_helper._voice_enabled())

    def test_status_reports_enabled_and_volume(self):
        with mock.patch.object(audio_helper, "_voice_enabled", return_value=True):
            snap = audio_helper.status()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["volume"], 80)


if __name__ == "__main__":
    unittest.main()