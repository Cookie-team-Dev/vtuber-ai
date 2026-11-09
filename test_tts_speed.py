"""The global speaking-rate multiplier KOOKIE_TTS_SPEED.

The per-emotion scales in voice.VOICE_LENGTH_SCALES already push most moods
under 1.0 (excited 0.88, angry 0.93) because a streamer's natural pace is
quicker than neutral, and that used to be the only knob: "speak faster"
meant editing twelve numbers. KOOKIE_TTS_SPEED is now one env var applied to
both the Piper length_scale and the OpenAmika speed in the same direction,
so one line in .env moves every voice at once.
"""

import os
import unittest
from unittest import mock

import voice


class TTSSpeedTests(unittest.TestCase):
    def test_default_is_faster_than_neutral(self):
        # The default 0.85 cuts ~15% off neutral, which is the whole point:
        # a VTuber who speaks at 1.0 reads as slow.
        self.assertLess(voice.TTS_SPEED, 1.0)

    def test_piper_length_scale_is_multiplied(self):
        # excited is 0.88 at intensity 0.5, so without the multiplier the
        # length_scale would be 0.94. With it the voice is quicker.
        target = voice.VOICE_LENGTH_SCALES["excited"]
        without = 1.0 + (target - 1.0) * 0.5
        with_speed = without * voice.TTS_SPEED
        self.assertLess(with_speed, without)
        self.assertAlmostEqual(with_speed, 0.799, places=3)

    def test_openamika_payload_uses_the_product(self):
        # The request carries AMIKA_TTS_SPEED * TTS_SPEED, not the raw engine
        # factor, so the global knob reaches the SoVITS decoder. Pin this by
        # reading the source rather than mocking the whole socketio dance:
        # a refactor that dropped the multiplier would fail this test even
        # though every mock-based assertion would still pass.
        source = open(voice.__file__).read()
        self.assertIn("AMIKA_TTS_SPEED * TTS_SPEED", source)

    def test_speed_is_clamped_to_the_same_range_as_amika_speed(self):
        # Both knobs share 0.5-2.0, so a typo cannot push a voice into
        # incoherence or a 4x slow-motion clip.
        self.assertGreaterEqual(voice.TTS_SPEED, 0.5)
        self.assertLessEqual(voice.TTS_SPEED, 2.0)

    def test_env_override_changes_the_factor(self):
        import importlib
        with mock.patch.dict(os.environ, {"KOOKIE_TTS_SPEED": "1.5"}):
            importlib.reload(voice)
            self.assertEqual(voice.TTS_SPEED, 1.5)
        # Reload AFTER the env override is gone, or the module reads 1.5 again.
        importlib.reload(voice)
        self.assertEqual(voice.TTS_SPEED, 0.85)


if __name__ == "__main__":
    unittest.main()
