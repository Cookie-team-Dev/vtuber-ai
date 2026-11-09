import tempfile
import unittest
from pathlib import Path
from unittest import mock

import amika_tts_codes
import voice
from openamika.speech.po_contract import VALID_EMOTIONS


class AmikaTtsCodeTests(unittest.TestCase):
    def test_emotion_tags_split_text_and_map_to_piper_profiles(self):
        segments = amika_tts_codes.parse_speech_codes(
            "Hi. 【casual_very_happy】Yay! [whisper_sad]Softly."
        )

        self.assertEqual(
            segments,
            [
                ("text", "Hi. ", "neutral"),
                ("text", "Yay! ", "casual_very_happy"),
                ("text", "Softly.", "whisper_sad"),
            ],
        )
        self.assertEqual(amika_tts_codes.piper_emotion("casual_very_happy"), "excited")
        self.assertEqual(amika_tts_codes.piper_emotion("whisper_sad"), "sad")
        self.assertEqual(amika_tts_codes.piper_emotion("polite_smile"), "happy")

    def test_every_contract_emotion_maps_to_a_supported_profile(self):
        supported_profiles = set(voice.VOICE_LENGTH_SCALES)
        mapped_profiles = {amika_tts_codes.piper_emotion(tag) for tag in VALID_EMOTIONS}

        self.assertLessEqual(mapped_profiles, supported_profiles)
        self.assertGreater(len(mapped_profiles), 5)

    def test_model_voice_emotion_is_normalized_to_contract_key(self):
        self.assertEqual(
            amika_tts_codes.normalize_voice_emotion("whisper_angry"),
            "whisper_worried",
        )
        self.assertEqual(
            amika_tts_codes.normalize_voice_emotion("happy"),
            "casual_happy",
        )

    def test_sfx_and_control_tags_are_not_spoken(self):
        self.assertEqual(
            amika_tts_codes.parse_speech_codes(
                "Wait. 【Sigh】【想想】[register_emotion]Okay."
            ),
            [
                ("text", "Wait. ", "neutral"),
                ("sfx", "Sigh", "neutral"),
                ("text", "Okay.", "neutral"),
            ],
        )

    def test_unknown_square_bracket_prose_is_preserved(self):
        self.assertEqual(
            amika_tts_codes.parse_speech_codes("Keep [this note] here."),
            [("text", "Keep [this note] here.", "neutral")],
        )

    def test_sfx_lookup_accepts_known_assets_but_not_arbitrary_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "ため息_SFX_2.wav"
            asset.touch()
            with (
                mock.patch.object(amika_tts_codes, "SFX_DIRECTORIES", (Path(directory),)),
                mock.patch.object(amika_tts_codes, "_SFX_FILE_CACHE", None),
            ):
                self.assertEqual(amika_tts_codes.find_sfx_file("Sigh"), asset)
                self.assertIsNone(amika_tts_codes.find_sfx_file("../../ため息"))

    def test_voice_speaks_segments_with_their_individual_emotions(self):
        with (
            mock.patch.object(voice, "VOICE_ENABLED", True),
            mock.patch.object(voice, "_speak_text") as speak_text,
            mock.patch.object(voice, "find_sfx_file", return_value=None),
        ):
            voice.speak("Hi. 【casual_very_happy】Yay!")

        self.assertEqual(
            speak_text.call_args_list,
            [
                mock.call("Hi.", "neutral", 0.5, amika_emotion="neutral"),
                mock.call("Yay!", "excited", 0.5, amika_emotion="casual_very_happy"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
