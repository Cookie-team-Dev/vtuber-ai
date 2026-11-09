"""Pipelined speak path: play sentence N while synthesizing sentence N+1.

The old path synthesized the whole reply, then played it, so time-to-first-audio
was the full length of the TTS call. The pipeline keeps a prefetch thread ahead
of the player so the first WAV starts as soon as it is ready. These tests pin
the contract in isolation: the splitter protects numbers, the pipeline plays in
order and cleans up, and a barge-in halts both the player and the prefetch
thread via the shared stop flag.
"""

import os
import tempfile
import threading
import unittest
from unittest import mock

import voice


class SplitSentencesTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(voice._split_into_sentences(""), [])
        self.assertEqual(voice._split_into_sentences("   "), [])

    def test_single_sentence(self):
        self.assertEqual(voice._split_into_sentences("hello world"), ["hello world"])

    def test_simple_split(self):
        self.assertEqual(
            voice._split_into_sentences("One. Two! Three?"),
            ["One.", "Two!", "Three?"],
        )

    def test_decimal_is_preserved(self):
        # A naive split on [.!?] would turn "22.5 degrees" into "22" and
        # "5 degrees", cutting every number in half.
        result = voice._split_into_sentences("It is 22.5 degrees today.")
        self.assertEqual(result, ["It is 22.5 degrees today."])

    def test_abbreviations_are_preserved(self):
        result = voice._split_into_sentences("Dr. Smith met Mr. Jones at 3 p.m.")
        self.assertEqual(len(result), 1)
        self.assertIn("Dr. Smith", result[0])
        self.assertIn("Mr. Jones", result[0])

    def test_commas_and_apostrophes_survive(self):
        result = voice._split_into_sentences("It's warm, isn't it?")
        self.assertEqual(result, ["It's warm, isn't it?"])


class PipelinedSpeakTests(unittest.TestCase):
    def setUp(self):
        self._played = []
        self._synthesized = []
        voice._PLAYBACK_PROC = None
        voice._PIPELINE_STOP.clear()

    def tearDown(self):
        voice._PLAYBACK_PROC = None
        voice._PIPELINE_STOP.clear()

    def _stub_synthesis(self, sentences):
        def fake_synthesize_to_file(sentence, emotion, intensity, amika_emotion):
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self._synthesized.append((sentence, path))
            return path

        return fake_synthesize_to_file

    def _stub_player(self):
        def fake_play(audio_path):
            self._played.append(audio_path)

        return fake_play

    def test_pipeline_plays_in_order_and_cleans_up(self):
        sentences = ["First sentence here.", "Second one too.", "And a third."]
        with mock.patch.object(
            voice, "_synthesize_to_file", side_effect=self._stub_synthesis(sentences)
        ), mock.patch.object(
            voice, "_play_synthesized_audio", side_effect=self._stub_player()
        ):
            voice._speak_pipelined(sentences, "neutral", 0.5, "neutral")

        # Every sentence was synthesized and played, in order, and the temp
        # files are gone because the player path unlinks them.
        self.assertEqual([s for s, _ in self._synthesized], sentences)
        self.assertEqual(len(self._played), len(sentences))
        for _, path in self._synthesized:
            self.assertFalse(os.path.exists(path))

    def test_pipeline_stops_early_on_barge_in(self):
        sentences = ["First.", "Second.", "Third."]

        def slow_play(audio_path):
            self._played.append(audio_path)
            # Simulate a long playback: the barge-in kills the player and sets
            # the pipeline stop flag, so the loop halts before the next sentence.
            class FakeProc:
                def kill(self):
                    pass

                def wait(self, timeout=None):
                    return 0

            voice._track_playback(FakeProc())
            voice._PIPELINE_STOP.set()

        with mock.patch.object(
            voice, "_synthesize_to_file", side_effect=self._stub_synthesis(sentences)
        ), mock.patch.object(
            voice, "_play_synthesized_audio", side_effect=slow_play
        ):
            voice._speak_pipelined(sentences, "neutral", 0.5, "neutral")

        # The first sentence started playing and the barge-in fired; the rest
        # of the line was discarded rather than played out.
        self.assertGreaterEqual(len(self._played), 1)
        self.assertLess(len(self._played), len(sentences))

    def test_pipeline_with_no_sentences_is_a_noop(self):
        voice._speak_pipelined([], "neutral", 0.5, "neutral")
        self.assertEqual(self._played, [])
        self.assertEqual(self._synthesized, [])

    def test_pipeline_survives_a_synthesis_failure(self):
        sentences = ["First.", "Second.", "Third."]

        def failing_synthesize(sentence, emotion, intensity, amika_emotion):
            if sentence == "Second.":
                return None
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self._synthesized.append((sentence, path))
            return path

        with mock.patch.object(
            voice, "_synthesize_to_file", side_effect=failing_synthesize
        ), mock.patch.object(
            voice, "_play_synthesized_audio", side_effect=self._stub_player()
        ):
            voice._speak_pipelined(sentences, "neutral", 0.5, "neutral")

        # The failed sentence is skipped, not crashed; the others play.
        self.assertEqual(len(self._played), 2)


if __name__ == "__main__":
    unittest.main()
