"""The pipeline starts the first sentence before the rest is synthesized.

Splitting alone does not make her speak faster: what matters is that the first
WAV is played while the second is still being synthesized. These tests measure
that directly by giving the synthesis stub a real delay and comparing the
pipeline's time-to-first-audio against the time the whole line would have taken
sequentially.
"""

import os
import tempfile
import time
import unittest
from unittest import mock

import voice


class PipelineTimingTests(unittest.TestCase):
    def setUp(self):
        self._played_at = []
        voice._PLAYBACK_PROC = None
        voice._PIPELINE_STOP.clear()

    def tearDown(self):
        voice._PLAYBACK_PROC = None
        voice._PIPELINE_STOP.clear()

    def _make_synthesizer(self, delay):
        def fake_synthesize(sentence, emotion, intensity, amika_emotion):
            time.sleep(delay)
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            return path

        return fake_synthesize

    def _make_player(self):
        def fake_play(audio_path):
            self._played_at.append(time.monotonic())

        return fake_play

    def test_first_audio_starts_before_all_sentences_are_synthesized(self):
        # Two sentences, each taking 0.2s to synthesize. Sequentially that is
        # 0.4s before anything plays; the pipeline starts the first WAV the
        # moment it is ready, so the first audio lands around 0.2s.
        sentences = ["First sentence.", "Second sentence."]
        start = time.monotonic()
        with mock.patch.object(
            voice, "_synthesize_to_file",
            side_effect=self._make_synthesizer(0.2),
        ), mock.patch.object(
            voice, "_play_synthesized_audio",
            side_effect=self._make_player(),
        ):
            voice._speak_pipelined(sentences, "neutral", 0.5, "neutral")

        first_audio = self._played_at[0] - start
        total_synthesis = len(sentences) * 0.2
        # The first sentence started well before the whole line was synthesized.
        self.assertLess(first_audio, total_synthesis)
        # And both sentences still played, in order.
        self.assertEqual(len(self._played_at), len(sentences))

    def test_pipeline_does_not_wait_for_the_last_sentence_before_playing(self):
        sentences = [f"Sentence {index}." for index in range(4)]
        start = time.monotonic()
        with mock.patch.object(
            voice, "_synthesize_to_file",
            side_effect=self._make_synthesizer(0.1),
        ), mock.patch.object(
            voice, "_play_synthesized_audio",
            side_effect=self._make_player(),
        ):
            voice._speak_pipelined(sentences, "neutral", 0.5, "neutral")

        first_audio = self._played_at[0] - start
        # Sequential synthesis of four 0.1s sentences would be 0.4s before the
        # first sound. The pipeline overlaps, so it lands near one sentence.
        self.assertLess(first_audio, 0.3)


if __name__ == "__main__":
    unittest.main()
