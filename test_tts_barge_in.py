"""stop_speaking silences an in-flight line instead of raising.

The old body called interrupt(), which does not exist in this module, so every
barge-in raised NameError and Kookie finished the sentence anyway. It also
carried fragments of sing_song, speak_cue and say_greeting referencing
parameters no caller passes. These tests pin the contract in isolation: a
tracked playback process is killed, and a stop with nothing playing is a
no-op rather than an exception.
"""

import unittest

import voice


class StopSpeakingTests(unittest.TestCase):
    def setUp(self):
        voice._PLAYBACK_PROC = None

    def tearDown(self):
        voice._PLAYBACK_PROC = None
    def test_stop_with_nothing_playing_is_a_noop(self):
        # The first thing stop_speaking used to do was call interrupt(), which
        # is not defined here. A quiet moment must not raise.
        voice.stop_speaking()

    def test_stop_kills_a_tracked_playback(self):
        killed = {}

        class FakeProc:
            def kill(self):
                killed["killed"] = True

            def wait(self, timeout=None):
                return 0

        proc = FakeProc()
        voice._track_playback(proc)
        self.assertIs(voice._PLAYBACK_PROC, proc)
        voice.stop_speaking()
        self.assertTrue(killed.get("killed"))
        self.assertIsNone(voice._PLAYBACK_PROC)

    def test_playback_is_released_after_it_exits(self):
        class FakeProc:
            def wait(self, timeout=None):
                return 0

        proc = FakeProc()
        voice._track_playback(proc)
        voice._clear_playback(proc)
        self.assertIsNone(voice._PLAYBACK_PROC)
        # Clearing a different process must not touch the tracked one.
        voice._track_playback(proc)
        voice._clear_playback(FakeProc())
        self.assertIs(voice._PLAYBACK_PROC, proc)
        voice.stop_speaking()


if __name__ == "__main__":
    unittest.main()