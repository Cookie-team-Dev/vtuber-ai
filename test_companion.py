"""Tests for the rewritten companion module."""

import unittest
from unittest import mock

import companion


class CompanionBehaviourTests(unittest.TestCase):
    def test_greet_uses_name(self):
        self.assertEqual(companion.greet("Mommy"), "Hi Mommy! How can I help?")

    def test_greet_defaults_to_mommy(self):
        self.assertIn("Mommy", companion.greet())

    def test_greet_handles_empty_name(self):
        self.assertIn("Mommy", companion.greet(""))

    def test_is_ready_checks_memory(self):
        with mock.patch("companion._memory.load_memory", return_value=[]):
            self.assertTrue(companion.is_ready())

    def test_is_ready_false_when_memory_unreadable(self):
        with mock.patch("companion._memory.load_memory", side_effect=OSError("nope")):
            self.assertFalse(companion.is_ready())

    def test_remind_delegates_to_controller(self):
        with mock.patch("controller.set_reminder", return_value="set") as m:
            result = companion.remind("buy milk", 30)
        m.assert_called_once_with("buy milk", 30)
        self.assertEqual(result, "set")

    def test_remind_falls_back_to_memory(self):
        with mock.patch("controller.set_reminder", side_effect=ImportError), \
             mock.patch("companion._memory.remember") as m:
            result = companion.remind("buy milk", 30)
        m.assert_called_once()
        self.assertIn("Remembered", result)

    def test_remind_rejects_blank(self):
        self.assertEqual(companion.remind("   "), "Error: remind needs some text")

    def test_summarize_truncates(self):
        text = "x" * 200
        result = companion.summarize(text, max_chars=20)
        self.assertLessEqual(len(result), 21)
        self.assertTrue(result.endswith("\u2026"))

    def test_summarize_empty(self):
        self.assertEqual(companion.summarize(""), "")

    def test_translate_is_passthrough(self):
        self.assertEqual(companion.translate("hola", "es"), "hola")

    def test_mood_boost_validates_vocabulary(self):
        with mock.patch("companion._memory.canonical_mood", return_value="happy", create=True):
            self.assertEqual(companion.mood_boost("HAPPY"), "feeling happy")

    def test_energy_level_high_when_just_active(self):
        with mock.patch("companion._memory.format_time_ago", return_value="just now"):
            self.assertEqual(companion.energy_level(), "high")

    def test_energy_level_low_when_idle(self):
        with mock.patch("companion._memory.format_time_ago", return_value="5 days ago"):
            self.assertEqual(companion.energy_level(), "low")

    def test_status_reports_readiness(self):
        with mock.patch("companion.is_ready", return_value=True), \
             mock.patch("companion._memory.get_mood_state", return_value={"mood": "happy"}):
            self.assertIn("awake", companion.status())
            self.assertIn("happy", companion.status())

    def test_farewell_uses_name(self):
        self.assertEqual(companion.farewell("Mommy"), "Bye Mommy! Talk soon.")

    def test_action_verbs_are_strings(self):
        for name in ("cheer", "comfort", "focus", "relax", "celebrate",
                     "encourage", "welcome", "ping", "thanks", "apologize"):
            self.assertIsInstance(getattr(companion, name)(), str)

    def test_now_iso_uses_core_utils(self):
        value = companion.now_iso()
        self.assertEqual(len(value), 20)
        self.assertEqual(value[10], "T")

    def test_sleep_for_caps_at_30(self):
        with mock.patch("companion.time.sleep") as m:
            companion.sleep_for(999)
        m.assert_called_once_with(30.0)


if __name__ == "__main__":
    unittest.main()