"""Focused prompt mode: maths questions get a lean rule set.

The full personality block is about 5,000 tokens of VTuber-daughter rules, and
on a maths question none of it changes the answer - it just eats the context
window the working steps need. Focused mode keeps the read-the-message rule
and the envelope rules, drops the personality, and adds a short "show your
work" instruction instead. These tests pin that contract in isolation.
"""

import unittest

import controller


class FocusedPromptTests(unittest.TestCase):
    def test_focused_prompt_is_much_smaller(self):
        full = controller.chat_system_prompt("hi", True, "neutral,happy")
        focused = controller.chat_system_prompt(
            "what is 123456789 * 987654321", True, "neutral,happy", focused=True
        )
        # The focused set is roughly a third of the full one - that is the
        # point of the change, not an exact ratio, so assert the direction and
        # a wide margin rather than a fixed number.
        self.assertLess(len(focused), len(full) // 2)

    def test_focused_prompt_keeps_the_maths_rule(self):
        focused = controller.chat_system_prompt(
            "solve 2x + 3 = 7", True, "neutral", focused=True
        )
        self.assertIn("MATHS", focused)

    def test_focused_prompt_drops_the_personality(self):
        focused = controller.chat_system_prompt(
            "what is 12 * 12", True, "neutral", focused=True
        )
        self.assertNotIn("VTuber daughter", focused)

    def test_full_prompt_still_has_personality(self):
        full = controller.chat_system_prompt("hi", True, "neutral")
        self.assertIn("VTuber daughter", full)

    def test_focused_prompt_keeps_the_envelope_rules(self):
        focused = controller.chat_system_prompt(
            "what is 12 * 12", True, "neutral", focused=True
        )
        self.assertIn("LEAVE THE ENVELOPE ALONE", focused)


if __name__ == "__main__":
    unittest.main()
