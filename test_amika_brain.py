import unittest
from unittest.mock import patch

from openamika.brain.methods import MemoryMethod, tokenize
from amika_brain import retrieve_relevant_notes
import controller


class AmikaBrainTests(unittest.TestCase):
    def test_openamika_retrieval_finds_relevant_kookie_memory(self):
        entries = [
            {"text": "User: My cat's name is Miso.", "time": 1},
            {"text": "User: I like rainy evenings.", "time": 2},
            {"text": "AI: We talked about my keyboard settings.", "time": 3},
        ]

        notes = retrieve_relevant_notes("What is my cat's name?", entries)

        self.assertEqual(len(notes), 1)
        self.assertIn("Miso", notes[0])

    def test_excludes_recent_notes_and_returns_no_unmatched_noise(self):
        entries = [
            {"text": "User: My cat's name is Miso.", "time": 1},
            {"text": "User: I like rainy evenings.", "time": 2},
        ]

        self.assertEqual(
            retrieve_relevant_notes(
                "What is my cat's name?",
                entries,
                exclude_texts=["User: My cat's name is Miso."],
            ),
            [],
        )
        self.assertEqual(retrieve_relevant_notes("zzqv xwrt", entries), [])

    def test_reuses_index_for_unchanged_notes_and_invalidates_changed_notes(self):
        entries = [
            {"text": "User: The observatory passphrase is cobalt-orchid.", "time": 1700000001},
            {"text": "User: The garden is full of lavender.", "time": 1700000002},
        ]
        with patch("amika_brain.MemoryMethod", wraps=MemoryMethod) as factory:
            first = retrieve_relevant_notes("What is the observatory passphrase?", entries)
            second = retrieve_relevant_notes("What is in the garden?", entries)

            self.assertEqual(factory.call_count, 1)
            self.assertIn("cobalt-orchid", first[0])
            self.assertIn("lavender", second[0])

            entries[0]["text"] = "User: The observatory passphrase is amber-comet."
            updated = retrieve_relevant_notes("What is the observatory passphrase?", entries)

            self.assertEqual(factory.call_count, 2)
            self.assertIn("amber-comet", updated[0])
            self.assertNotIn("cobalt-orchid", updated[0])

    def test_appending_note_extends_the_existing_lexical_index(self):
        entries = [
            {"text": "User: My telescope is named Northstar.", "time": 1700000011},
            {"text": "User: My notebook is named Fieldbook.", "time": 1700000012},
        ]
        with patch("openamika.brain.methods.tokenize", wraps=tokenize) as tokenizer:
            retrieve_relevant_notes("What is my telescope named?", entries)
            self.assertEqual(tokenizer.call_count, len(entries) + 1)

            entries.append({"text": "User: My camera is named Starlight.", "time": 1700000013})
            result = retrieve_relevant_notes("What is my camera named?", entries)

            self.assertIn("Starlight", result[0])
            self.assertEqual(tokenizer.call_count, len(entries) + 2)

    def test_controller_adds_secondary_recall_to_kookies_memory_context(self):
        entries = [
            {"text": "User: My cat's name is Miso.", "time": 1},
            {"text": "User: I opened the editor this morning.", "time": 2},
        ]

        context = controller.build_memory_context(
            "What is my cat's name?",
            entries,
            "today",
            recent_limit=1,
            char_budget=1800,
        )

        self.assertIn("Second-brain cross-check", context)
        self.assertIn("Miso", context)


if __name__ == "__main__":
    unittest.main()
