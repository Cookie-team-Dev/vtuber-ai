import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import training_data


class _StructuredTeacher:
    def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        if "four labeled lines" in system:
            reply = (
                "First question: What does an API do?\n"
                "First answer: It lets one program request data from another.\n"
                "Follow-up question: Is that how you talk to Gemini?\n"
                "Follow-up answer: Yes, an API carries my request and brings its response back."
            )
        else:
            reply = "What does a compiler do?\nWhy do errors happen?"
        return {"text": json.dumps({"reply": reply})}


class TrainingDataTests(unittest.TestCase):
    def test_teacher_multiturn_keeps_both_contextual_exchange_answers(self):
        exchange = (
            "What does an API do?",
            "It lets one program request data or actions from another.",
            "Is that how you talk to Gemini?",
            "Yes, the API carries my request to Gemini and brings its response back.",
        )

        rows = training_data.build_multiturn_from_teacher([exchange])

        self.assertEqual(len(rows), 1)
        messages = rows[0]["messages"]
        self.assertEqual(
            [(message["role"], message["content"]) for message in messages[-4:]],
            [
                ("user", exchange[0]),
                ("assistant", exchange[1]),
                ("user", exchange[2]),
                ("assistant", exchange[3]),
            ],
        )

    def test_teacher_multiturn_parser_preserves_validated_context(self):
        reply = (
            "First question: What does an API do?\n"
            "First answer: It lets one program request data from another.\n"
            "Follow-up question: Is that how you talk to Gemini?\n"
            "Follow-up answer: Yes, an API carries my request to Gemini and brings its response back."
        )
        expected = [(
            "What does an API do?",
            "It lets one program request data from another.",
            "Is that how you talk to Gemini?",
            "Yes, an API carries my request to Gemini and brings its response back.",
        )]
        with patch(
            "training_data._teacher_rows",
            return_value=([{"reply": reply}], {"ok": 1}),
        ):
            pairs, stats = training_data._multiturn_from_teacher("unused.jsonl")

        self.assertEqual(pairs, expected)
        self.assertEqual(stats, {"ok": 1})

    def test_teacher_multiturn_parser_drops_incomplete_exchange(self):
        with patch(
            "training_data._teacher_rows",
            return_value=([{"reply": "Follow-up question: What about it?"}], {}),
        ):
            pairs, _ = training_data._multiturn_from_teacher("unused.jsonl")

        self.assertEqual(pairs, [])

    def test_teacher_question_parser_extracts_questions_from_reply_lines(self):
        with patch(
            "training_data._teacher_rows",
            return_value=([{
                "reply": "1. What does a compiler do?\n- Why do errors happen?",
            }], {"ok": 1}),
        ):
            questions, stats = training_data._teacher_questions(
                ["programming"], "unused.jsonl", per_topic=2,
            )

        self.assertEqual(
            questions,
            ["What does a compiler do?", "Why do errors happen?"],
        )
        self.assertEqual(stats["questions"], 2)

    def test_teacher_generators_use_the_shared_reply_schema_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="kookie-training-test-") as temp:
            teacher = _StructuredTeacher()
            questions, _ = training_data._teacher_questions(
                ["programming"], str(Path(temp) / "questions.jsonl"),
                per_topic=2, teacher=teacher,
            )
            pairs, _ = training_data._multiturn_from_teacher(
                str(Path(temp) / "multiturn.jsonl"), teacher=teacher, limit=1,
            )

        self.assertEqual(
            questions,
            ["What does a compiler do?", "Why do errors happen?"],
        )
        self.assertEqual(pairs, [(
            "What does an API do?",
            "It lets one program request data from another.",
            "Is that how you talk to Gemini?",
            "Yes, an API carries my request and brings its response back.",
        )])

    def test_split_keeps_variants_of_one_exchange_out_of_both_splits(self):
        messages = [
            {"role": "system", "content": "Prompt variant A"},
            {"role": "user", "content": "What does an API do?"},
            {"role": "assistant", "content": "It lets programs request data."},
        ]
        repeated = [
            {"role": "system", "content": "Prompt variant B"},
            *messages[1:],
        ]

        train, evaluation = training_data.split_rows(
            [{"messages": messages}, {"messages": repeated}],
            eval_share=1.0,
        )

        self.assertEqual(train, [])
        self.assertEqual(len(evaluation), 1)


if __name__ == "__main__":
    unittest.main()
