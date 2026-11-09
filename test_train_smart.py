import tempfile
import unittest
from pathlib import Path

import train_smart


class _ChatTemplateTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class TrainSmartTests(unittest.TestCase):
    def test_prompt_completion_masks_prompt_and_keeps_assistant_template(self):
        row = {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ],
        }

        result = train_smart.build_prompt_completion(
            row, _ChatTemplateTokenizer(),
        )

        self.assertEqual(
            result,
            {
                "prompt": "<system>Be concise.</system>"
                          "<user>What is 2 + 2?</user><assistant>",
                "completion": "4</assistant>",
            },
        )

    def test_qwen_template_is_rendered_without_thinking_mode(self):
        class QwenTokenizer(_ChatTemplateTokenizer):
            chat_template = "enable_thinking is a supported template option"

            def __init__(self):
                self.options = []

            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    **kwargs):
                self.options.append(kwargs)
                return super().apply_chat_template(
                    messages, tokenize, add_generation_prompt)

        tokenizer = QwenTokenizer()
        train_smart.build_prompt_completion(
            {
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "4"},
                ],
            },
            tokenizer,
        )

        self.assertEqual(tokenizer.options, [
            {"enable_thinking": False},
            {"enable_thinking": False},
        ])

    def test_prompt_completion_rejects_rows_without_final_assistant(self):
        with self.assertRaisesRegex(ValueError, "end with an assistant"):
            train_smart.build_prompt_completion(
                {"messages": [{"role": "user", "content": "Hello"}]},
                _ChatTemplateTokenizer(),
            )

    def test_prompt_completion_rejects_non_prefix_chat_templates(self):
        class NonPrefixTokenizer(_ChatTemplateTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                result = super().apply_chat_template(
                    messages, tokenize, add_generation_prompt)
                return result if not messages or messages[-1]["role"] == "assistant" else "changed"

        row = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "does not preserve the prompt prefix"):
            train_smart.build_prompt_completion(row, NonPrefixTokenizer())

    def test_strict_jsonl_loading_reports_malformed_rows_and_line_numbers(self):
        with tempfile.TemporaryDirectory(prefix="kookie-training-test-") as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text('{"messages": []}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"rows\.jsonl at line 2"):
                train_smart.load_rows(str(path), strict=True)

    def test_dataset_validation_rejects_content_leakage_across_system_variants(self):
        user = {"role": "user", "content": "What is 2 + 2?"}
        assistant = {"role": "assistant", "content": "4"}
        train = [{"messages": [
            {"role": "system", "content": "Be concise."}, user, assistant,
        ]}]
        evaluation = [{"messages": [
            {"role": "system", "content": "Use a playful voice."},
            user, assistant,
        ]}]

        validation = train_smart.validate_dataset(train, evaluation)

        self.assertEqual(validation["shared_content_groups"], 1)
        self.assertTrue(train_smart._dataset_validation_failed(validation))

    def test_dataset_validation_accepts_distinct_valid_examples(self):
        train = [{"messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]}]
        evaluation = [{"messages": [
            {"role": "user", "content": "What is 3 + 3?"},
            {"role": "assistant", "content": "6"},
        ]}]

        validation = train_smart.validate_dataset(train, evaluation)

        self.assertEqual(validation["shared_content_groups"], 0)
        self.assertFalse(train_smart._dataset_validation_failed(validation))


if __name__ == "__main__":
    unittest.main()
