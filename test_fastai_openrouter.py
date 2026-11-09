import unittest
from unittest.mock import patch

import fastai


class _Response:
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop",
            }],
        }


class FastAIOpenRouterTests(unittest.TestCase):
    def test_static_fallback_model_ids_exist_in_current_catalog(self):
        openrouter = next(
            provider for provider in fastai.PROVIDERS
            if provider[0] == "OPENROUTER_API_KEY"
        )
        self.assertEqual(openrouter[2], (
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
        ))

    def test_openrouter_request_has_bearer_auth_and_attribution(self):
        provider = {
            "label": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "key": "unit-test-key",
        }
        with patch.object(fastai._SESSION, "post", return_value=_Response()) as post:
            result = fastai._ask(
                provider, "meta-llama/llama-3.3-70b-instruct",
                [{"role": "user", "content": "Hello"}],
                tools=None, tool_choice=None, max_tokens=40,
                temperature=0.4, top_p=0.9, timeout=3,
                stop=None, json_mode=False,
            )

        self.assertEqual(result["text"], "Hello!")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer unit-test-key",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["HTTP-Referer"],
            "https://github.com/Cookie-team-Dev/vtuber-ai",
        )
        self.assertEqual(post.call_args.kwargs["headers"]["X-Title"], "Kookie")
        self.assertEqual(
            post.call_args.args[0],
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(post.call_args.kwargs["json"]["model"],
                         "meta-llama/llama-3.3-70b-instruct")

    def test_provider_model_discovery_runs_once_for_multiple_keys(self):
        with (
            patch.object(fastai, "PROVIDERS", (("GROQ_API_KEY", "https://example.test", ("fallback",), "Groq"),)),
            patch.object(fastai, "keys_for", return_value=["key-1", "key-2"]),
            patch.object(fastai, "_key_for", return_value=""),
            patch.object(fastai, "_discover_models", return_value=("model-a",)) as discover,
        ):
            providers = fastai._providers()

        self.assertGreaterEqual(len(providers), 2)
        self.assertEqual(discover.call_count, 1)
        self.assertEqual([entry["models"] for entry in providers[:2]], [("model-a",)] * 2)

    def test_openrouter_model_discovery_uses_the_same_auth_and_attribution(self):
        with patch.object(fastai._SESSION, "get") as get:
            get.return_value = type(
                "DiscoveryResponse", (), {
                    "status_code": 200,
                    "json": staticmethod(lambda: {
                        "data": [
                            {
                                "id": "meta-llama/llama-3.1-8b-instruct",
                                "supported_parameters": [
                                    "tools", "response_format",
                                ],
                            },
                            {
                                "id": "meta-llama/llama-3.1-70b-instruct",
                                "supported_parameters": [
                                    "max_tokens", "temperature",
                                ],
                            },
                        ],
                    }),
                },
            )()
            discovered = fastai._fetch_provider_models(
                "https://openrouter.ai/api/v1", "unit-test-key",
            )

        self.assertEqual(discovered, ("meta-llama/llama-3.1-8b-instruct",))
        headers = get.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer unit-test-key")
        self.assertEqual(
            headers["HTTP-Referer"],
            "https://github.com/Cookie-team-Dev/vtuber-ai",
        )
        self.assertEqual(headers["X-Title"], "Kookie")


if __name__ == "__main__":
    unittest.main()
