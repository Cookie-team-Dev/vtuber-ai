import io
import json
import unittest
import urllib.error
from unittest import mock

import fastai
import omniroute


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}'


class OmniRouteTests(unittest.TestCase):
    def test_chat_uses_omniroute_when_no_other_provider_is_configured(self):
        expected = {"text": "hello", "tool_calls": None, "finish_reason": "stop"}
        with (
            mock.patch.object(fastai, "_providers", return_value=[]),
            mock.patch.object(fastai.omniroute, "chat", return_value=expected) as chat,
        ):
            result = fastai.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, expected)
        chat.assert_called_once()

    def test_chat_sends_the_configured_bearer_key(self):
        with (
            mock.patch.object(omniroute, "ensure_available", return_value=True),
            mock.patch.object(omniroute, "OMNIROUTE_API_KEY", "unit-test-key"),
            mock.patch.object(
                omniroute.urllib.request, "urlopen", return_value=_Response()
            ) as urlopen,
        ):
            result = omniroute.chat([{"role": "user", "content": "hi"}])

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-key")
        self.assertEqual(result["text"], "hello")
        self.assertIsNone(result["tool_calls"])

    def test_chat_forwards_tool_contract_and_normalizes_tool_arguments(self):
        response = _Response()
        response.read = lambda: (
            b'{"choices":[{"message":{"content":"","tool_calls":[{"id":"call-1",'
            b'"function":{"name":"browse","arguments":"{\\"query\\":\\"weather\\"}"}}]},'
            b'"finish_reason":"tool_calls"}]}'
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "browse",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }]
        with (
            mock.patch.object(omniroute, "ensure_available", return_value=True),
            mock.patch.object(omniroute.urllib.request, "urlopen", return_value=response) as urlopen,
        ):
            result = omniroute.chat(
                [{"role": "user", "content": "weather"}],
                tools=tools,
                tool_choice="auto",
                stop=["END"],
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["stop"], ["END"])
        self.assertEqual(
            result["tool_calls"],
            [{"function": {"name": "browse", "arguments": {"query": "weather"}}}],
        )
        self.assertEqual(result["finish_reason"], "tool_calls")

    def test_direct_omniroute_path_forwards_tool_arguments(self):
        expected = {"text": "", "tool_calls": [], "finish_reason": "stop"}
        tools = [{"type": "function", "function": {"name": "browse"}}]
        with (
            mock.patch.object(fastai, "_providers", return_value=[]),
            mock.patch.object(fastai.omniroute, "chat", return_value=expected) as chat,
        ):
            result = fastai.chat(
                [{"role": "user", "content": "search"}],
                tools=tools,
                tool_choice="auto",
                stop=["END"],
            )

        self.assertEqual(result, expected)
        self.assertEqual(chat.call_args.kwargs["tools"], tools)
        self.assertEqual(chat.call_args.kwargs["tool_choice"], "auto")
        self.assertEqual(chat.call_args.kwargs["stop"], ["END"])

    def test_omniroute_fallback_forwards_tool_arguments(self):
        expected = {"text": "", "tool_calls": [], "finish_reason": "stop"}
        tools = [{"type": "function", "function": {"name": "browse"}}]
        provider = {
            "env_var": "GEMINI_API_KEY",
            "key_number": 1,
            "key_count": 1,
            "label": "Gemini",
            "slot": "GEMINI_API_KEY#1",
            "models": (),
        }
        with (
            mock.patch.object(fastai, "_providers", return_value=[provider]),
            mock.patch.object(
                fastai, "_chat_one_key", side_effect=fastai.FastAIError("offline")
            ),
            mock.patch.object(fastai.omniroute, "chat", return_value=expected) as chat,
        ):
            result = fastai.chat(
                [{"role": "user", "content": "search"}],
                tools=tools,
                tool_choice="auto",
                stop=["END"],
            )

        self.assertEqual(result, expected)
        self.assertEqual(chat.call_args.kwargs["tools"], tools)
        self.assertEqual(chat.call_args.kwargs["tool_choice"], "auto")
        self.assertEqual(chat.call_args.kwargs["stop"], ["END"])

    def test_connection_status_reports_authentication_requirement(self):
        error = urllib.error.HTTPError(
            "http://localhost:20128/v1/models", 401, "Unauthorized", {}, io.BytesIO()
        )
        with (
            mock.patch.object(omniroute, "ensure_available", return_value=True),
            mock.patch.object(
                omniroute.urllib.request, "urlopen", side_effect=error
            ),
        ):
            status = omniroute.connection_status()

        self.assertEqual(status, "authentication required")

    def test_running_check_uses_configured_authentication(self):
        with (
            mock.patch.object(omniroute, "OMNIROUTE_API_KEY", "unit-test-key"),
            mock.patch.object(
                omniroute.urllib.request, "urlopen", return_value=_Response()
            ) as urlopen,
        ):
            running = omniroute._is_running()

        request = urlopen.call_args.args[0]
        self.assertTrue(running)
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-key")



    def test_chat_retries_transient_http_5xx(self):
        error = urllib.error.HTTPError(
            "http://localhost:20128/v1/chat/completions",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b"down"),
        )
        with (
            mock.patch.object(omniroute, "ensure_available", return_value=True),
            mock.patch.object(omniroute, "OMNIROUTE_RETRY_ATTEMPTS", 2),
            mock.patch.object(omniroute, "OMNIROUTE_RETRY_DELAY_SECONDS", 0.0),
            mock.patch.object(
                omniroute.urllib.request, "urlopen", side_effect=[error, _Response()]
            ) as urlopen,
            mock.patch.object(omniroute.time, "sleep") as sleep,
        ):
            result = omniroute.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(result["text"], "hello")
        sleep.assert_not_called()

    def test_chat_stops_retrying_on_permanent_auth_failure(self):
        error = urllib.error.HTTPError(
            "http://localhost:20128/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"bad key"),
        )
        with (
            mock.patch.object(omniroute, "ensure_available", return_value=True),
            mock.patch.object(omniroute, "OMNIROUTE_RETRY_ATTEMPTS", 3),
            mock.patch.object(
                omniroute.urllib.request, "urlopen", side_effect=error
            ) as urlopen,
            mock.patch.object(omniroute.time, "sleep") as sleep,
        ):
            with self.assertRaises(Exception):
                omniroute.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_chat_uses_exponential_backoff_between_transient_attempts(self):
        error = urllib.error.HTTPError(
            "http://localhost:20128/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b"slow down"),
        )
        with (
            mock.patch.object(omniroute, "ensure_available", return_value=True),
            mock.patch.object(omniroute, "OMNIROUTE_RETRY_ATTEMPTS", 3),
            mock.patch.object(omniroute, "OMNIROUTE_RETRY_DELAY_SECONDS", 0.2),
            mock.patch.object(
                omniroute.urllib.request, "urlopen", side_effect=error
            ) as urlopen,
            mock.patch.object(omniroute.time, "sleep") as sleep,
        ):
            with self.assertRaises(Exception):
                omniroute.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_has_calls([mock.call(0.2), mock.call(0.4)])


if __name__ == "__main__":
    unittest.main()
