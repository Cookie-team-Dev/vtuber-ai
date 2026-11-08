"""Tests for the retry behaviour behind ``controller.http_request``.

The shared ``kookie_resilience.http_call`` does the real work, but this file
makes sure the controller wires it up correctly: public-host filtering, URL
normalisation, the truncation cap, and the fact that a transient failure is
retried before the caller ever sees an error.
"""

import unittest
from unittest import mock

import controller


class _FakeResponse:
    def __init__(self, status_code=200, content=b"hello world", content_type="text/plain"):
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self._content = content

    def iter_content(self, chunk_size=8192):
        data = self._content
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]


class HttpRequestTests(unittest.TestCase):
    def setUp(self):
        self._session_patcher = mock.patch("controller._HTTP_SESSION")
        self.mock_session = self._session_patcher.start()
        self.addCleanup(self._session_patcher.stop)
        self.mock_session.request.return_value = _FakeResponse()

    def test_plain_url_gets_scheme(self):
        controller.http_request("example.com/page")
        call = self.mock_session.request.call_args
        self.assertEqual(call.args[0], "GET")
        self.assertEqual(call.args[1], "https://example.com/page")

    def test_rejects_private_host(self):
        result = controller.http_request("http://127.0.0.1/secret")
        self.assertIn("not fetched", result)
        self.assertFalse(self.mock_session.request.called)

    def test_truncates_large_body(self):
        big = _FakeResponse(content=b"x" * 50_000)
        self.mock_session.request.return_value = big
        result = controller.http_request("https://example.com/big")
        self.assertIn("truncated", result)
        self.assertLessEqual(len(result), 21_000)

    def test_retries_on_transient_failure(self):
        from kookie_resilience import transient_http_error

        boom = RuntimeError("connection reset")
        self.mock_session.request.side_effect = [
            boom,
            boom,
            _FakeResponse(content=b"recovered"),
        ]
        with mock.patch.object(controller, "HTTP_REQUEST_RETRY_ATTEMPTS", 3), \
             mock.patch.object(controller, "HTTP_REQUEST_RETRY_DELAY_SECONDS", 0.0), \
             mock.patch("kookie_resilience.time.time", side_effect=[100.0] * 20), \
             mock.patch("kookie_resilience.time.sleep"):
            result = controller.http_request("https://example.com/retry")
        self.assertTrue(self.mock_session.request.called)
        self.assertIn("recovered", result)


if __name__ == "__main__":
    unittest.main()