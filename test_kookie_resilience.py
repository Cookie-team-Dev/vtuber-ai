import unittest
from unittest import mock

import requests

import kookie_resilience


class _Response:
    """A context-manager-like object mimicking urllib's response for tests."""


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self):
        self.breaker = kookie_resilience.CircuitBreaker(
            "unit_test",
            cooldown_seconds=60.0,
            attempts=1,
            delay_seconds=0.0,
            threshold=1,
            is_transient=kookie_resilience.default_is_transient,
        )

    def test_allow_starts_open(self):
        self.assertTrue(self.breaker.allow())

    def test_success_clears_failures(self):
        self.breaker._failures = 2
        self.breaker._down_until = 1_000_000.0
        self.breaker.success()
        self.assertEqual(self.breaker._failures, 0)
        self.assertEqual(self.breaker._down_until, 0.0)
        self.assertTrue(self.breaker.allow())

    def test_permanent_error_is_not_counted_as_transient(self):
        class PermanentError(Exception):
            pass

        def predicate(error):
            return False

        self.breaker.is_transient = predicate
        # record() returns True for permanent errors so the caller re-raises,
        # and leaves the breaker untouched so the next call still goes through.
        self.assertTrue(self.breaker.record(PermanentError("bad key")))
        self.assertEqual(self.breaker._failures, 0)
        self.assertEqual(self.breaker._down_until, 0.0)
        self.assertTrue(self.breaker.allow())

    def test_threshold_trips_after_repeated_transient_failures(self):
        self.breaker.threshold = 2
        self.breaker.record(RuntimeError("first"))
        self.assertTrue(self.breaker.allow())
        self.assertEqual(self.breaker._down_until, 0.0)

        self.breaker.record(RuntimeError("second"))
        self.assertGreater(self.breaker._down_until, 0.0)
        self.assertFalse(self.breaker.allow())

    def test_call_retries_with_exponential_backoff(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("transient")
            return "recovered"

        with mock.patch.object(kookie_resilience.time, "sleep") as sleep:
            result = self.breaker.call(
                flaky, attempts=3, delay_seconds=0.2, jitter=0.0
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_has_calls([mock.call(0.2), mock.call(0.4)])

    def test_snapshot_counts_calls_successes_and_short_circuits(self):
        self.breaker._down_until = 1_000_000.0  # closed
        self.breaker._last_tripped_at = 100.0
        self.breaker.calls = 0
        self.breaker.successes = 0
        self.breaker.failures = 0
        self.breaker.short_circuits = 0
        self.breaker.down_seconds_total = 0.0

        with mock.patch.object(kookie_resilience.time, "time",
                               side_effect=[500.0, 500.0, 1_000_001.0, 1_000_001.0]):
            self.breaker.allow()  # short-circuit
            self.breaker.allow()  # short-circuit
            self.breaker.allow()  # cooldown expires -> rolls down-time into total

            snap = self.breaker.snapshot()
        self.assertEqual(snap["calls"], 3)
        self.assertEqual(snap["short_circuits"], 2)
        self.assertEqual(snap["down_seconds_total"], 999_900.0)
        self.assertFalse(snap["open"])

    def test_call_reraises_permanent_error_without_backoff(self):
        class PermanentError(Exception):
            pass

        def always_fails():
            raise PermanentError("nope")

        with mock.patch.object(kookie_resilience.time, "sleep") as sleep:
            with self.assertRaises(PermanentError):
                self.breaker.call(
                    always_fails, attempts=3, delay_seconds=0.2,
                    is_transient=lambda error: False,
                )

        sleep.assert_not_called()


class TransientHttpErrorTests(unittest.TestCase):
    def test_retries_5xx(self):
        self.assertTrue(kookie_resilience.transient_http_error("Omniroute HTTP 503: down"))

    def test_retries_429(self):
        self.assertTrue(kookie_resilience.transient_http_error("Omniroute HTTP 429: slow down"))

    def test_retries_timeouts(self):
        self.assertTrue(kookie_resilience.transient_http_error("ReadTimeout: boom"))
        self.assertTrue(kookie_resilience.transient_http_error("ConnectionError: boom"))

    def test_does_not_retry_auth_failure(self):
        self.assertFalse(kookie_resilience.transient_http_error("Omniroute HTTP 401: bad key"))

    def test_does_not_retry_bad_request(self):
        self.assertFalse(kookie_resilience.transient_http_error("Omniroute HTTP 400: bad request"))


class _FakeResponse:
    """A stand-in for requests.Response: status_code + raise_for_status."""

    def __init__(self, status_code=200):
        self.status_code = status_code
        self.text = "body"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class HttpCallTests(unittest.TestCase):
    """Covers kookie_resilience.http_call, the shared HTTP retry helper."""

    def setUp(self):
        self.session = mock.Mock()
        self.session.request = mock.Mock()

    def test_returns_response_on_success(self):
        self.session.request.return_value = _FakeResponse(200)
        response = kookie_resilience.http_call(
            self.session, "GET", "http://example.com", attempts=1
        )
        self.assertEqual(response.status_code, 200)

    def test_retries_on_500_then_succeeds(self):
        self.session.request.side_effect = [_FakeResponse(500), _FakeResponse(200)]
        with mock.patch.object(kookie_resilience.time, "sleep") as sleep:
            response = kookie_resilience.http_call(
                self.session, "GET", "http://example.com", attempts=3, delay_seconds=0.1, jitter=0.0
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.session.request.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_does_not_retry_on_401(self):
        self.session.request.return_value = _FakeResponse(401)
        with mock.patch.object(kookie_resilience.time, "sleep") as sleep:
            response = kookie_resilience.http_call(
                self.session, "GET", "http://example.com", attempts=3, delay_seconds=0.1
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.session.request.call_count, 1)
        sleep.assert_not_called()

    def test_retries_transport_error_then_raises(self):
        import requests as _requests

        self.session.request.side_effect = _requests.ConnectionError("boom")
        with mock.patch.object(kookie_resilience.time, "sleep") as sleep:
            with self.assertRaises(_requests.ConnectionError):
                kookie_resilience.http_call(
                    self.session, "GET", "http://example.com", attempts=2, delay_seconds=0.1, jitter=0.0
                )
        self.assertEqual(self.session.request.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_exponential_backoff_between_retries(self):
        self.session.request.side_effect = [_FakeResponse(503)] * 3
        with mock.patch.object(kookie_resilience.time, "sleep") as sleep, \
             mock.patch.object(kookie_resilience.random, "uniform", return_value=0.0):
            response = kookie_resilience.http_call(
                self.session, "GET", "http://example.com", attempts=3, delay_seconds=0.2, jitter=0.0
            )
        # The last transient response is returned to the caller, who can then
        # decide what to do with it; the backoff between retries is what this
        # test is really checking.
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.session.request.call_count, 3)
        sleep.assert_has_calls([mock.call(0.2), mock.call(0.4)])


if __name__ == "__main__":
    unittest.main()