"""Shared retry + circuit-breaker helpers for Kookie.

Mature OSS AI stacks (tenacity, backoff, Chroma, Mem0) all converge on the same
small set of ideas, so instead of re-implementing them in every module this
module owns them once:

* bounded exponential backoff between immediate retries,
* a failure threshold before entering a cooldown,
* a transient-error predicate so permanent failures (auth, bad request) are
  never retried.

Every caller keeps its own cooldown clock so a single unstable backend cannot
take the whole process down, but the shape of the behaviour is identical.
"""

import logging
import os
import random
import time

_LOG = logging.getLogger(__name__)


def _env_int(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _env_float(name, default):
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(0.0, value)


class CircuitBreaker:
    """A tiny cooldown gate with bounded retries and a transient predicate.

    Usage::

        breaker = CircuitBreaker("fast_vision")
        if not breaker.allow():
            return fallback()
        try:
            result = do_call()
            breaker.success()
            return result
        except Exception as error:
            if breaker.record(error):
                raise
            return fallback()
    """

    def __init__(self, name, cooldown_seconds=None, attempts=None,
                 delay_seconds=None, threshold=None, is_transient=None,
                 jitter=None):
        self.name = name
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else \
            _env_float(f"KOOKIE_{name.upper()}_COOLDOWN_SECONDS", 30.0)
        self.attempts = attempts if attempts is not None else \
            _env_int(f"KOOKIE_{name.upper()}_RETRY_ATTEMPTS", 1)
        self.delay_seconds = delay_seconds if delay_seconds is not None else \
            _env_float(f"KOOKIE_{name.upper()}_RETRY_DELAY_SECONDS", 0.15)
        self.threshold = threshold if threshold is not None else \
            _env_int(f"KOOKIE_{name.upper()}_FAILURE_THRESHOLD", 1)
        self.jitter = jitter if jitter is not None else \
            _env_float(f"KOOKIE_{name.upper()}_RETRY_JITTER", 0.25)
        self.is_transient = is_transient or (lambda error: True)
        self._down_until = 0.0
        self._failures = 0

    # -- state ----------------------------------------------------------

    def allow(self):
        """True when the gate is open and the call may proceed."""
        return time.time() >= self._down_until

    def success(self):
        """Record a successful call: clear failures and cooldown."""
        self._failures = 0
        self._down_until = 0.0

    def record(self, error):
        """Record a failure. Returns True if the caller should re-raise.

        Permanent errors (auth, bad request) are re-raised immediately. After
        ``threshold`` transient failures the breaker trips and subsequent
        calls short-circuit to the fallback until the cooldown expires.
        """
        if not self.is_transient(error):
            return True
        self._failures += 1
        if self._failures >= self.threshold:
            self._down_until = time.time() + self.cooldown_seconds
            self._failures = 0
            _LOG.debug("%s circuit breaker tripped for %.0fs: %s",
                       self.name, self.cooldown_seconds, error)
        return False

    # -- convenience ----------------------------------------------------

    def call(self, func, *args, attempts=None, delay_seconds=None,
             is_transient=None, jitter=None, **kwargs):
        """Run func with bounded retries, returning its result or raising.

        ``attempts``, ``delay_seconds``, ``is_transient`` and ``jitter``
        override the breaker's configured values for this single call, which
        is how a permanent auth failure can be re-raised without changing
        global state.

        ``jitter`` randomises the backoff by a fraction (0-1) so that several
        clients failing over at once do not retry in lockstep.
        """
        attempts = self.attempts if attempts is None else max(1, attempts)
        delay = self.delay_seconds if delay_seconds is None else max(0.0, delay_seconds)
        if jitter is None:
            jitter = self.jitter
        predicate = self.is_transient if is_transient is None else is_transient
        last_error = None
        for attempt in range(attempts):
            try:
                result = func(*args, **kwargs)
                self.success()
                return result
            except Exception as error:  # noqa: BLE001 - re-raised below
                last_error = error
                if attempt + 1 >= attempts or not predicate(error):
                    break
                if delay > 0:
                    time.sleep(_jittered_delay(delay, attempt, jitter))
        if last_error is not None and not predicate(last_error):
            raise last_error
        if last_error is not None:
            self.record(last_error)
        raise last_error if last_error is not None else RuntimeError("no attempt made")


def transient_http_error(error):
    """True for HTTP 429/5xx, timeouts and connection failures.

    A 401/403 (bad key) or a 400 (bad request) is the caller's problem and is
    never retried - the same request would fail again.
    """
    text = str(error)
    if "HTTP 429" in text or "HTTP 5" in text:
        return True
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "timeout",
            "connection",
            "readtimeout",
            "connecttimeout",
            "chunkedencodingerror",
            "jsondecodeerror",
        )
    )


def default_is_transient(error):
    """Fallback predicate: treat every exception as transient."""
    return True


def http_call(session, method, url, attempts=None, delay_seconds=None,
              timeout=None, jitter=None, **kwargs):
    """Run an HTTP call with bounded retries, returning the response.

    Permanent failures (auth, bad request) are re-raised immediately. 429/5xx
    and transport errors are retried with exponential backoff. The caller
    still owns the session, so connection pooling and headers stay consistent.

    ``jitter`` is the fraction of the computed delay that is randomised (0-1).
    A small amount of jitter prevents every client in a fleet from retrying at
    exactly the same instant after an outage, which is what happens with a
    plain deterministic backoff.
    """
    attempts = _env_int("KOOKIE_HTTP_REQUEST_RETRY_ATTEMPTS", 1) if attempts is None else max(1, attempts)
    delay = _env_float("KOOKIE_HTTP_REQUEST_RETRY_DELAY_SECONDS", 0.15) if delay_seconds is None else max(0.0, delay_seconds)
    if jitter is None:
        jitter = _env_float("KOOKIE_HTTP_REQUEST_RETRY_JITTER", 0.25)
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            status = getattr(response, "status_code", None)
            if status is not None and transient_http_error(f"HTTP {status}") and attempt + 1 < attempts:
                last_error = response
                if delay > 0:
                    time.sleep(_jittered_delay(delay, attempt, jitter))
                continue
            return response
        except Exception as error:  # noqa: BLE001 - re-raised below
            last_error = error
            if attempt + 1 >= attempts:
                raise
            if delay > 0:
                time.sleep(_jittered_delay(delay, attempt, jitter))
    raise last_error if last_error is not None else RuntimeError("no attempt made")


def _jittered_delay(base_delay, attempt, jitter):
    """Exponential backoff with a randomised fraction to avoid thundering herds."""
    if jitter <= 0:
        return base_delay * (2 ** attempt)
    spread = base_delay * (2 ** attempt) * jitter
    return max(0.0, base_delay * (2 ** attempt) - spread + random.uniform(0, 2 * spread))