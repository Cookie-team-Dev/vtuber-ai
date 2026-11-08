"""Optional vision adapter for Kookie.

Tries the configured OpenAI-compatible fast provider first, then falls back to
local OpenAmika vision when available.
"""

import base64
import io
import logging
import os
import tempfile
import time

import kookie_resilience

_LOG = logging.getLogger(__name__)


def _env_float(name, default):
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(0.0, value)


def _env_int(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _image_limit_bytes():
    try:
        megabytes = float(os.environ.get("KOOKIE_MAX_VISION_IMAGE_MB", "16"))
    except (TypeError, ValueError):
        megabytes = 16
    return max(1, int(megabytes * 1024 * 1024))


_MAX_IMAGE_BYTES = _image_limit_bytes()
_FAST_PROVIDER_BREAKER = kookie_resilience.CircuitBreaker(
    "fast_vision",
    cooldown_seconds=_env_float("KOOKIE_FAST_VISION_COOLDOWN_SECONDS", 30.0),
    attempts=_env_int("KOOKIE_FAST_VISION_RETRY_ATTEMPTS", 1),
    delay_seconds=_env_float("KOOKIE_FAST_VISION_RETRY_DELAY_SECONDS", 0.15),
    threshold=_env_int("KOOKIE_FAST_VISION_FAILURE_THRESHOLD", 1),
    is_transient=kookie_resilience.default_is_transient,
)


def _read_image_bytes(image):
    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
    elif isinstance(image, (str, os.PathLike)):
        image = os.fspath(image)
        if os.path.getsize(image) > _MAX_IMAGE_BYTES:
            raise ValueError("image is too large")
        with open(image, "rb") as handle:
            raw = handle.read(_MAX_IMAGE_BYTES + 1)
    elif hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        raw = buffer.getvalue()
    else:
        raise TypeError("image must be bytes, a file path, or a PIL-like image")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError("image is too large")
    return raw


def _describe_with_fast_provider(raw_image_bytes, prompt):
    from fastai import describe_image

    encoded = base64.b64encode(raw_image_bytes).decode("ascii")
    return describe_image(encoded, prompt=prompt).strip()


def _describe_with_openamika(raw_image_bytes):
    import video_perception

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        handle.write(raw_image_bytes)
        image_path = handle.name
    try:
        description = video_perception.describe_image_file(image_path)
        return (description or "").strip()
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass


def _retry_fast_provider(raw_image, prompt):
    """Run the fast provider with bounded retries, returning its description."""
    return _FAST_PROVIDER_BREAKER.call(
        _describe_with_fast_provider, raw_image, prompt
    )


def describe(image, prompt="Describe this image clearly and factually."):
    """Return a provider-backed description, then local fallback, else unavailability."""
    try:
        raw_image = _read_image_bytes(image)
    except ValueError:
        return "I cannot describe that image because it is too large."
    except Exception as error:
        _LOG.debug("vision input could not be prepared: %s", error)
        return "I cannot describe that image because no vision model is available."

    if _FAST_PROVIDER_BREAKER.allow():
        try:
            fast_description = _retry_fast_provider(raw_image, prompt)
            if fast_description:
                return fast_description
        except Exception as error:
            # _retry_fast_provider already recorded the failure through the
            # breaker's call() helper, so the only thing left to do here is
            # log and fall through to the local fallback.
            _LOG.debug("fast vision provider unavailable: %s", error)
    else:
        _LOG.debug("fast vision provider temporarily disabled due to recent failures")

    try:
        openamika_description = _describe_with_openamika(raw_image)
        if openamika_description:
            return openamika_description
    except Exception as error:
        _LOG.debug("openamika vision fallback unavailable: %s", error)

    return "I cannot describe that image because no vision model is available."
