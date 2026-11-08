"""Audio helper for Kookie.

Thin public surface over ``voice.py``. The synthesis logic (Piper, Amika
GPT-SoVITS, CLI fallbacks) and the playback subprocess management live in
``voice.py``; this module is where other modules and tools reach for
playback, volume and status, so the details stay in one place and can be
mocked in tests without spawning subprocesses.

It offers:

* ``play(path)``       - play a WAV file through the configured audio player.
* ``stop()``           - halt playback immediately.
* ``volume()``        - the current volume as a 0-100 int.
* ``set_volume(level)``- clamp and store the volume for the next play.
* ``is_enabled()``    - whether speech output is switched on at all.
* ``status()``        - a short dict for ``/status`` lines.
"""

import logging
import os

import voice as _voice

_LOG = logging.getLogger(__name__)


def _voice_enabled():
    """Probe whether speech output is switched on.

    controller.voice_enabled() is the authority, but controller imports this
    module at startup, so we reach for it lazily and fall back to voice.py's
    own flag when controller is not importable (e.g. in tests).
    """
    try:
        import controller
        return bool(controller.voice_enabled())
    except Exception:  # noqa: BLE001 - controller may not be importable standalone
        try:
            return bool(getattr(_voice, "VOICE_ENABLED", False))
        except Exception:  # noqa: BLE001 - a broken probe is "off", not a crash
            return False

# 0-100, persisted across calls in the module so a ``set_volume`` call outlives
# the request that made it. voice.py reads this through ``_volume`` below.
_VOLUME = int(os.environ.get("KOOKIE_AUDIO_VOLUME", "80"))
_MIN_VOLUME = 0
_MAX_VOLUME = 100


def _clamp_volume(level):
    try:
        value = int(level)
    except (TypeError, ValueError):
        return _VOLUME
    return max(_MIN_VOLUME, min(_MAX_VOLUME, value))


def play(path):
    """Play a WAV file. Returns the player command list, or an error string."""
    if not isinstance(path, str) or not path:
        return "Error: no audio path"
    if not os.path.isfile(path):
        return f"Error: {path} does not exist"
    try:
        return _voice.audio_player(path)
    except Exception as error:  # noqa: BLE001 - surface to the caller
        return f"Error: {error}"


def stop():
    """Ask the voice layer to halt playback. Returns True when it did."""
    try:
        _voice.stop_speaking()
        return True
    except Exception as error:  # noqa: BLE001 - surface to the caller
        _LOG.debug("audio: stop failed: %s", error)
        return False


def volume():
    """The current volume as a 0-100 int."""
    return _VOLUME


def set_volume(level):
    """Clamp ``level`` to 0-100 and store it. Returns the stored value."""
    global _VOLUME
    _VOLUME = _clamp_volume(level)
    return _VOLUME


def is_enabled():
    """Whether speech output is switched on at all."""
    return bool(_voice_enabled())


def status():
    """A short dict for status lines and tests."""
    return {
        "enabled": is_enabled(),
        "volume": _VOLUME,
    }
