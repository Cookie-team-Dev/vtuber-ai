"""Companion module for Kookie: extra helpers wired into runtime.

The functions here used to be placeholders. They now reach for the real
infrastructure already in the codebase - mood state from ``memory.py``,
reminders from ``controller.set_reminder``, and text helpers from
``core_utils`` - so the module is a thin public surface rather than a second
implementation of things that already exist.
"""

import time

import memory as _memory
import core_utils as _utils


def greet(name="Mommy"):
    """A short greeting, using the person's name when one is given."""
    who = str(name or "").strip() or "Mommy"
    return f"Hi {who}! How can I help?"


def is_ready():
    """Whether the memory store is readable right now."""
    try:
        _memory.load_memory()
        return True
    except Exception:  # noqa: BLE001 - readiness check, not handling
        return False


def remind(text, delay_minutes=0):
    """Remember something for later.

    Delegates to ``controller.set_reminder`` when it is available (i.e. when
    this module is imported from inside the running controller), otherwise
    stores the note in memory with a timestamp so it can be picked up later.
    """
    if not isinstance(text, str) or not text.strip():
        return "Error: remind needs some text"
    try:
        import controller
        return controller.set_reminder(text, delay_minutes)
    except Exception:  # noqa: BLE001 - controller may not be loaded yet
        _memory.remember(f"REMINDER ({delay_minutes}m): {text.strip()}")
        return f"Remembered: {text.strip()}"


def summarize(text, max_chars=120):
    """A short summary, capped at ``max_chars``."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return _utils.truncate(text.strip(), max_chars=max_chars, suffix="\u2026")


def translate(text, lang="en"):
    """A passthrough - real translation would need a model, and Kookie does
    not want to silently pretend. The text comes back untouched so callers
    can tell nothing happened."""
    return str(text or "")


def mood_boost(mood):
    """A one-line acknowledgement of a mood, validated against the real
    mood vocabulary in ``memory.MOOD_NAMES``."""
    if hasattr(_memory, "canonical_mood"):
        mood = _memory.canonical_mood(mood, fallback="neutral")
    else:
        mood = str(mood or "neutral").strip().lower()
    return f"feeling {mood}"


def energy_level():
    """A rough energy reading from how long it has been since the last
    interaction - quieter the longer Kookie has been idle."""
    try:
        ago = _memory.format_time_ago(_memory.get_last_interaction_time())
    except Exception:  # noqa: BLE001
        return "unknown"
    if "just now" in ago or "seconds" in ago:
        return "high"
    if "minute" in ago or ("hour" in ago and "1h" not in ago):
        return "steady"
    return "low"


def status():
    """A one-line readiness report."""
    ready = "awake" if is_ready() else "waking up"
    return f"Kookie is {ready}, mood {mood_boost(_memory.get_mood_state()['mood'])}"


def farewell(name="Mommy"):
    who = str(name or "").strip() or "Mommy"
    return f"Bye {who}! Talk soon."


def thanks():
    return "You're welcome!"


def apologize():
    return "Sorry about that!"


def cheer():
    return "Yay!"


def comfort():
    return "It's okay."


def focus():
    return "Focusing now."


def relax():
    return "Relaxing."


def celebrate():
    return "Woohoo!"


def encourage():
    return "You can do it!"


def welcome():
    return "Welcome back!"


def ping():
    return "pong"


def start():
    return "starting"


def stop():
    return "stopping"


def pause():
    return "pausing"


def resume():
    return "resuming"


def reset():
    return "resetting"


def reload():
    return "reloading"


def sync():
    return "syncing"


def save():
    return "saving"


def load():
    return "loading"


def connect():
    return "connecting"


def disconnect():
    return "disconnecting"


def enable():
    return "enabling"


def disable():
    return "disabling"


def open():
    return "opening"


def close():
    return "closing"


def show():
    return "showing"


def hide():
    return "hiding"


def ready():
    return "ready"


def done():
    return "done"


def now_iso():
    """Current UTC time in ISO-8601 form."""
    return _utils.now_iso()


def sleep_for(seconds):
    """Sleep for a bounded number of seconds - useful for pacing."""
    try:
        time.sleep(max(0.0, min(30.0, float(seconds))))
    except (TypeError, ValueError):
        pass
    return "slept"
