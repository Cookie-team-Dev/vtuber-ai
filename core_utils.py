"""Core utilities for Kookie.

The small text/format helpers that turn up everywhere - truncating strings,
slugging identifiers, splitting work into bounded chunks, parsing ints that
may or may not be digits, and formatting durations - had no home, so several
one-function stub modules (``misc``, ``more_utils``, ``tools_extra``,
``helpers``) each carried a placeholder. This module owns them once.

The stubs still import from here so existing ``import misc`` calls keep
working.
"""

import re
import time
import uuid


def truncate(text, max_chars=200, suffix="…"):
    """Cut ``text`` to ``max_chars`` and append ``suffix`` when it was cut."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def slugify(text, max_chars=64):
    """Lowercase, hyphen-separated identifier safe for filenames and URLs."""
    if not text:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_chars]


def unique_id(prefix=""):
    """A short random id, useful for temp filenames and request ids."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def chunks(items, size):
    """Yield successive ``size``-sized slices of ``items``."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_int(value, default=0):
    """Parse an int from a string, float, bool or anything else.

    ``"42"`` -> 42, ``42.0`` -> 42, ``True`` -> 1, ``"nope"`` -> default.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def human_seconds(seconds):
    """Render a duration in seconds as ``1h 2m 3s`` style, or ``just now``."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 1:
        return "just now"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs:
        parts.append(f"{secs}s")
    return " ".join(parts) if parts else "0s"


def now_iso():
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def coalesce(*values):
    """Return the first non-None argument, or None."""
    for value in values:
        if value is not None:
            return value
    return None


def safe_get(mapping, *keys, default=None):
    """Walk nested dicts without raising on a missing intermediate key."""
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current if current is not None else default


def core():
    """Backwards-compatible placeholder used by the old stub."""
    return "core"
