"""Memory helper for Kookie.

The heavy lifting lives in ``memory.py`` (append-only journal, slur filter,
compaction). This module is the thin public surface that other modules and
tools should reach for, so the storage details stay in one place and the
behaviour can be tested without touching the journal file.

It offers:

* ``remember(text)``     - append one note, honouring the slur filter.
* ``forget(text)``       - drop every note that matches ``text``.
* ``search(query, n)``   - keyword recall over the stored notes.
* ``recent(n)``          - the newest notes, newest first.
* ``summary()``          - how many notes are stored and when the last one
  landed, useful for a ``/status`` line.
"""

import logging
import time

import memory as _memory

_LOG = logging.getLogger(__name__)


def remember(text):
    """Append one note. Returns True when stored, False when refused.

    ``memory.remember`` returns None on success (it is a side-effect function),
    so the helper checks the store rather than the return value.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    before = summary()["count"]
    _memory.remember(text)
    return summary()["count"] == before + 1


def forget(text):
    """Drop every stored note whose text contains ``text`` (case-insensitive).

    Returns the number of notes removed. The slur filter is not consulted -
    this is an explicit deletion, not a storage decision.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    needle = text.strip().lower()
    entries = _memory.load_memory()
    remaining = [
        entry for entry in entries
        if needle not in str(entry.get("text", "")).lower()
    ]
    removed = len(entries) - len(remaining)
    if removed:
        _memory.save_memory(remaining)
    _LOG.debug("memory: forgot %d note(s) matching %r", removed, text)
    return removed


def search(query, limit=10):
    """Keyword recall over stored notes, newest first.

    Matching is case-insensitive substring matching on the note text. It is
    intentionally simple - a real embedding index can be layered on later
    without changing this signature.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    needle = query.strip().lower()
    matches = []
    for entry in reversed(_memory.load_memory()):
        text = str(entry.get("text", ""))
        if needle in text.lower():
            matches.append({"text": text, "time": entry.get("time", 0.0)})
            if len(matches) >= max(1, limit):
                break
    return matches


def recent(limit=10):
    """The newest notes, newest first."""
    entries = _memory.load_memory()
    out = []
    for entry in reversed(entries):
        text = entry.get("text")
        if not text:
            continue
        out.append({"text": text, "time": entry.get("time", 0.0)})
        if len(out) >= max(1, limit):
            break
    return out


def summary():
    """A small dict for status lines and tests."""
    entries = _memory.load_memory()
    last = None
    for entry in reversed(entries):
        if entry.get("time"):
            last = entry["time"]
            break
    return {
        "count": len(entries),
        "last_at": last,
        "last_ago": _memory.format_time_ago(last) if last else "never",
    }


def recall_context(query, limit=5):
    """Search plus a short formatted string ready to drop into a prompt.

    Returns an empty string when nothing matches, so callers can write::

        context = recall_context("what did I say about pizza")
        if context:
            prompt += "\\nRelevant memory:\\n" + context
    """
    hits = search(query, limit=limit)
    if not hits:
        return ""
    lines = []
    for hit in hits:
        ago = _memory.format_time_ago(hit.get("time", 0.0))
        lines.append(f"- {hit['text']} ({ago})")
    return "\n".join(lines)


def wait_for_ready(timeout=5.0):
    """Block until the memory store is readable, or timeout.

    The journal is append-only and created lazily on first write, so a brand
    new process can race the first read. This gives the writer a moment.
    """
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        try:
            _memory.load_memory()
            return True
        except Exception:  # noqa: BLE001 - retrying, not handling
            time.sleep(0.05)
    return False
