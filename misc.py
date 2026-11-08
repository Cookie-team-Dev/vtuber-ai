"""Misc utilities - re-exports from core_utils.

The real helpers live in ``core_utils``. This module exists so existing
``import misc`` calls keep working.
"""

from core_utils import (  # noqa: F401
    chunks,
    coalesce,
    core,
    human_seconds,
    now_iso,
    parse_int,
    safe_get,
    slugify,
    truncate,
    unique_id,
)


def x():
    """Backwards-compatible placeholder used by the old stub."""
    return "x"
