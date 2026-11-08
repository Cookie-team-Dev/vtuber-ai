"""Helpers for Kookie - re-exports from core_utils.

The real helpers live in ``core_utils``. This module exists so existing
``import helpers`` calls keep working.
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


def a():
    """Backwards-compatible placeholder used by the old stub."""
    return 1


def b():
    """Backwards-compatible placeholder used by the old stub."""
    return 2
