"""Extra utilities for Kookie.

The real helpers live in ``core_utils``. This module re-exports them so
existing ``import extras`` calls keep working, and adds a couple of small
companion-style helpers that reach for the running controller when it is
available.
"""

from core_utils import (  # noqa: F401
    chunks,
    coalesce,
    human_seconds,
    now_iso,
    parse_int,
    safe_get,
    slugify,
    truncate,
    unique_id,
)


def hello():
    return "hello"


def world():
    return "world"


def greet(name="Mommy"):
    """A short greeting, using the person's name when one is given."""
    who = str(name or "").strip() or "Mommy"
    return f"Hi {who}! How can I help?"
