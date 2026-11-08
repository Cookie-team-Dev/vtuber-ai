"""All extras for Kookie - re-exports from ``extras`` and ``core_utils``.

This module used to be a placeholder with 80 ``extra_N`` stubs. It now
re-exports the real helpers so the startup banner can still say "Extras all
loaded", and so any code that reached for ``extras_all.truncate`` keeps
working.
"""

from extras import greet, hello, world  # noqa: F401
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


def extra(name):
    """Look up an ``extra_N`` placeholder by name.

    The placeholders were never real functionality - they were just so the
    module imported. This keeps the same surface but is honest about it:
    unknown names raise rather than silently returning a number.
    """
    raise KeyError(name)
