"""All mods for Kookie - re-exports from ``core_utils``.

This module used to be a placeholder with ``mod_a``..``mod_o`` stubs. It now
re-exports the real helpers so the startup banner can still say "Mods all
loaded", and so any code that reached for ``mods_all.slugify`` keeps working.
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


def mod(name):
    """Look up a ``mod_X`` placeholder by name.

    The placeholders were never real functionality. Unknown names raise
    rather than silently returning a letter.
    """
    raise KeyError(name)
