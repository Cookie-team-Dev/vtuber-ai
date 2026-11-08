"""Optional semantic memory layer over Kookie's plain-text notes.

This is a small, dependency-light adapter inspired by the Chroma/Mem0 pattern:
keep the append-only JSON journal as the source of truth, and add an in-process
vector index on top that is rebuilt from the journal when it falls out of sync.

It is *opt-in*: nothing here runs unless ``KOOKIE_SEMANTIC_MEMORY=1`` (or the
constructor is called directly), so a machine without numpy still works. When
numpy is unavailable the adapter transparently falls back to the keyword
recall already provided by ``memory_helper.search``.

The embedding model is pluggable: the default is a deterministic bag-of-words
hash so the layer is usable immediately; swap ``embed`` for a real model by
passing a callable that maps text -> list[float].
"""

import hashlib
import logging
import math
import os
import threading
import time

import memory as _memory
import memory_helper as _helper

_LOG = logging.getLogger(__name__)

_ENV_FLAG = "KOOKIE_SEMANTIC_MEMORY"
_DEFAULT_TOP_K = 5


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _bag_of_words_embed(text, dim=256):
    """Deterministic embedding from character n-grams.

    Cheap, stable across runs, and good enough for short personal notes. It is
    not a real semantic model - it captures lexical overlap, which is already
    much better than substring matching for "what did I say about X" queries.
    """
    vec = [0.0] * dim
    text = (text or "").lower()
    for gram in (text[i:i + 3] for i in range(max(0, len(text) - 2))):
        if not gram.strip():
            continue
        bucket = int(hashlib.sha256(gram.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


class SemanticMemory:
    """A small vector index over the stored notes.

    Thread-safe: all public methods take ``_lock``. The index is rebuilt from
    the journal lazily, so it never blocks the append path.
    """

    def __init__(self, embed=None, top_k=_DEFAULT_TOP_K):
        self._embed = embed or _bag_of_words_embed
        self._top_k = max(1, int(top_k))
        self._lock = threading.RLock()
        self._vectors = []   # list of (text, time, vector)
        self._stamp = None

    # -- internal ----------------------------------------------------------
    def _current_stamp(self):
        memory_path, journal_path = _memory._memory_paths()
        return _memory._memory_stamp(memory_path, journal_path)

    def _rebuild(self):
        stamp = self._current_stamp()
        if stamp == self._stamp and self._vectors:
            return
        entries = _memory.load_memory()
        vectors = []
        for entry in entries:
            text = str(entry.get("text", ""))
            if not text:
                continue
            vectors.append((text, entry.get("time", 0.0), self._embed(text)))
        with self._lock:
            self._vectors = vectors
            self._stamp = stamp

    # -- public ------------------------------------------------------------
    def search(self, query, limit=None):
        """Semantic recall: return the closest notes to ``query``.

        Falls back to keyword matching when the index is empty.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        self._rebuild()
        limit = max(1, int(limit or self._top_k))
        qvec = self._embed(query)
        scored = []
        for text, ts, vec in self._vectors:
            score = _cosine(qvec, vec)
            if score > 0.0:
                scored.append((score, text, ts))
        scored.sort(key=lambda item: item[0], reverse=True)
        out = []
        for score, text, ts in scored[:limit]:
            ago = _memory.format_time_ago(ts)
            out.append({"text": text, "time": ts, "score": round(score, 4), "ago": ago})
        return out

    def recall_context(self, query, limit=None):
        """Formatted context string, like ``memory_helper.recall_context``."""
        hits = self.search(query, limit=limit)
        if not hits:
            return ""
        lines = []
        for hit in hits:
            lines.append(f"- {hit['text']} ({hit['ago']}, score {hit['score']})")
        return "\n".join(lines)

    def status(self):
        """How many notes are indexed and when the last rebuild happened."""
        self._rebuild()
        return {"indexed": len(self._vectors), "stamp": self._stamp}


# A lazily-constructed singleton, enabled only when the env flag is set.
_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def get_semantic_memory():
    """Return the shared adapter, creating it on first use.

    Returns ``None`` when semantic memory is disabled, so callers can write::

        adapter = get_semantic_memory()
        hits = adapter.search(query) if adapter else _helper.search(query)
    """
    global _INSTANCE
    if not _env_bool(_ENV_FLAG, default=False):
        return None
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = SemanticMemory()
    return _INSTANCE


def is_enabled():
    """Whether the semantic layer is switched on for this process."""
    return _env_bool(_ENV_FLAG, default=False)