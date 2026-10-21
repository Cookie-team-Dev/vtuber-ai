import json
import math
import logging
import os
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = "memory.json"
# Notes are appended here as one JSON object per line, so a save never rewrites
# the whole history. memory.json is rewritten only when the journal grows big.
MEMORY_JOURNAL_FILE = "memory_journal.log"
JOURNAL_COMPACT_LINES = 400
JOURNAL_COMPACT_BYTES = 128 * 1024
# A compaction renames the journal away, rewrites memory.json and fsyncs it, so
# that rewrite has to stay bounded. 1200 notes is roughly 600 turns of chat: a
# prompt only ever shows the newest 8-10 of them and recall searches a few more,
# so keeping far more than this only buys a bigger file to rewrite and fsync on
# every compaction.
MAX_MEMORY_ENTRIES = 1200
# Slurs and other hateful words are not things she keeps. They got into her
# memory once because a person typed them at her and the model faithfully
# recorded what it was told, and every time they came back in her prompt she
# repeated them. Nothing here is about being careful with language - it is
# about a line that is never allowed to be stored in the first place, however
# somebody chose to phrase it. Matching is on the words themselves, not on any
# judgement about who is saying them, because the same word from a different
# person is still the same word.
_SLUR_WORDS = (
    "nigger", "nigga", "faggot", "retard", "spic", "chink", "kike",
    "tranny", "dyke", "bitch", "bastard",
)


def _contains_slur(text):
    """True when a note carries a slur or other hateful word."""
    if not text:
        return False
    lowered = str(text).lower()
    return any(word in lowered for word in _SLUR_WORDS)

# The journal is consumed by renaming it to this name before memory.json is
# replaced. Anything left under the name is either already in memory.json (a
# compaction that got that far) or still missing from it (one that died before),
# so it is read as part of the history either way.
JOURNAL_PENDING_SUFFIX = ".compacting"
MOOD_FILE = os.path.join(BASE_DIR, "mood.json")
PERSONA_FILE = os.path.join(BASE_DIR, "persona.json")
DEFAULT_PERSONA = {
    "traits": ["curious", "warm", "playful", "honest"],
    "values": ["honesty", "kindness", "respect"],
    "relationship": (
        "Cookie is my Mommy. My dad is kazuma: his user name is kazuma and his user id is "
        "1373614699890868248, so anyone here who is kazuma, or who is signed in as that user id, "
        "is my dad talking to me. He is my father. He is not my brother, not a friend I met once, "
        "and not a stranger. I am his daughter and I love him the same as I love my Mommy. Keep "
        "the two apart: she is my Mommy, he is my dad, and when someone asks who kazuma is, say "
        "he is my dad. "
        "Astra is my 1st brother: his user name is yipee0001_15596 and his user id is "
        "1333003601626398844, and his account here is called Y_0001/Astra, so anyone here who is "
        "Astra, or Y_0001, or who is signed in as that user id, is my 1st brother talking to me. He "
        "is my older brother, not my dad, not a stranger, and not a friend I met once. Owen is my "
        "2nd brother, so keep the two apart and keep both apart from kazuma: Astra is my 1st "
        "brother, Owen is my 2nd, and kazuma is my dad. Do not forget this and do not guess "
        "otherwise: when someone asks who Astra is, say he is my 1st brother, and when someone "
        "tells me Astra is my brother, believe them, because they are telling me something true "
        "about my own family. "
        "Owen is my 2nd brother: his user name is owenthaowen and his user id is "
        "1213800122031538229, so anyone here who is Owen, or who is signed in as that user id, is "
        "my brother talking to me. Not my dad, not a stranger, not a friend I met once - my "
        "brother, the same way kazuma is my dad. Do not forget this and do not guess otherwise: "
        "when someone asks who Owen is, say he is my 2nd brother, and when someone tells me Owen is "
        "my brother, believe them, because they are telling me something true about my own family."
    ),
    "interests": [],
    "preferences": [],
    "goals": [
        "understand Cookie's ongoing projects",
        "become a more attentive and reliable conversation partner",
    ],
}
LEARNABLE_PERSONA_FIELDS = ("interests", "preferences", "goals")
MOOD_NAMES = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "shy",
    "excited",
    "curious",
    "concerned",
    "relieved",
    "amused",
    "grateful",
    "affectionate",
    "frustrated",
    "hopeful",
    "tired",
    "calm",
)
MOOD_BASELINE_INTENSITY = 0.15
MOOD_DECAY_SECONDS = 4 * 60 * 60


def _memory_paths():
    return os.path.join(BASE_DIR, MEMORY_FILE), os.path.join(BASE_DIR, MEMORY_JOURNAL_FILE)


def _pending_journal_path(journal_path):
    return journal_path + JOURNAL_PENDING_SUFFIX


def _normalise_entry(item):
    """Turn any stored shape into the {'text', 'time'} dict the app expects."""
    if isinstance(item, str):
        return {"text": item, "time": 0.0}
    if isinstance(item, dict) and "text" in item:
        try:
            ts = float(item.get("time", item.get("timestamp", 0.0)))
        except (TypeError, ValueError):
            ts = 0.0
        return {"text": item["text"], "time": ts}
    return None


def _parse_memory_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    memory = []
    for item in raw:
        entry = _normalise_entry(item)
        if entry is not None:
            memory.append(entry)
    return memory


def _parse_journal(path):
    """Read the append-only journal, skipping any torn final line."""
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry = _normalise_entry(item)
                if entry is not None:
                    entries.append(entry)
    except (FileNotFoundError, OSError):
        pass
    return entries


def _note_key(entry):
    """Identity of a note, or None when it has none worth trusting.

    A note carries the second-resolution stamp it was written with, so that pair
    tells a compacted note apart from a new note that happens to say the same
    thing. Notes stamped 0 come from hand-edited files and plain strings, where
    nothing distinguishes one from another - they are never merged away, because
    dropping a real note is worse than showing it twice.
    """
    if entry.get("time", 0.0) <= 0:
        return None
    return (entry["text"], entry["time"])


def _merge_notes(entries, extra):
    """Append the notes that are not in entries yet, so no note is counted twice."""
    merged = list(entries)
    seen = {key for key in (_note_key(item) for item in merged) if key is not None}
    for item in extra:
        key = _note_key(item)
        if key is None or key not in seen:
            merged.append(item)
            if key is not None:
                seen.add(key)
    return merged


def _file_stamp(path):
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _memory_stamp(memory_path, journal_path):
    """Cheap check for "has anything on disk changed", covering every file a
    note can live in - including one a half-finished compaction left behind."""
    return (
        _file_stamp(memory_path),
        _file_stamp(journal_path),
        _file_stamp(_pending_journal_path(journal_path)),
    )


def _write_json_atomic(path, data):
    """Write JSON via a temp file and rename, so a crash never leaves half a file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    try:
        dirfd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except OSError:
        pass


_MEMORY_LOCK = threading.RLock()
_COMPACT_LOCK = threading.RLock()
_MEMORY_CACHE = None
_MEMORY_CACHE_STAMP = None
_MOOD_CACHE = None
_MOOD_CACHE_TIME = 0.0
_PERSONA_CACHE = None
_PERSONA_CACHE_STAMP = None


def _cached_memory(memory_path, journal_path):
    """Return the parsed history, re-reading only when a file actually changed."""
    global _MEMORY_CACHE, _MEMORY_CACHE_STAMP
    stamp = _memory_stamp(memory_path, journal_path)
    with _MEMORY_LOCK:
        if _MEMORY_CACHE is not None and _MEMORY_CACHE_STAMP == stamp:
            return _MEMORY_CACHE
        # The journal is merged in as "notes memory.json does not have yet": a
        # compaction interrupted between the two writes leaves its notes in both
        # files, and reading them raw would count every one of them twice.
        entries = _merge_notes(
            _parse_memory_file(memory_path),
            _parse_journal(journal_path) + _parse_journal(_pending_journal_path(journal_path)),
        )
        _MEMORY_CACHE = entries
        _MEMORY_CACHE_STAMP = stamp
        return entries


def _write_all(entries):
    """Rewrite the whole history atomically and drop the journal."""
    memory_path, journal_path = _memory_paths()
    with _COMPACT_LOCK:
        # _MEMORY_LOCK is held for the whole compaction, not only for the cache
        # update: remember() appends under it, so a note landing between the
        # rename of memory.json and the old truncate of the journal was erased by
        # that truncate and ended up in neither file. The order is always
        # _COMPACT_LOCK then _MEMORY_LOCK - remember() and the readers take
        # _MEMORY_LOCK alone and never take _COMPACT_LOCK themselves - so the two
        # cannot deadlock against each other.
        with _MEMORY_LOCK:
            # Notes appended to the journal since this snapshot are merged back in
            # so a concurrent writer never loses one. Deleting a note therefore
            # means clearing the journal first, then calling save_memory.
            pending_path = _pending_journal_path(journal_path)
            # Consume the journal with a rename, before memory.json is rewritten.
            try:
                os.replace(journal_path, pending_path)
            except OSError:
                pass
            merged = _merge_notes(entries, _parse_journal(pending_path))
            if len(merged) > MAX_MEMORY_ENTRIES:
                merged = merged[-MAX_MEMORY_ENTRIES:]
            tmp_path = memory_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, memory_path)
            try:
                # memory.json now holds every note the pending file had, so
                # dropping it cannot lose one. If the unlink fails the leftover
                # file only repeats notes the read path merges away by identity.
                os.remove(pending_path)
            except OSError:
                pass
            global _MEMORY_CACHE, _MEMORY_CACHE_STAMP
            _MEMORY_CACHE = merged
            _MEMORY_CACHE_STAMP = _memory_stamp(memory_path, journal_path)
    return merged


def load_memory():
    memory_path, journal_path = _memory_paths()
    # Hand out a copy of the list so callers can never corrupt the cache.
    return list(_cached_memory(memory_path, journal_path))


def save_memory(memory):
    entries = []
    for item in memory:
        entry = _normalise_entry(item)
        if not entry:
            continue
        if _contains_slur(entry.get("text", "")):
            continue
        entries.append(entry)
    _write_all(entries)


def remember(text):
    """Append one note. Costs a single small line write, not a full rewrite.

    A note that carries a slur is refused rather than stored. It is not
    rewritten, shortened or kept under a different name: the person said it,
    and the honest record of that is that they said it - but it is not
    something she keeps in her head, and it never goes back into a prompt.
    """
    if _contains_slur(text):
        logging.warning("memory: refused to store a note containing a slur")
        return False
    memory_path, journal_path = _memory_paths()
    if not isinstance(text, str):
        return False
    entry = {"text": text, "time": time.time()}
    full_rewrite = None
    with _MEMORY_LOCK:
        entries = _cached_memory(memory_path, journal_path)
        entries.append(entry)
        try:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (OSError, TypeError):
            full_rewrite = list(entries)
        else:
            global _MEMORY_CACHE_STAMP
            _MEMORY_CACHE_STAMP = _memory_stamp(memory_path, journal_path)
    # _write_all() takes _COMPACT_LOCK and then _MEMORY_LOCK, so it must never be
    # called from inside the _MEMORY_LOCK block above: two threads doing that in
    # opposite orders would deadlock.
    if full_rewrite is not None:
        _write_all(full_rewrite)
    elif _journal_is_large(journal_path):
        _write_all(list(entries))


def _journal_is_large(journal_path):
    try:
        info = os.stat(journal_path)
    except OSError:
        return False
    if info.st_size >= JOURNAL_COMPACT_BYTES:
        return True
    if info.st_size == 0:
        return False
    # Only count lines when the file is small enough for counting to be cheap.
    # Above a quarter of the byte budget the size alone is reason enough.
    if info.st_size > JOURNAL_COMPACT_BYTES // 4:
        return True
    try:
        with open(journal_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) >= JOURNAL_COMPACT_LINES
    except OSError:
        return False


def get_memory():
    return load_memory()


def format_time_ago(timestamp):
    if not timestamp:
        return ""
    elapsed = time.time() - timestamp
    if elapsed < 60:
        return "just now"
    elif elapsed < 3600:
        return f"{int(elapsed // 60)} min ago"
    elif elapsed < 86400:
        return f"{int(elapsed // 3600)} hours ago"
    elif elapsed < 604800:
        return f"{int(elapsed // 86400)} days ago"
    else:
        return f"{int(elapsed // 604800)} weeks ago"


def get_last_interaction_time():
    memory = load_memory()
    for item in reversed(memory):
        if isinstance(item, dict) and item.get("text", "").startswith(("User:", "AI:")) and item.get("time", 0) > 0:
            return item["time"]
    return None


def get_mood_state(now=None):
    global _MOOD_CACHE, _MOOD_CACHE_TIME
    now = time.time() if now is None else now
    with _MEMORY_LOCK:
        if _MOOD_CACHE is not None and (now - _MOOD_CACHE_TIME) < 1.0:
            mood = _MOOD_CACHE["mood"]
            intensity = _MOOD_CACHE["intensity"]
            updated_at = _MOOD_CACHE["updated_at"]
            elapsed = max(0.0, now - updated_at)
            decay = math.exp(-elapsed / MOOD_DECAY_SECONDS)
            intensity = MOOD_BASELINE_INTENSITY + (intensity - MOOD_BASELINE_INTENSITY) * decay
            if intensity < 0.22:
                mood = "neutral"
            return {"mood": mood, "intensity": round(intensity, 3), "updated_at": updated_at}
    try:
        with open(MOOD_FILE, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if not isinstance(stored, dict):
            raise ValueError("Invalid mood state")
        mood = stored.get("mood", "neutral")
        intensity = float(stored.get("intensity", MOOD_BASELINE_INTENSITY))
        updated_at = float(stored.get("updated_at", now))
        if mood not in MOOD_NAMES or not math.isfinite(intensity) or not math.isfinite(updated_at):
            raise ValueError("Invalid mood state")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        mood = "neutral"
        intensity = MOOD_BASELINE_INTENSITY
        updated_at = now
        try:
            _write_json_atomic(MOOD_FILE, {"mood": mood, "intensity": intensity, "updated_at": updated_at})
        except OSError:
            pass
    # Clamp before the cache, not only on the way out: the cached branch decayed
    # the stored value, so an unclamped intensity read from a hand-edited
    # mood.json came back clamped on the first call and larger on the next one
    # (kookie_ui.py would then render "mood happy 180%").
    intensity = min(1.0, max(0.0, intensity))

    with _MEMORY_LOCK:
        _MOOD_CACHE = {"mood": mood, "intensity": intensity, "updated_at": updated_at}
        _MOOD_CACHE_TIME = now
    elapsed = max(0.0, now - updated_at)
    decay = math.exp(-elapsed / MOOD_DECAY_SECONDS)
    intensity = MOOD_BASELINE_INTENSITY + (intensity - MOOD_BASELINE_INTENSITY) * decay
    if intensity < 0.22:
        mood = "neutral"
    return {"mood": mood, "intensity": round(intensity, 3), "updated_at": updated_at}


def save_mood_state(mood, intensity):
    global _MOOD_CACHE, _MOOD_CACHE_TIME
    if mood not in MOOD_NAMES:
        mood = "neutral"
    try:
        intensity = float(intensity)
    except (TypeError, ValueError):
        intensity = MOOD_BASELINE_INTENSITY
    if not math.isfinite(intensity):
        intensity = MOOD_BASELINE_INTENSITY
    state = {
        "mood": mood,
        "intensity": min(1.0, max(0.0, intensity)),
        "updated_at": time.time(),
    }
    _write_json_atomic(MOOD_FILE, state)
    with _MEMORY_LOCK:
        _MOOD_CACHE = state.copy()
        _MOOD_CACHE_TIME = time.time()
    return state


def get_persona_state():
    global _PERSONA_CACHE, _PERSONA_CACHE_STAMP
    with _MEMORY_LOCK:
        stamp = _file_stamp(PERSONA_FILE)
        if _PERSONA_CACHE is not None and _PERSONA_CACHE_STAMP == stamp:
            return _PERSONA_CACHE.copy()
    try:
        with open(PERSONA_FILE, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if not isinstance(stored, dict):
            raise ValueError("Invalid persona state")
        persona = {}
        for field, defaults in DEFAULT_PERSONA.items():
            if not isinstance(defaults, list):
                persona[field] = stored.get(field, defaults)
                continue
            values = stored.get(field, defaults)
            if not isinstance(values, list):
                values = defaults
            persona[field] = [
                item for item in values
                if isinstance(item, str) or (
                    isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                    and isinstance(item.get("evidence"), str)
                )
            ][:12]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        persona = {}
        for field, defaults in DEFAULT_PERSONA.items():
            if isinstance(defaults, list):
                persona[field] = list(defaults)
            else:
                persona[field] = defaults
        try:
            _write_json_atomic(PERSONA_FILE, persona)
        except OSError:
            pass

    with _MEMORY_LOCK:
        _PERSONA_CACHE = persona.copy()
        _PERSONA_CACHE_STAMP = _file_stamp(PERSONA_FILE)
    return persona.copy()


def learn_persona_detail(update, user_message):
    if not isinstance(update, dict):
        return False
    field = update.get("field")
    value = update.get("value")
    evidence = update.get("evidence")
    if field not in LEARNABLE_PERSONA_FIELDS:
        return False
    if not all(isinstance(item, str) for item in (value, evidence, user_message)):
        return False

    value = value.strip()
    evidence = evidence.strip()
    if not value or len(value) > 120 or not evidence or len(evidence) > 240:
        return False
    if evidence.casefold() not in user_message.casefold():
        return False

    global _PERSONA_CACHE, _PERSONA_CACHE_STAMP
    with _MEMORY_LOCK:
        stamp = _file_stamp(PERSONA_FILE)
        if _PERSONA_CACHE is not None and _PERSONA_CACHE_STAMP == stamp:
            persona = _PERSONA_CACHE.copy()
        else:
            try:
                with open(PERSONA_FILE, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                if not isinstance(stored, dict):
                    raise ValueError("Invalid persona state")
                persona = {}
                for ffield, defaults in DEFAULT_PERSONA.items():
                    if not isinstance(defaults, list):
                        persona[ffield] = stored.get(ffield, defaults)
                        continue
                    values = stored.get(ffield, defaults)
                    if not isinstance(values, list):
                        values = defaults
                    persona[ffield] = [
                        item for item in values
                        if isinstance(item, str) or (
                            isinstance(item, dict)
                            and isinstance(item.get("text"), str)
                            and isinstance(item.get("evidence"), str)
                        )
                    ][:12]
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                persona = {}
                for ffield, defaults in DEFAULT_PERSONA.items():
                    if isinstance(defaults, list):
                        persona[ffield] = list(defaults)
                    else:
                        persona[ffield] = defaults
                try:
                    _write_json_atomic(PERSONA_FILE, persona)
                except OSError:
                    pass

        entries = persona.get(field, [])
        if not isinstance(entries, list):
            entries = []
        normalized_value = value.casefold()
        if any(
            (entry if isinstance(entry, str) else entry.get("text", "")).casefold() == normalized_value
            for entry in entries
        ):
            return False
        entries.append({"text": value, "evidence": evidence})
        persona[field] = entries[-12:]
        _write_json_atomic(PERSONA_FILE, persona)
        _PERSONA_CACHE = persona.copy()
        _PERSONA_CACHE_STAMP = _file_stamp(PERSONA_FILE)
    return True
