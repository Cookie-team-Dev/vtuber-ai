"""A fast cloud model for the work that does not need her personality.

Kookie thinks on a local llama3.2 through Ollama, on a CPU-only box that has
7.5GiB of RAM and 8 cores shared with the editor. That model was measured at
0.3 tokens/second while the machine was short of memory, and 4-5 tokens/second
when it was not. Either way a chat reply, a background thought and a browser
turn are all measured in tens of seconds, and the browser agent was missing its
whole 240 second budget every single attempt.

None of that is a quality problem, it is a throughput problem, and a hosted GPU
does not have it. So this module talks to one of those, and everything it
returns is shaped exactly like an Ollama /api/chat response: the same text, the
same tool_calls list with a function name and an arguments dict, the same
finish_reason. That is deliberate. The call sites in controller.py already know
how to read that shape from the browser loop and from the salvaging logic, and
matching it means adding a fallback rather than rewriting three call paths.

All three speak the OpenAI chat API, so they share one code path:

    GEMINI_API_KEY      Google Gemini, which is what she uses when this is set.
                         It has an OpenAI compatible endpoint of its own, so it
                         costs one entry in this list rather than a second client.
    GROQ_API_KEY        Groq, as a second choice.
    OPENROUTER_API_KEY  OpenRouter last.
    OMNIROUTE_OMNIOUTE_URL/Omniroute, as a final fallback when every other key
    is out of room.

Whichever key is present is used, and the order above is the order of
preference: Gemini, then Groq, then OpenRouter, then Omniroute. With none of
them set Omniroute can still answer when installed, and every caller quietly
keeps using the local model otherwise, so nothing breaks by adding it and
nothing is lost by leaving it out.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import omniroute

# ---------------------------------------------------------------------------
# Automatic model discovery
#
# The static candidate list in PROVIDERS is a starting point, not a ceiling.
# Providers retire models and ship new ones underneath this process, so each
# provider's /models endpoint is queried once and the fresh list is merged with
# the static candidates. Newly released models are tried before the static
# fallback, so she picks up a new Flash or Llama without a code change.
#
# Discovery is cached with a TTL so a reply does not pay for an HTTP call, and
# a network or auth failure silently falls back to the static list - nothing
# breaks when discovery cannot run.
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}  # env_var -> (tuple of model ids, expiry monotonic time)
_MODEL_TTL_SECONDS = 3600  # refresh the model list once an hour
_MODEL_FAIL_TTL_SECONDS = 600  # don't retry a failed discovery for 10 minutes

# The probe costs 8.5s the first time, and the cache is per-process, so a Kookie
# restart paid it again on the first reply. The result is written to disk in the
# same state directory the launcher uses for the Amika log, so it survives a
# restart and the first reply after one is quick instead of cold. A corrupt or
# stale file is ignored - it is a cache, not a config, and falling back to a
# fresh probe is the correct behaviour when it cannot be trusted.
_MODEL_CACHE_FILE = os.path.join(
    os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")),
    "kookie", "fastai_models.json",
)

# Don't print the same key error more than once per dedup window. chat() can be
# called more than once for the same message (initial attempt, then retry after
# Ollama fails), and without this every key's refusal is logged twice.
_RECENT_ERRORS = {}  # (env_var, key_number, error_text) -> last print time
_DEDUP_WINDOW_SECONDS = 5

# Chat-capable model name hints. A discovered model is only used when its name
# looks like something she can actually talk to: a raw embedding or image gen
# model is not a chat model.
_CHAT_MODEL_HINTS = (
    "flash", "gemini", "llama", "qwen", "mistral", "mixtral", "deepseek",
    "command", "claude", "gpt", "nemotron", "glm", "hermes",
)
_CHAT_MODEL_SKIPS = (
    "embed", "embedding", "rerank", "image", "generate", "moderation",
    "whisper", "tts", "audio", "transcrib",
)


def _looks_like_chat_model(model_id):
    """True when a model name sounds like a chat/completion model."""
    lowered = model_id.lower()
    if any(skip in lowered for skip in _CHAT_MODEL_SKIPS):
        return False
    return any(hint in lowered for hint in _CHAT_MODEL_HINTS)


def _fetch_provider_models(base_url, key):
    """Ask one provider's /models endpoint for the models it will serve.

    Returns None on any failure so the caller falls back to the static list.
    """
    if not key:
        return None

    # Try the OpenAI-compatible /models path first, then a provider-specific
    # fallback. Gemini's own endpoint is generativelanguage.googleapis.com but
    # its OpenAI-compatible mirror also lists models.
    candidates = []
    base = base_url.rstrip("/")
    if "/openai" in base:
        candidates.append(base + "/models")
        candidates.append("https://generativelanguage.googleapis.com/v1beta/models")
    else:
        candidates.append(base + "/models")
        candidates.append(base + "/openai/v1/models")

    last_error = None
    for url in candidates:
        try:
            # Gemini's native endpoint authenticates the key as a query string
            # parameter; the x-goog-api-key header alone is a 401. OpenAI-style
            # providers reject an unknown query parameter, so it is only added
            # for the generativelanguage host.
            params = (
                {"key": key}
                if "generativelanguage.googleapis.com" in url
                else None
            )
            response = _SESSION.get(
                url,
                params=params,
                headers=_provider_headers(
                    {"label": "Gemini" if "generativelanguage.googleapis.com" in url
                     else "OpenRouter" if "openrouter.ai" in base else "provider"},
                    key, url,
                ),
                timeout=15,
            )
            if response.status_code >= 400:
                last_error = f"{url}: HTTP {response.status_code}"
                continue
            parsed = response.json()
            models = []
            # Both providers' list shapes are handled here. OpenAI returns
            # {"data": [{"id": "..."}]}; Gemini returns
            # {"models": [{"name": "models/gemini-2.5-flash"}]} where the id
            # carries a "models/" prefix that has to be stripped before use.
            raw_entries = parsed.get("data") or parsed.get("models") or []
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                mid = entry.get("id") or entry.get("name") or ""
                if mid.startswith("models/"):
                    mid = mid[len("models/"):]
                if not mid:
                    continue
                if "generativelanguage.googleapis.com" in base:
                    methods = entry.get("supportedGenerationMethods") or []
                    if (
                        isinstance(methods, list)
                        and "generateContent" not in methods
                    ):
                        continue
                else:
                    parameters = entry.get("supported_parameters")
                    openrouter_contract_ok = (
                        "openrouter.ai" not in base
                        or not isinstance(parameters, list)
                        or {"tools", "response_format"}.issubset(parameters)
                    )
                    if not openrouter_contract_ok:
                        continue
                if _looks_like_chat_model(mid):
                    models.append(mid)
            if models:
                return tuple(models)
        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as error:
            last_error = f"{url}: {error}"
            continue
    print(f"fastai: model discovery failed ({last_error}), using static list")
    return None


def _provider_headers(provider, key, url=""):
    """Build provider auth headers, including OpenRouter app attribution.

    Gemini speaks two endpoints and they want different credentials. The native
    generateContent path takes the key as x-goog-api-key; the OpenAI-compatible
    /chat/completions path takes it as an Authorization: Bearer header and
    rejects x-goog-api-key with "Missing or invalid Authorization header". The
    same key therefore has to be sent in the shape each endpoint expects, which
    is why the URL is part of the call here rather than just the label.
    """
    headers = {"Content-Type": "application/json"}
    label = provider.get("label") if isinstance(provider, dict) else provider
    if label == "Gemini" and "chat/completions" not in url:
        headers["x-goog-api-key"] = key
    else:
        headers["Authorization"] = "Bearer " + key
    if label == "OpenRouter":
        headers["HTTP-Referer"] = "https://github.com/Cookie-team-Dev/vtuber-ai"
        headers["X-Title"] = "Kookie"
    return headers


_PROBE_TIMEOUT = 8  # seconds per model; a model that has not answered by then
                     # is treated as unavailable for this discovery cycle.
_PROBE_BUDGET = 20  # hard cap on the whole probe, so a provider that is slow
                    # across the board never delays the first reply by minutes.


def _probe_live_models(base_url, key, models, label):
    """Trim a discovered list to the models this key can actually answer.

    Set KOOKIE_PROBE_LIVE=0 to skip the probes - the tests and anything else
    that imports this module without wanting to hit a provider need the
    discovered list returned unchanged.

    The /models endpoint lists what a provider *sells*; it says nothing about
    what this key can reach today. Measured here on the Gemini free tier:
    gemini-2.5-flash and gemini-2.5-pro both 404 "no longer available to new
    users", gemini-flash-latest 429s, gemini-3.8-flash hangs 25s before a read
    timeout, and gemini-3.5-flash answers the same 1-token probe in 0.8s. With
    23 listed models, asking every one on every reply is what turned a hosted
    reply into a 68s wait, so the list is trimmed to the ones that answer -
    once per discovery TTL, in parallel, and never longer than the budget.
    """
    if os.environ.get("KOOKIE_PROBE_LIVE", "1").lower() in ("0", "false", "no"):
        return models
    if not models:
        return models
    native = "generativelanguage.googleapis.com" in base_url
    deadline = time.monotonic() + _PROBE_BUDGET
    results = {}

    def probe(model):
        if time.monotonic() > deadline:
            return model, False
        body = (
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
             "generationConfig": {"maxOutputTokens": 4, "temperature": 0}}
            if native else
            {"model": model, "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 4, "temperature": 0}
        )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
            if native else f"{base_url.rstrip('/')}/chat/completions"
        )
        try:
            response = _SESSION.post(
                url,
                headers=_provider_headers(
                    {"label": label}, key, url if native else ""
                ),
                json=body,
                timeout=(CONNECT_TIMEOUT, _PROBE_TIMEOUT),
            )
            return model, response.status_code == 200
        except requests.exceptions.RequestException:
            return model, False

    workers = min(len(models), 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for model, ok in pool.map(probe, models):
            results[model] = ok
    live = [m for m in models if results.get(m)]
    if not live:
        # Nothing answered in the budget: keep the original list rather than
        # dropping the provider entirely, since the failures may all be
        # transient and the probe itself is only one snapshot.
        return models
    return live


def _load_model_cache():
    """Restore the probed model list from disk, if it is still fresh.

    The probe costs 8.5s and the in-memory cache does not survive a restart, so
    a Kookie that had just come up paid it again on the first reply. The file
    lives in the same state directory as the Amika log and carries a monotonic
    expiry, so a stale or corrupt file is simply ignored and a fresh probe runs.
    """
    try:
        with open(_MODEL_CACHE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return
        now = time.monotonic()
        for env_var, entry in data.items():
            if not isinstance(entry, dict):
                continue
            models = entry.get("models")
            expires = entry.get("expires")
            if not isinstance(models, list) or not isinstance(expires, (int, float)):
                continue
            if expires > now:
                _MODEL_CACHE[env_var] = (tuple(models), float(expires))
    except (OSError, ValueError, TypeError):
        # A cache is a hint, not a contract. Missing, corrupt or unreadable
        # means "probe again", which is the same work the first reply would do.
        pass


def _save_model_cache():
    """Persist the probed model list so a restart does not re-probe."""
    try:
        directory = os.path.dirname(_MODEL_CACHE_FILE)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {}
        for env_var, (models, expires) in _MODEL_CACHE.items():
            if expires > time.monotonic():
                payload[env_var] = {"models": list(models), "expires": expires}
        if not payload:
            return
        tmp = _MODEL_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, _MODEL_CACHE_FILE)
    except OSError:
        # Disk full, read-only filesystem, state dir unwritable: none of these
        # is worth breaking a reply over. The in-memory cache still works.
        pass


def _discover_models(env_var, base_url, models, label):
    """Return the model tuple for one provider: discovered first, static second.

    A pinned model ({env_var}_MODEL) wins outright and discovery is skipped.
    Otherwise the live provider list is fetched and merged with the static
    candidates, deduplicated, discovered-first.

    Failures are cached with a short TTL so a single 401 does not log on every
    reply - the next retry happens after _MODEL_FAIL_TTL_SECONDS.
    """
    pinned = _key_for(f"{env_var}_MODEL")
    if pinned:
        return tuple(models)

    now = time.monotonic()
    cached = _MODEL_CACHE.get(env_var)
    if cached and cached[1] > now:
        return cached[0]
    # The first discovery call in a process loads the disk cache so a restart
    # does not pay the 8.5s probe again on the very first reply.
    if not _MODEL_CACHE:
        _load_model_cache()
        cached = _MODEL_CACHE.get(env_var)
        if cached and cached[1] > now:
            return cached[0]

    # keys_for splits the comma/space-separated list; _key_for would return the
    # whole blob as one key and every discovery call would 401.
    all_keys = keys_for(env_var)
    key = all_keys[0] if all_keys else _key_for(env_var)
    discovered = _fetch_provider_models(base_url, key)
    if discovered:
        # Discovered models first, then static candidates that were not
        # already discovered. Deduplicated, order preserved.
        merged = list(discovered)
        for mid in models:
            if mid not in merged:
                merged.append(mid)
        # A listing says a model exists; it does not say the key can use it.
        # Measured here, gemini-2.5-flash and gemini-2.5-pro both 404 "no
        # longer available to new users", gemini-flash-latest 429s on the free
        # tier, and gemini-3.8-flash hangs for 25s before timing out - while
        # gemini-3.5-flash answers the same 1-token probe in 0.8s. Asking all
        # 23 of them on every reply is what made a hosted reply cost 68s, so the
        # list is trimmed to the ones that actually answer, once per TTL.
        merged = _probe_live_models(base_url, key, merged, label)
        result = tuple(merged)
        _MODEL_CACHE[env_var] = (result, now + _MODEL_TTL_SECONDS)
        _save_model_cache()
        return result

    # Discovery failed or returned nothing: cache the static list briefly so
    # we do not hammer the provider's /models endpoint on every reply.
    result = tuple(models)
    _MODEL_CACHE[env_var] = (result, now + _MODEL_FAIL_TTL_SECONDS)
    _save_model_cache()
    print(f"fastai: model discovery failed for {label}, using static list "
          f"(retry in {_MODEL_FAIL_TTL_SECONDS // 60}m)")
    return result


PROVIDERS = (
    (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        (
            # Discovery populates this first; the static list is the fallback
            # only when the /models call fails. Ordered by what actually answers
            # on the free tier today: gemini-2.5-flash and gemini-2.5-pro 404
            # "no longer available to new users", gemini-flash-latest 429s,
            # gemini-3.8-flash hangs 25s before a read timeout, and
            # gemini-3.5-flash answers a 1-token probe in 0.8s.
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
        ),
        "Gemini",
    ),
    (
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        (
            # Fastest thing on the platform, and good enough for the envelopes and
            # the tool calls. Listed first on purpose: the whole point of this
            # module is that she answers quickly.
            "openai/gpt-oss-120b",
            # Twice the size for when the smaller one gets something wrong.
            "qwen/qwen3.6-27b",
            "llama-3.3-70b-versatile",
            "mistral-7b-v0.3",
        ),
        "Groq",
    ),
    (
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        (
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
        ),
        "OpenRouter",
    ),
)
PROVIDER_ORDER = ("gemini", "groq", "openrouter")

# Allow forcing a specific provider via environment variable
# KOOKIE_FAST_PROVIDER=gemini|groq|openrouter
_FAST_PROVIDER_OVERRIDE = os.environ.get("KOOKIE_FAST_PROVIDER", "").strip().lower()
if _FAST_PROVIDER_OVERRIDE and _FAST_PROVIDER_OVERRIDE not in PROVIDER_ORDER:
    print(f"fastai: invalid KOOKIE_FAST_PROVIDER '{_FAST_PROVIDER_OVERRIDE}', ignoring")
    _FAST_PROVIDER_OVERRIDE = ""

# The hosted models are a network service, so the failure modes are the network's:
# a cold connect, a provider that is briefly down, a free tier that rate limits
# under a burst of thoughts. None of those should look like a crash to the
# caller, so timeouts are tight and everything else is left to the local model.
# 60s rather than 30s: Gemini 3 Flash Preview is slower on the free tier than
# the old Flash-Lite, and 30s was timing out on every reply.
DEFAULT_TIMEOUT = 60
CONNECT_TIMEOUT = 15

_SESSION = requests.Session()


class FastAIError(Exception):
    """The fast provider could not answer. Callers fall back to the local model."""


class FastAITimeout(FastAIError):
    """It did not say anything in time.

    Its own kind of failure, because it is the one worth trying again: a provider
    that has run out of quota for the day is not coming back in twenty seconds,
    but a hosted model that took too long on the first try is usually one slow
    response away from answering. Measured here, a turn that fell through to the
    local model on a timeout took 167.6s, and the same prompt through the hosted
    model takes 1.0s once it answers.
    """


_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_KEY_FILE_VALUES = None
_KEY_FILE_STAMP = None


def _key_file_values():
    """Keys saved in the .env beside this file, reread whenever the file changes.

    An exported variable only survives in the terminal that exported it, so
    without this every other launch silently reported "Fast replies: off" and
    went back to the slow local model. One line in .env is a one-time setup, and
    .env is already in .gitignore, so the key stays on this machine.

    Read once per process used to be enough, and then a key added at 21:00 did
    nothing until the bot was restarted - which is the wrong time to decide to
    restart a bot, because a restart drops the Discord session and there is
    usually a message in flight. Adding a key is something that gets done while
    she is answering, so the file is checked by its timestamp on every ask: one
    stat call, and a real read only when the file actually moved.
    """
    global _KEY_FILE_VALUES, _KEY_FILE_STAMP
    try:
        stamp = os.stat(_KEY_FILE).st_mtime_ns
    except OSError:
        stamp = None
    if _KEY_FILE_VALUES is not None and stamp == _KEY_FILE_STAMP:
        return _KEY_FILE_VALUES
    values = {}
    try:
        with open(_KEY_FILE, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if value:
                    values[name.strip()] = value
    except OSError:
        pass
    _KEY_FILE_VALUES, _KEY_FILE_STAMP = values, stamp
    return values


def _key_for(name):
    """A key from the environment, or the same key saved in .env."""
    return (os.environ.get(name) or "").strip() or _key_file_values().get(name, "")


def keys_for(env_var):
    """Every key configured for one provider, in the order they should be tried.

    More than one is a normal thing to have rather than a special case, because
    the free tiers are per key: measured here, one Gemini key refusing "you
    exceeded your current quota - limit 500 per day, retry in 11 hours" says
    nothing about a second key, which has its own 500. Sitting on one key means
    an eleven hour wait for a reply that takes one second through the next one.

    So the keys live in one variable, separated by commas or spaces:
    `GEMINI_API_KEY=first-key,second-key`. Whitespace separation is read too
    because a key pasted with a trailing newline is a thing that happens.
    """
    raw = os.environ.get(env_var) or _key_file_values().get(env_var, "")
    keys = []
    for chunk in re.split(r"[,\s]+", str(raw or "")):
        # Quotes are stripped per key as well as around the whole value, since a
        # list of keys is usually written out one at a time and quoted per key.
        chunk = chunk.strip().strip('"').strip("'").strip()
        if chunk and chunk not in keys:
            keys.append(chunk)
    return keys


def _env_int(name, default):
    """An integer setting from the environment, or the default when it is not one.

    A typo in the value used to raise a ValueError while this module was being
    imported, which took the whole bot down with it - the local model and every
    other module that imports this one. A bad number in one setting is a
    configuration mistake, not a reason for the bot not to start.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"fastai: {name}={raw!r} is not a number, using {default}")
        return default


# Which key answered last, per provider, so the next reply tries that one first.
# With six keys where five share a spent quota this is worth 2s and five lines of
# log per message: without it every reply walks the dead ones first, and the walk
# is only avoided while their cooldowns last.
_KEY_PREFERENCE = {}

# Persisted across restarts so she stays on the key that worked even after being
# turned off and back on. Written once she answers on a key, read at import.
_PREFERENCE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kookie_key_preference.json"
)


def _load_preferences():
    """Read the saved key preference from disk, if any."""
    global _KEY_PREFERENCE
    try:
        with open(_PREFERENCE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for env_var, slot in data.items():
                if isinstance(env_var, str) and isinstance(slot, str):
                    _KEY_PREFERENCE[env_var] = slot
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
        # No preference file yet, or it is unreadable: start fresh.
        pass


def _save_preferences():
    """Write the current key preference to disk."""
    try:
        with open(_PREFERENCE_FILE, "w", encoding="utf-8") as f:
            json.dump(_KEY_PREFERENCE, f, indent=1)
    except OSError:
        # Best effort only - a failed write does not stop her answering.
        pass


# Read what was saved last time she ran.
_load_preferences()

# How much of a key's daily allowance to spend before moving on to the next one.
# The free tier is 500 requests a day for GenerateRequestsPerDayPerProjectPerModel,
# measured here as a hard refusal rather than a slowdown, so the last slice is
# left alone: a key that runs out mid-conversation stops the reply instead of
# slowing it, and there is no way to ask Google where the line is before the 429.
KEY_DAILY_BUDGET = _env_int("KOOKIE_KEY_DAILY_REQUESTS", 400)
# The hour a key's allowance starts over. Measured here from the 429s: every one
# of them said the quota was back between 07:00 and 08:00 Cambodia time, not at
# midnight, so a counter that resets at midnight hands a key back seven hours
# early and collects a refusal for each reply in between.
QUOTA_RESET_HOUR = _env_int("KOOKIE_QUOTA_RESET_HOUR", 7)
_KEY_USAGE = {}

# One lock for the key state rather than one per table. The tables are read and
# written together - a key that answers is preferred, counted and has its
# cooldown lifted - and three separate locks let two threads interleave a
# half-updated state. It is reentrant because the helpers below call each other
# while holding it.
_KEY_STATE_LOCK = threading.RLock()


def _quota_window():
    """Which allowance window it is, as a number that changes at the reset.

    Hours since the last reset, so two calls an hour apart across the boundary
    give different numbers and two calls either side of it do not.
    """
    now = time.localtime()
    hours = now.tm_hour - QUOTA_RESET_HOUR
    if hours < 0:
        hours += 24
    return (now.tm_yday, hours)


def _prefer_key(slot):
    env_var = slot.split("#", 1)[0]
    with _KEY_STATE_LOCK:
        _KEY_PREFERENCE[env_var] = slot
    _save_preferences()


def _key_use(slot):
    """How many replies this key has answered in the current allowance window."""
    window = _quota_window()
    with _KEY_STATE_LOCK:
        seen, count = _KEY_USAGE.get(slot, (None, 0))
        return count if seen == window else 0


def _note_key_use(slot):
    """Count one answered reply against this key's window, so it is retired in time."""
    window = _quota_window()
    with _KEY_STATE_LOCK:
        seen, count = _KEY_USAGE.get(slot, (None, 0))
        _KEY_USAGE[slot] = (window, count + 1 if seen == window else 1)


def _providers():
    """One entry per configured key, providers first and keys in order.

    A provider with three keys is three entries sharing an endpoint and a model
    list. Rotating onto the next entry is what a quota refusal, a bad key or a
    key nobody has configured any more looks like from the caller's side.
    """
    # Filter by provider override if set
    providers_to_check = PROVIDERS
    if _FAST_PROVIDER_OVERRIDE:
        provider_map = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        target_env = provider_map.get(_FAST_PROVIDER_OVERRIDE)
        if target_env:
            providers_to_check = [p for p in PROVIDERS if p[0] == target_env]
    
    found = []
    for env_var, base_url, models, label in providers_to_check:
        keys = keys_for(env_var)
        if not keys:
            continue
        # A model named in the environment is a deliberate choice, so it is used
        # on its own. Otherwise every candidate is fair game and the request can
        # move on to the next one if this model has been retired.
        chosen = _key_for(f"{env_var}_MODEL")
        # Model discovery is provider-wide. Resolve it once instead of repeating
        # the same cache lookup (or failed discovery log) for every configured key.
        discovered_models = None if chosen else _discover_models(
            env_var, base_url, models, label
        )
        for index, key in enumerate(keys):
            found.append({
                "env_var": env_var,
                "key": key,
                # Which key this is, for the log and for the cooldown bookkeeping.
                # A model that is out of room on one key is in room on the next,
                # so the cooldown is kept per key and never per model alone.
                "slot": f"{env_var}#{index + 1}",
                "key_number": index + 1,
                "key_count": len(keys),
                "base_url": base_url.rstrip("/"),
                "models": (chosen,) if chosen else discovered_models,
                # The rest of this provider's list, kept even when one model was
                # named. Naming a model says which one to prefer; it does not say
                # to sit on a dead provider while the local model takes two
                # minutes to answer, which is the situation this is for.
                "fallbacks": tuple(models) if chosen else discovered_models,
                "pinned": bool(chosen),
                "label": label,
            })
    # Only this provider's keys move, and only inside their own group: another
    # provider stays where it was in PROVIDERS. Inside the group the order is
    # keys that still have room today, then the key with the most of its day
    # left, then the one that answered last, then the order they were written
    # in. Moving on from a key that is nearly spent is the point: the switch
    # happens while she can still answer, rather than after a refusal that
    # costs a whole reply.
    #
    # The providers keep the order they were configured in. They used to be
    # walked through a set, whose order is whatever the hash happens to be on
    # this run, so Groq was sometimes tried before Gemini for no reason at all.
    groups = {}
    order = []
    for entry in found:
        env_var = entry["env_var"]
        if env_var not in groups:
            groups[env_var] = []
            order.append(env_var)
        groups[env_var].append(entry)
    ordered = []
    for env_var in order:
        # Stay on the key that last answered until it stops working. Only when
        # it is out of room or out of budget does the list fall through to the
        # next key, and then stay there until that one fails too.
        preferred = _KEY_PREFERENCE.get(env_var)
        ordered.extend(sorted(
            groups[env_var],
            key=lambda e: (
                2 if e["slot"] in _KEY_BAD_SESSION else 0,
                0 if (e["slot"] == preferred and _key_use(e["slot"]) < KEY_DAILY_BUDGET) else 1,
                1 if _key_use(e["slot"]) >= KEY_DAILY_BUDGET else 0,
                e["key_number"],
            ),
        ))
    return ordered


def _provider():
    """The first provider with a key behind it, or None when there is none."""
    found = _providers()
    return found[0] if found else None


def available():
    """True when a key is configured, so a caller can skip the local detour."""
    return bool(_providers())


def describe():
    """A short line for the startup banner, or None when this is switched off."""
    provider = _provider()
    if provider is None:
        return None
    keys = (
        f" ({provider['key_count']} keys)" if provider["key_count"] > 1 else ""
    )
    return f"{provider['label']} {provider['models'][0]}{keys}"


def _translate(messages):
    """Turn Ollama-shaped messages into OpenAI-shaped ones.

    Two things differ, and the second one only shows up on the second turn.
    Ollama names a tool result with tool_name and lets the id go, while the
    hosted API wants name plus a tool_call_id and rejects the request without
    one. And the assistant turn that made the call comes back too: Ollama leaves
    it as function.name with a dict of arguments, but the hosted API wants an
    id, a type, and the arguments as a JSON text. The browser loop keeps the
    assistant message and sends it straight back on the next turn, so a call
    left in Ollama's shape fails the second request with a complaint about the
    first one.

    The ids are handed out and matched up in a single forward pass for the same
    reason. A result is tied to its call by id, so an invented id on the result
    and an invented id on the call have to be the same invented id, which only
    walking the conversation once in order can guarantee.
    """
    translated = []
    # Calls made and not yet answered, oldest first: (name, id).
    pending = []
    # A counter for the ids made up here, because the number of unanswered
    # calls is not an id: it goes back down as the results come in, so two calls
    # in one conversation were both handed "call_1" and the two results could
    # not be told apart.
    call_seq = 0
    used_ids = set()
    for message in messages:
        entry = dict(message)
        if entry.get("role") == "tool":
            name = entry.get("tool_name") or entry.get("name") or "tool"
            entry["name"] = name
            entry["tool_call_id"] = _match_call(pending, name)
            entry.pop("tool_name", None)
        elif entry.get("role") == "assistant" and entry.get("tool_calls"):
            calls = []
            for call in entry["tool_calls"]:
                function = call.get("function") or {}
                call_id = str(call.get("id") or "").strip()
                if not call_id:
                    call_seq += 1
                    call_id = f"call_{call_seq}"
                    while call_id in used_ids:
                        call_seq += 1
                        call_id = f"call_{call_seq}"
                used_ids.add(call_id)
                pending.append((function.get("name", ""), call_id))
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": _as_arguments_string(function.get("arguments")),
                    },
                })
            entry["tool_calls"] = calls
        else:
            entry.pop("tool_name", None)
        translated.append(entry)
    return translated


def _match_call(pending, name):
    """The id of the oldest unanswered call to this tool.

    Falling back to a made up id is the right last resort rather than an error:
    the browser loop already has code for a result it cannot attribute, and the
    result text itself is still the useful part of the message.
    """
    for index, (called, call_id) in enumerate(pending):
        if called == name:
            del pending[index]
            return call_id
    return f"call_unmatched_{name}"


def _as_arguments_string(arguments):
    """The hosted API wants the arguments as JSON text, not as a dict."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments or {})
    except (TypeError, ValueError):
        return "{}"


def _normalise_tool_calls(raw_calls):
    """Ollama shapes a tool call as function.name plus a dict of arguments.

    The hosted API sends the arguments as a JSON string, and the browser loop
    already copes with either, but normalising here means one shape to reason
    about and no chance of a string leaking into click_page_element's argument.
    """
    calls = []
    for call in raw_calls or []:
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                # Half a tool call is worse than none: the browser loop would
                # report every required argument as missing instead of being
                # able to act on the ones that did survive.
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({"function": {"name": name, "arguments": arguments}})
    return calls


def _gemini_contents(messages):
    """Ollama-shaped messages as Gemini's own contents and system instruction.

    Only used when tools are in play. Gemini will not carry a tool conversation
    over the OpenAI compatible endpoint, because that layer drops the
    thought_signature it hands back with every function call and then refuses
    the next turn without it. So the browser loop is spoken to in Gemini's own
    dialect, and the signature rides along inside the tool call it belongs to.
    """
    system = []
    contents = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system.append({"text": str(message.get("content") or "")})
        elif role == "tool":
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": message.get("tool_name") or message.get("name") or "tool",
                        "response": {"result": str(message.get("content") or "")},
                    }
                }],
            })
        elif role == "assistant" and message.get("tool_calls"):
            parts = []
            for call in message["tool_calls"]:
                part = {"functionCall": {
                    "name": (call.get("function") or {}).get("name", ""),
                    "args": (call.get("function") or {}).get("arguments") or {},
                }}
                signature = call.get("thought_signature")
                if signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            if message.get("content"):
                parts.insert(0, {"text": str(message["content"])})
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({"role": role if role in ("user", "model") else "user",
                             "parts": [{"text": str(message.get("content") or "")}]})
    return system, contents


# Anything that looks like a key or a token is redacted before a provider's
# answer goes into a log: the body of an error is not ours to print, and a
# provider that echoes the request back does not get to print our key with it.
_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:sk|api[_-]?key|key|token|bearer|aiza)[-_: =]*[A-Za-z0-9_\-]{8,}"
)


def _safe_error_text(response, limit=200):
    """A short, sanitized description of what the provider said.

    The status code and the provider's own message are enough to tell one
    failure from another. The raw body is not, and it is the part that ends up
    in a log that gets pasted around, so it is reduced to a message with
    anything key-shaped taken out of it.
    """
    message = ""
    try:
        parsed = response.json()
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif error:
            message = str(error)
        if not message:
            message = str(parsed.get("message") or "")
    if not message:
        try:
            message = " ".join((response.text or "").split())
        except Exception:
            message = ""
    message = _SECRET_PATTERN.sub("<redacted>", message)
    message = " ".join(message.split())
    if len(message) > limit:
        message = message[:limit].rstrip() + "..."
    return message or "no message"


def _chat_gemini_native(provider, model, messages, tools, tool_choice, max_tokens,
                       temperature, top_p, timeout, json_mode):
    """One tool-calling turn against Gemini's own API rather than the compat one."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    system, contents = _gemini_contents(messages)
    declarations = []
    for tool in tools or []:
        function = tool.get("function") or tool
        declarations.append({
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        })
    body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": int(max_tokens),
            "temperature": float(temperature),
            "topP": float(top_p),
        },
        "tools": [{"functionDeclarations": declarations}],
        # "ANY" is what makes it act rather than describe what it would do, which
        # is the same instruction the OpenAI form gets from tool_choice=required.
        "tool_config": {"function_calling_config": {
            "mode": "ANY" if tool_choice else "AUTO"}},
    }
    if system:
        body["systemInstruction"] = {"parts": system}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    try:
        response = _SESSION.post(
            url,
            headers={"x-goog-api-key": provider["key"], "Content-Type": "application/json"},
            json=body,
            timeout=(CONNECT_TIMEOUT, timeout),
        )
    except requests.exceptions.Timeout as error:
        raise FastAITimeout(
            f"{provider['label']} did not answer in {timeout}s: {error}"
        ) from error
    except requests.exceptions.RequestException as error:
        raise FastAIError(f"{provider['label']} could not be reached: {error}") from error

    if _model_retired(response.status_code, response.text):
        if response.status_code == 429:
            raise _TryNext(_quota_refusal(model, response.text))
        raise _TryNext(f"{model} is not on {provider['label']} any more")
    if response.status_code >= 400:
        raise FastAIError(
            f"{provider['label']} answered {response.status_code}: "
            f"{_safe_error_text(response)}"
        )

    try:
        candidate = response.json()["candidates"][0]
        parts = candidate["content"]["parts"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise FastAIError(f"{provider['label']} sent an unreadable reply: {error}") from error

    texts, calls = [], []
    for part in parts:
        if part.get("functionCall"):
            call = part["functionCall"]
            if not call.get("name"):
                continue
            entry = {"function": {
                "name": call["name"],
                "arguments": call.get("args") if isinstance(call.get("args"), dict) else {},
            }}
            # Kept beside the call it belongs to, so the next turn can hand it
            # back and Gemini will accept the conversation.
            if part.get("thoughtSignature"):
                entry["thought_signature"] = part["thoughtSignature"]
            calls.append(entry)
        elif part.get("text"):
            texts.append(part["text"])
    return {
        "text": "".join(texts),
        "tool_calls": calls,
        "finish_reason": "stop",
    }


def _ask_one(provider, model, messages, tools, tool_choice, max_tokens,
             temperature, top_p, timeout, stop, json_mode):
    """One attempt against one model, in whichever shape that provider wants."""
    if tools and provider["label"] == "Gemini":
        return _chat_gemini_native(
            provider, model, messages, tools, tool_choice, max_tokens,
            temperature, top_p, timeout, json_mode,
        )
    return _ask(provider, model, _translate(messages), tools, tool_choice, max_tokens,
                temperature, top_p, timeout, stop, json_mode)


def _try_provider_fallbacks(provider, models, messages, tools, tool_choice,
                          max_tokens, temperature, top_p, timeout, stop, json_mode,
                          last_error):
    """Try the rest of one provider's list after the named model ran out of room.

    Stays inside the provider on purpose. Being told "use Gemini" is not consent
    to answer from somewhere else, but it is also not a reason to sit on a dead
    key while the local model takes minutes, so the same service's next model is
    fair game and anything beyond it is not.
    """
    slot = provider.get("slot", "")
    # Check if the original error was a hard quota - if so, the whole key is out
    original_was_quota = _is_hard_quota(last_error) or _is_quota(last_error)
    for model in _skip_exhausted(models, slot):
        try:
            answer = _ask_one(provider, model, messages, tools, tool_choice, max_tokens,
                              temperature, top_p, timeout, stop, json_mode)
        except _TryNext as skip:
            last_error = skip
            if _is_hard_quota(skip):
                _note_quota(model, skip, slot)
            continue
        except FastAIError as error:
            last_error = _TryNext(str(error))
            continue
        _note_working(model, slot)
        print(
            f"{provider['models'][0]} is out of room, so {model} answered instead "
            f"({provider['label']}). Unpinning it in .env puts the named one back "
            "first."
        )
        return answer
    # If the pinned model hit quota and no fallback worked, the whole key is out
    if original_was_quota:
        raise _KeyOutOfRoom(f"{provider['models'][0]} quota exhausted on this key")
    raise FastAIError(
        f"{provider['models'][0]} is out of room and no other {provider['label']} "
        f"model answered either: {last_error}"
    )


def describe_image(image_b64, prompt="Describe this image in detail.", max_tokens=256,
                   temperature=0.3, timeout=DEFAULT_TIMEOUT):
    """Ask the fast model to describe a base64-encoded image.

    Returns the text description, or raises FastAIError. Uses the OpenAI
    compatible image_url form so any provider that supports vision can answer.
    """
    if not available():
        raise FastAIError("no fast provider key is set")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64.strip()}",
            }},
        ],
    }]
    answer = chat(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    return answer.get("text", "").strip()


def chat(messages, tools=None, tool_choice=None, max_tokens=512, temperature=0.8,
         top_p=0.9, timeout=DEFAULT_TIMEOUT, stop=None, json_mode=False):
    """Ask the fast model, and answer in Ollama's shape.

    Returns {"text", "tool_calls", "finish_reason"}. Raises FastAIError for
    anything that went wrong, because every caller here has a local model to fall
    back to and a half answer is worse than a slower one.
    """
    providers = _providers()
    if not providers:
        try:
            return omniroute.chat(
                messages, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, timeout=timeout, json_mode=json_mode,
                tools=tools, tool_choice=tool_choice, stop=stop,
            )
        except Exception as omni_error:
            raise FastAIError(
                f"no fast provider key is set and OmniRoute could not answer: "
                f"{omni_error}"
            ) from omni_error


    last_error = None
    for provider in providers:
        try:
            return _chat_one_key(
                provider, messages, tools, tool_choice, max_tokens,
                temperature, top_p, timeout, stop, json_mode,
            )
        except FastAIError as error:
            # Every failure earns the next entry in the list. That entry is
            # either another key of this provider or another provider
            # entirely, and the whole point of the list is that one of them
            # can answer. The old code gave up on all of them the moment the
            # error was not about the key, so a provider that was briefly down
            # took the whole reply with it and the local model answered.
            last_error = error
            if isinstance(error, _KeyOutOfRoom):
                # Nothing was even asked of this key, so there is nothing to
                # report: the next entry is simply the next place to try.
                continue
            later = next(
                (
                    other for other in providers
                    if other["env_var"] == provider["env_var"]
                    and other["key_number"] > provider["key_number"]
                ),
                None,
            )
            reason = (
                "this key cannot serve it" if _is_key_problem(error)
                else "the provider did not answer"
            )
            error_key = (provider["env_var"], provider["key_number"], str(error))
            now = time.monotonic()
            last_print = _RECENT_ERRORS.get(error_key, 0.0)
            if now - last_print >= _DEDUP_WINDOW_SECONDS:
                recharge = ""
                if _is_quota(error):
                    recharge = _recharge_text(provider["slot"], provider["models"])
                if later is None:
                    print(
                        f"{provider['label']} key {provider['key_number']} of "
                        f"{provider['key_count']} cannot serve this ({reason}): "
                        f"{error}. Moving to the next provider.{recharge}"
                    )
                else:
                    print(
                        f"{provider['label']} key {provider['key_number']} of "
                        f"{provider['key_count']} cannot serve this ({reason}): {error}. "
                        f"Moving to key {later['key_number']}.{recharge}"
                    )
                _RECENT_ERRORS[error_key] = now
            continue
    # OmniRoute is the final hosted fallback after all configured providers.
    try:
        return omniroute.chat(messages, max_tokens=max_tokens, temperature=temperature,
                              top_p=top_p, timeout=timeout, json_mode=json_mode,
                              tools=tools, tool_choice=tool_choice, stop=stop)
    except Exception as omni_error:
        print(f"OmniRoute fallback also could not answer: {omni_error}")
    raise FastAIError(str(last_error))


def _chat_one_key(provider, messages, tools, tool_choice, max_tokens,
                  temperature, top_p, timeout, stop, json_mode):
    """Ask this one key, in this provider's order of preference.

    Split out of chat() because with more than one key behind a provider the
    question is no longer "which model" but "which key, then which model".
    """
    slot = provider.get("slot", "")
    last_error = None
    # A key that already returned a hard key error this session is never worth
    # trying again - skip it immediately instead of paying a full timeout on
    # every model before moving to the next key.
    if slot in _KEY_BAD_SESSION:
        raise _KeyOutOfRoom(
            f"{slot} already failed this session, skipping to the next key"
        )
    tried = _skip_exhausted(provider["models"], slot)
    if provider["pinned"] and not tried:
        # Every candidate was already found out of room on an earlier call, so
        # nothing was even attempted here. Saying so is what lets the pinned
        # model's quota failure reach the fallback path below - without it, the
        # pinned branch raised the string "None" and the local model answered
        # every message for the next two minutes.
        last_error = _TryNext(
            f"{provider['models'][0]} is out of quota right now"
        )
    for model in tried:
        try:
            if tools and provider["label"] == "Gemini":
                answer = _chat_gemini_native(
                    provider, model, messages, tools, tool_choice, max_tokens,
                    temperature, top_p, timeout, json_mode,
                )
            else:
                answer = _ask(provider, model, _translate(messages), tools, tool_choice,
                              max_tokens, temperature, top_p, timeout, stop, json_mode)
        except _TryNext as skip:
            # Either that model is gone or it is out of room. Both are worth
            # remembering: the room coming back is not something to keep
            # rediscovering one failed call at a time.
            last_error = skip
            if _is_hard_quota(skip):
                _note_quota(model, skip, slot)
            continue
        except FastAIError as error:
            # A key-level problem (401, invalid key, revoked) means this key is
            # dead for this session - mark it and move to the next key at once
            # instead of burning a timeout on every remaining model. Quota and
            # high-demand errors are temporary: they move to the next key but
            # this key stays in the running for the next call.
            last_error = error
            if _is_key_problem(error) and not _is_quota(error):
                _KEY_BAD_SESSION.add(slot)
                break
            continue
        _note_working(model, slot)
        return answer
    if provider["pinned"]:
        # They named this model on purpose, so say what was wrong with it rather
        # than quietly answering from somewhere they did not ask for.
        #
        # Out of room is the one exception, because that is an outage rather than
        # a choice: measured here at 13:15 on a Gemini free tier pinned to
        # gemini-3.5-flash-lite, which answered 429 "you exceeded your current
        # quota" in 0.4s while gemini-flash-latest answered the same prompt in
        # 5.0s. Refusing to move then means every reply goes to a local model
        # that needs 167s from a cold cache - so a quota failure on the named
        # model tries the rest of *that provider's* list, and says so. Anything
        # else still reports rather than substituting.
        if _is_quota(last_error):
            return _try_provider_fallbacks(
                provider, [m for m in provider.get("fallbacks", ()) if m not in provider["models"]],
                messages, tools, tool_choice, max_tokens, temperature, top_p, timeout,
                stop, json_mode, last_error,
            )
    if not tried:
        # Nothing was even asked: every model on this key was already found out
        # of room, which was said when it happened. Marked so the caller can
        # move to the next key quietly, instead of writing the same line again
        # for every message until the quota comes back.
        raise _KeyOutOfRoom(
            f"every {provider['label']} model is out of quota right now"
        )
    raise FastAIError(
        f"none of {provider['label']}'s models answered: {last_error}"
    )


class _TryNext(Exception):
    """This model cannot serve the request right now. Try the next candidate."""


class _KeyOutOfRoom(FastAIError):
    """This key is already known to be spent, and nothing was asked of it."""


def _is_quota(error):
    """Was this one of them being out of room, rather than gone for good?

    Overload counts. A 503 "this model is currently experiencing high demand" is
    a queue at Google's end, not a model that has been switched off, and being
    told otherwise had two costs: the message said a model "is not on Gemini any
    more", which was simply untrue, and the model was remembered as out of room
    for two minutes, so a provider that had been busy for a few seconds stayed
    unusable long after it had stopped being busy.
    """
    text = str(error).lower()
    return (
        "out of quota" in text
        or "quota exhausted" in text
        or "rate limited" in text
        or "high demand" in text
        or "overloaded" in text
        or "resource_exhausted" in text
    )


def _is_key_problem(error):
    """Is this something about the key, rather than about the request?

    Those are the only failures another key can fix, and they are the ones the
    second key is there for: out of room until the quota resets, or a key that
    was revoked and will never work again. Both are worth one more try elsewhere.
    A timeout, a malformed request or an unreadable reply is not - it happens
    the same way through every key, so the next one only adds a wait.
    """
    if _is_hard_quota(error) or _is_quota(error):
        return True
    text = str(error).lower()
    return any(
        phrase in text
        for phrase in (
            "api key not valid",
            "api key expired",
            "invalid_api_key",
            "unauthorized",
            "permission denied",
            " 401",
            " 403",
        )
    )


def _retry_delay_seconds(text):
    """How long the provider says it will be out of room for, in seconds.

    Google puts it in the 429 twice: a google.rpc.RetryInfo detail
    (`"retryDelay": "39658s"`) and a sentence in the message ("Please retry in
    11h58.8s"). Measured on this key: a per-day limit of 500 requests, refused
    with retryDelay 39658s. Two values for the same fact that disagree by an
    hour, so the smaller is used - being early costs one wasted request, being
    late means sitting on a dead model. Only plain seconds and h/m/s are read,
    and anything implausible is ignored in favour of the default cooldown.
    """
    # The last pattern reads back this module's own "(they say it is back in
    # 10h 58m)" note, so a delay survives being passed around inside an error
    # message - which is how it reaches the place that remembers it.
    for pattern in (
        r'"retryDelay"\s*:\s*"(\d+)s"',
        r"retry in (\d+)h(?:([\d.]+)m)?",
        r"back in (\d+)h (\d{2})m",
    ):
        match = re.search(pattern, str(text))
        if not match:
            continue
        if "h" in pattern:
            hours = int(match.group(1))
            minutes = float(match.group(2) or 0)
            return int(hours * 3600 + minutes * 60)
        return int(match.group(1))
    return None


def _quota_span(text):
    """Which limit was hit: the day's allowance, or the per-minute one.

    Measured here: a key refused with the day's allowance gone names
    GenerateRequestsPerDayPerProjectPerModel and says "back in 10h 58m", while
    the same key seconds later answers in 1.1s. Both arrive as a bare 429, and
    treating the short one as the long one parked a working key for the better
    part of an hour - the cooldown and the "is out of quota" wording both belong
    to the day's allowance, not to a minute's worth of requests.
    """
    lowered = str(text).lower()
    # Hyphens go too, because this is also read back out of the sentence built
    # above - "over the per-minute limit" - where the word is hyphenated. That is
    # the only copy of the limit's name that survives being passed around inside
    # an error message, so it has to answer the question as well as the raw body.
    squashed = lowered.replace(" ", "").replace("_", "").replace("-", "")
    if "perday" in squashed or "per day" in lowered or "daily" in lowered:
        return "day"
    if "perminute" in squashed or "per minute" in lowered or "tokens per minute" in lowered:
        return "minute"
    return None


def _quota_refusal(model, body):
    """The sentence for a 429, which says which limit it actually was."""
    if _quota_span(body) == "minute":
        return f"{model} is over the per-minute limit right now"
    return f"{model} is out of quota right now" + _retry_delay_text(body)


def _retry_delay_text(body):
    """A human-readable note of when to try again, from the provider's own hint."""
    seconds = _retry_delay_seconds(body)
    if not seconds:
        return ""
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    return f" (they say it is back in {hours}h {minutes:02d}m)"


def _recharge_text(slot, models):
    """A human-readable note of when this key's models will be back, if any are cooling down."""
    now = time.monotonic()
    with _KEY_STATE_LOCK:
        remaining = []
        for model in models:
            until = _exhausted_until.get(_quota_key(slot, model), 0.0)
            if until > now:
                seconds = int(until - now)
                hours, rest = divmod(seconds, 3600)
                minutes = rest // 60
                remaining.append(f"{model}: {hours}h {minutes:02d}m")
    if not remaining:
        return ""
    return " (recharge: " + ", ".join(remaining) + ")"


def _is_hard_quota(error):
    """Is this one out of room for a while, rather than busy for a moment?

    The two are told apart because they are remembered differently. A daily quota
    is gone for hours, so a model that refused for one is remembered and not asked
    again for a couple of minutes. "High demand" is a queue at the provider's end
    that clears in seconds: remembering that turned one busy minute into two
    minutes of the local model answering every message, because both of Gemini's
    other models were on the same cooldown at once.
    """
    text = str(error).lower()
    return (
        "out of quota" in text
        or "rate limit" in text
        or "resource_exhausted" in text
    )


def _model_retired(status_code, text):
    """Should this failure be retried against the next candidate?

    Retired models and out-of-capacity models are both worth another try
    somewhere else, and so is a rate limit, which is the one that surprised
    this: Google's quota is counted per model, not per account, and it was
    measured here returning 429 for the two larger Flash models in the same
    second that the small one answered in 1.3s. Treating a 429 as fatal therefore
    threw away a working fast model because an unrelated one was out of room,
    and dropped her onto the local one at 0.3 tokens a second for no reason.

    What is deliberately still not retried: a 401 or 403, which is the key being
    wrong, and a 400 about the request itself. Walking past those would only
    report a different error than the one that actually happened.

    503 (service unavailable) is NOT treated as retired - it's a transient
    capacity issue that often clears in seconds. Retrying the same model or
    another key is better than permanently marking it retired.
    """
    if status_code in (404, 429, 500, 502, 504):
        return True
    if status_code != 400:
        return False
    lowered = text.lower()
    return "model" in lowered and any(
        word in lowered
        for word in ("not found", "does not exist", "not available", "deprecat", "invalid model", "unknown model")
    )


# A model that just refused for quota is very likely to refuse again for the next
# while, and walking past two of them on every reply is a second of wasted round
# trips before reaching one that works. So remember which ones are out of room,
# and for how long. Long enough to cover a burst of replies, short enough that a
# quota that resets is picked up again without anyone restarting anything.
EXHAUSTED_FOR_SECONDS = 30
# A per-minute limit is back within the minute, so that is how long the key is
# left alone. Longer than that and the wait is longer than the limit.
RATE_LIMIT_FOR_SECONDS = 75
# Never sit on a model for longer than this, whatever the provider says. A day is
# a long time to not try a model because one number in a 429 said so.
EXHAUSTED_MAX_SECONDS = 3600
_exhausted_until = {}

# Keys that returned a hard key error (401, invalid key, revoked) during this
# session. They are skipped without even trying, so a dead key never costs a
# full timeout before she moves to the next one.
_KEY_BAD_SESSION = set()


def _quota_key(slot, model):
    """One model's name, plus which key was refused, for the cooldown table.

    Quota is counted per key, not per model name, so the two have to be told
    apart: a model refused on key 1 is in room on key 2, and remembering it as
    gone for the whole provider would take the working key out of the running
    just because a neighbour ran out of room. Without a slot this is still the
    bare model name, which is how everything that asks about one model alone
    keeps reading the table it always did.
    """
    return f"{slot}|{model}" if slot else model


def _skip_exhausted(models, slot=""):
    """The candidates worth asking right now, in order.

    Only the entries being asked about are looked at. Sweeping the whole
    table on every call made every reply pay for every key's cooldowns, and
    the table's own expiry does the rest of the work: an entry whose time has
    passed is not in the way any more.
    """
    now = time.monotonic()
    with _KEY_STATE_LOCK:
        return [
            name for name in models
            if _exhausted_until.get(_quota_key(slot, name), 0.0) <= now
        ]


# How often the cooldown table is swept of the entries whose time has passed.
# The sweep used to happen on every _skip_exhausted call, which put the whole
# table on the hot path of every reply; doing it once every so often on a write
# keeps the table small without that cost.
SWEEP_EVERY_WRITES = 32
_sweep_count = 0


def _sweep_expired_locked(now):
    """Drop the cooldown entries whose time has passed. The lock is held."""
    global _sweep_count
    _sweep_count += 1
    if _sweep_count % SWEEP_EVERY_WRITES:
        return
    for name in [n for n, until in _exhausted_until.items() if until <= now]:
        del _exhausted_until[name]


# Slots that have refused for quota, so that a key coming back is visible rather
# than silent. Without this, a key that recovers at 07:00 is simply the one that
# answers again, indistinguishable from a key that never ran out - and the only
# way to tell whether the rotation is working is to read every line of the log.
_KEY_WAS_SPENT = set()


def _note_quota(model, error, slot=""):
    """Remember a refusal for as long as that limit lasts, and no longer.

    The day's allowance is remembered for the provider's own figure, capped at
    EXHAUSTED_MAX_SECONDS. A per-minute limit is remembered for about a minute
    and is not treated as the key running out at all - it is not, and saying so
    is what used to take a working key out of the running for the rest of the
    hour.
    """
    seconds = _retry_delay_seconds(str(error))
    if _quota_span(error) == "minute":
        _note_exhausted(
            model, min(seconds or RATE_LIMIT_FOR_SECONDS, RATE_LIMIT_FOR_SECONDS),
            slot, spent=False, floor=RATE_LIMIT_FOR_SECONDS,
        )
        return
    _note_exhausted(model, seconds, slot)


def _note_exhausted(model, seconds=None, slot="", spent=True, floor=None):
    """Remember a model as out of room, for as long as the provider said it would be.

    The provider's own number, where there is one. A two minute default applied
    to an eleven hour quota asked the same dead model 330 times before the quota
    came back; the 429s cost nothing against the quota, but every one of them was
    a request, and the log filled with a refusal nobody needed to see again.
    """
    if seconds is None:
        seconds = _retry_delay_seconds(str(seconds)) or 0
    # The floor is what covers a refusal with no time in it. A per-minute limit
    # passes its own, because two minutes is longer than the limit it is waiting
    # out, and the key is fine the moment the minute is up.
    wait = max(EXHAUSTED_FOR_SECONDS if floor is None else floor,
               min(seconds, EXHAUSTED_MAX_SECONDS))
    now = time.monotonic()
    with _KEY_STATE_LOCK:
        _exhausted_until[_quota_key(slot, model)] = now + wait
        _sweep_expired_locked(now)
        if slot and spent:
            _KEY_WAS_SPENT.add(slot)


def _note_working(model, slot=""):
    with _KEY_STATE_LOCK:
        _exhausted_until.pop(_quota_key(slot, model), None)
        was_spent = bool(slot) and slot in _KEY_WAS_SPENT
        if was_spent:
            _KEY_WAS_SPENT.discard(slot)
    if slot:
        _prefer_key(slot)
        _note_key_use(slot)
        if was_spent:
            print(
                f"{slot} is answering again, so its quota is back and she is on "
                f"it. With every key sorted by what is left of its day, the one "
                f"that recovered is chosen first."
            )


def _ask(provider, model, messages, tools, tool_choice, max_tokens, temperature,
         top_p, timeout, stop, json_mode):
    """One request against one model ID."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    if stop:
        body["stop"] = stop
    if json_mode:
        # Ollama's format field pins the output to a JSON schema, so her reply
        # and thought envelopes came back parseable every time. The hosted API
        # has no equivalent for an arbitrary schema, but it can be asked for
        # JSON as such, which is enough for the reply parser and its existing
        # fallbacks to have something to work with.
        body["response_format"] = {"type": "json_object"}

    try:
        response = _SESSION.post(
            f"{provider['base_url']}/chat/completions",
            headers=_provider_headers(provider, provider["key"],
                                   f"{provider['base_url']}/chat/completions"),
            json=body,
            timeout=(CONNECT_TIMEOUT, timeout),
        )
    except requests.exceptions.Timeout as error:
        raise FastAITimeout(
            f"{provider['label']} did not answer in {timeout}s: {error}"
        ) from error
    except requests.exceptions.RequestException as error:
        raise FastAIError(f"{provider['label']} could not be reached: {error}") from error

    if _model_retired(response.status_code, response.text):
        if response.status_code == 429:
            raise _TryNext(_quota_refusal(model, response.text))
        raise _TryNext(f"{model} is not on {provider['label']} any more")
    if response.status_code >= 400:
        raise FastAIError(
            f"{provider['label']} answered {response.status_code}: "
            f"{_safe_error_text(response)}"
        )

    try:
        parsed = response.json()
        choice = parsed["choices"][0]
        message = choice.get("message") or {}
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise FastAIError(f"{provider['label']} sent an unreadable reply: {error}") from error

    return {
        "text": message.get("content") or "",
        "tool_calls": _normalise_tool_calls(message.get("tool_calls")),
        # "length" is the one value the callers already understand, from the
        # streaming path: it means the answer was cut off and is worth retrying.
        "finish_reason": "length" if choice.get("finish_reason") == "length" else "stop",
    }


# ---------------------------------------------------------------------------
# Anime detection
# ---------------------------------------------------------------------------
# A keyword list can never hold every anime in existence - there are tens of
# thousands of series and the list here was already out of date the moment it
# was written. So this asks the fast model instead, which knows them all. The
# call is tiny and cached, and on any failure it falls back to the keyword
# check, so a provider hiccup never costs the user their drawing.
_ANIME_CACHE = {}
_ANIME_CACHE_LOCK = threading.Lock()


def is_anime_prompt(prompt):
    """Is this prompt asking for an anime-style image?

    Returns True, False, or None when it cannot tell. The None case is the
    signal to the caller to fall back to its own keyword list, so this is a
    strict improvement rather than a replacement: it knows more, and it never
    knows less.

    The question is asked in one shot rather than "name the show" because a
    user typing "draw Levi from Attack on Titan" or "make me a waifu" is
    describing what they want, not testing a catalogue, and a yes/no answer
    is all the drawing code needs to add the style boost.
    """
    if not prompt or not available():
        return None
    key = prompt.strip().lower()
    with _ANIME_CACHE_LOCK:
        if key in _ANIME_CACHE:
            return _ANIME_CACHE[key]
    try:
        answer = chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decide whether a user's image prompt is asking for "
                        "anime, manga, or Japanese cartoon style. Answer exactly "
                        "one word: YES or NO. A prompt is YES if it names a "
                        "character or show from any anime or manga, or uses "
                        "anime-style terms like waifu, kawaii, senpai, or asks "
                        "for anime art style. It is NO for real people, photos, "
                        "landscapes, objects, or anything not anime-related."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
            temperature=0.0,
            timeout=20,
        ).get("text", "").strip().upper()
    except Exception as error:
        print(f"fastai: anime detection failed, falling back to keywords: {error}")
        return None
    result = answer.startswith("YES")
    with _ANIME_CACHE_LOCK:
        _ANIME_CACHE[key] = result
    return result
