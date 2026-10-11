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

Whichever key is present is used, and the order above is the order of
preference: Gemini, then Groq, then OpenRouter. With none of them set this
module reports itself unavailable and every caller quietly keeps using the local
model, so nothing breaks by adding it and nothing is lost by leaving it out.
"""

import json
import os
import re
import threading
import time

import requests

# Each entry is (env var name, base url, candidate models, label). The
# differ only in host, so one request builder covers them both.
# Each entry is (env var name, base url, candidate models, label). The two URLs
# differ only in host, so one request builder covers them both. A model ID here
# is a candidate rather than a fixed choice because these catalogues get retired
# underneath you: Groq shut llama-3.3-70b-versatile down in August 2026, a month
# after it was their strongest free model, and a request naming it now fails
# outright. Pinning one ID means being wrong the day it is switched off, and the
# symptom is silent, because a failed request falls back to the slow local model
# and the only thing you notice is that she went back to being slow.
PROVIDERS = (
    (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        (
            # Primary: the fast, cheap Flash-Lite (alias to current Flash-Lite)
            "gemini-3.5-flash-lite",
            # Alias rather than a dated ID: Google keeps flash-latest pointing at
            # the current Flash, so it cannot be retired out from under this.
            "gemini-flash-latest",
            # Fallback to 1.5 Flash if newer models are unavailable.
            "gemini-1.5-flash",
            # Smaller 1.5 Flash variant.
            "gemini-1.5-flash-8b",
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
            "openai/gpt-oss-20b",
            # Twice the size for when the smaller one gets something wrong.
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
        ),
        "Groq",
    ),
    (
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        (
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-8b-instant",
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
DEFAULT_TIMEOUT = 30
CONNECT_TIMEOUT = 10

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


# Which key answered last, per provider, so the next reply tries that one first.
# With six keys where five share a spent quota this is worth 2s and five lines of
# log per message: without it every reply walks the dead ones first, and the walk
# is only avoided while their cooldowns last.
_KEY_PREFERENCE = {}
_KEY_PREFERENCE_LOCK = threading.Lock()

# How much of a key's daily allowance to spend before moving on to the next one.
# The free tier is 500 requests a day for GenerateRequestsPerDayPerProjectPerModel,
# measured here as a hard refusal rather than a slowdown, so the last slice is
# left alone: a key that runs out mid-conversation stops the reply instead of
# slowing it, and there is no way to ask Google where the line is before the 429.
KEY_DAILY_BUDGET = int((os.environ.get("KOOKIE_KEY_DAILY_REQUESTS") or "400"))
# The hour a key's allowance starts over. Measured here from the 429s: every one
# of them said the quota was back between 07:00 and 08:00 Cambodia time, not at
# midnight, so a counter that resets at midnight hands a key back seven hours
# early and collects a refusal for each reply in between.
QUOTA_RESET_HOUR = int((os.environ.get("KOOKIE_QUOTA_RESET_HOUR") or "7"))
_KEY_USAGE = {}
_KEY_USAGE_LOCK = threading.Lock()


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


def _preferred_slot(env_var):
    with _KEY_PREFERENCE_LOCK:
        return _KEY_PREFERENCE.get(env_var)


def _prefer_key(slot):
    env_var = slot.split("#", 1)[0]
    with _KEY_PREFERENCE_LOCK:
        _KEY_PREFERENCE[env_var] = slot


def _key_use(slot):
    """How many replies this key has answered in the current allowance window."""
    window = _quota_window()
    with _KEY_USAGE_LOCK:
        seen, count = _KEY_USAGE.get(slot, (None, 0))
        return count if seen == window else 0


def _note_key_use(slot):
    """Count one answered reply against this key's window, so it is retired in time."""
    window = _quota_window()
    with _KEY_USAGE_LOCK:
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
                "models": (chosen,) if chosen else tuple(models),
                # The rest of this provider's list, kept even when one model was
                # named. Naming a model says which one to prefer; it does not say
                # to sit on a dead provider while the local model takes two
                # minutes to answer, which is the situation this is for.
                "fallbacks": tuple(models),
                "pinned": bool(chosen),
                "label": label,
            })
    for env_var in {entry["env_var"] for entry in found}:
        # Only this provider's keys move, and only inside their own group: another
        # provider stays where it was in PROVIDERS. Inside the group the order is
        # keys that still have room today, then the key with the most of its day
        # left, then the one that answered last, then the order they were written
        # in. Moving on from a key that is nearly spent is the point: the switch
        # happens while she can still answer, rather than after a refusal that
        # costs a whole reply.
        group = sorted(
            (e for e in found if e["env_var"] == env_var),
            key=lambda e: (
                1 if _key_use(e["slot"]) >= KEY_DAILY_BUDGET else 0,
                e["key_number"],
            ),
        )
        found = [e for e in found if e["env_var"] != env_var] + group
    return found


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
                call_id = str(call.get("id") or f"call_{len(pending)}")
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
            f"{provider['label']} answered {response.status_code}: {response.text[:200]}"
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
             temperature, top_p, timeout, json_mode):
    """One attempt against one model, in whichever shape that provider wants."""
    if tools and provider["label"] == "Gemini":
        return _chat_gemini_native(
            provider, model, messages, tools, tool_choice, max_tokens,
            temperature, top_p, timeout, json_mode,
        )
    return _ask(provider, model, _translate(messages), tools, tool_choice, max_tokens,
                temperature, top_p, timeout, stop=None, json_mode=json_mode)


def _try_provider_fallbacks(provider, models, messages, tools, tool_choice,
                          max_tokens, temperature, top_p, timeout, json_mode, last_error):
    """Try the rest of one provider's list after the named model ran out of room.

    Stays inside the provider on purpose. Being told "use Gemini" is not consent
    to answer from somewhere else, but it is also not a reason to sit on a dead
    key while the local model takes minutes, so the same service's next model is
    fair game and anything beyond it is not.
    """
    slot = provider.get("slot", "")
    # Check if the original error was a hard quota - if so, the whole key is out
    original_was_quota = _is_hard_quota(last_error) or _is_quota(last_error)
    last_error = last_error
    for model in _skip_exhausted(models, slot):
        try:
            answer = _ask_one(provider, model, messages, tools, tool_choice, max_tokens,
                              temperature, top_p, timeout, json_mode)
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


def chat(messages, tools=None, tool_choice=None, max_tokens=512, temperature=0.8,
         top_p=0.9, timeout=DEFAULT_TIMEOUT, stop=None, json_mode=False):
    """Ask the fast model, and answer in Ollama's shape.

    Returns {"text", "tool_calls", "finish_reason"}. Raises FastAIError for
    anything that went wrong, because every caller here has a local model to fall
    back to and a half answer is worse than a slower one.
    """
    providers = _providers()
    if not providers:
        raise FastAIError("no fast provider key is set")

    last_error = None
    for provider in providers:
        try:
            return _chat_one_key(
                provider, messages, tools, tool_choice, max_tokens,
                temperature, top_p, timeout, stop, json_mode,
            )
        except FastAIError as error:
            # Only the failures that belong to this key earn the next one. A
            # malformed request or an unreadable reply happens the same way
            # through every key, so walking past those would just turn one real
            # error into several and make the wait longer.
            last_error = error
            if not _is_key_problem(error):
                raise
            later = next(
                (
                    other for other in providers
                    if other["env_var"] == provider["env_var"]
                    and other["key_number"] > provider["key_number"]
                ),
                None,
            )
            if later is None:
                raise
            if isinstance(error, _KeyOutOfRoom):
                continue
            print(
                f"{provider['label']} key {provider['key_number']} of "
                f"{provider['key_count']} cannot serve this: {error}. Moving to "
                f"key {later['key_number']}."
            )
    raise FastAIError(str(last_error))


def _chat_one_key(provider, messages, tools, tool_choice, max_tokens,
                  temperature, top_p, timeout, stop, json_mode):
    """Ask this one key, in this provider's order of preference.

    Split out of chat() because with more than one key behind a provider the
    question is no longer "which model" but "which key, then which model".
    """
    slot = provider.get("slot", "")
    last_error = None
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
                json_mode, last_error,
            )
        raise FastAIError(str(last_error))
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
EXHAUSTED_FOR_SECONDS = 120
# A per-minute limit is back within the minute, so that is how long the key is
# left alone. Longer than that and the wait is longer than the limit.
RATE_LIMIT_FOR_SECONDS = 75
# Never sit on a model for longer than this, whatever the provider says. A day is
# a long time to not try a model because one number in a 429 said so.
EXHAUSTED_MAX_SECONDS = 3600
_exhausted_until = {}
_exhausted_lock = threading.Lock()


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
    """The candidates worth asking right now, in order."""
    now = time.monotonic()
    with _exhausted_lock:
        stale = [name for name, until in _exhausted_until.items() if until <= now]
        for name in stale:
            del _exhausted_until[name]
        return [
            name for name in models
            if _quota_key(slot, name) not in _exhausted_until
        ]


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
    with _exhausted_lock:
        _exhausted_until[_quota_key(slot, model)] = time.monotonic() + wait
    if slot and spent:
        _KEY_WAS_SPENT.add(slot)


def _note_working(model, slot=""):
    with _exhausted_lock:
        _exhausted_until.pop(_quota_key(slot, model), None)
    if slot:
        _prefer_key(slot)
        _note_key_use(slot)
        if slot in _KEY_WAS_SPENT:
            _KEY_WAS_SPENT.discard(slot)
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
            headers={
                "Authorization": f"Bearer {provider['key']}",
                "Content-Type": "application/json",
            },
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
    if response.status_code == 429:
        # A free tier running out of room is the expected failure here, not an
        # outage, and the local model is right there. Say so plainly so the log
        # explains the sudden slowdown.
        raise FastAIError(f"{provider['label']} is rate limited right now")
    if response.status_code >= 400:
        raise FastAIError(
            f"{provider['label']} answered {response.status_code}: {response.text[:200]}"
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
