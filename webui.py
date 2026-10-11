"""Kookie on the web - her window in a phone's browser.

Open ./Kookie --web, then visit the address it prints from any
phone on the same network. She talks, her replies stream in,
and everything - the model, the code, her voice - stays on
this box: the phone only needs the LAN.

    Her website is the page in web/index.html: a locked
    door that opens for her access key, and behind it a
    page that chats with each visitor's own AI provider
    using the API key they paste in. The provider calls
    are relayed through this server (/api/provider), so
    a browser's same-origin rules cannot block them.
    The really bad words are starred out before a
    provider hears them and before she says them;
    the milder heat is left as it is said.
    The API below is separate: it still hands out keys
    to projects, and the family key still opens the
    family door for a known Discord account. The
    door's answer travels with every chat turn: her
    own words for who that person is to her.

A tunnel (KOOKIE_WEB_TUNNEL=1) puts her behind
cloudflared, so she answers from anywhere while this
box's own address stays hidden.
"""

import hashlib
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer

import controller
import memory
import voice

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
TTS_CACHE = os.path.join(WEB_DIR, "tts")
PORT = int(os.environ.get("KOOKIE_WEB_PORT", "8080"))
# The family door: this key plus a Discord account she knows.
WEB_FAMILY_KEY = os.environ.get(
    "KOOKIE_WEB_FAMILY_KEY", "kookieitsmeoneofyourfamily10293927")
# A door through the router: cloudflared's quick tunnel gives
# her website an address on the open internet, so a phone can
# reach her from anywhere without a forwarded port. It needs
# cloudflared installed; without it the tunnel door is shut.
TUNNEL = os.environ.get("KOOKIE_WEB_TUNNEL", "")
# With a tunnel up, the website answers through it alone and
# this box listens on itself only - the address it hides never
# appears anywhere. KOOKIE_WEB_BIND=0.0.0.0 also serves the
# local network at the same time.
BIND = os.environ.get("KOOKIE_WEB_BIND", "")
# Her address on the internet is not printed: anyone who
# connects to it can see it, so it is only shown on purpose.
SHOW_PUBLIC_IP = os.environ.get("KOOKIE_WEB_SHOW_PUBLIC_IP", "")
# Custom domain for GitHub Pages (e.g., kookieAI.com)
CUSTOM_DOMAIN = os.environ.get("KOOKIE_WEB_CUSTOM_DOMAIN", "").strip()

os.makedirs(TTS_CACHE, exist_ok=True)

# The keys of her own. Her website hands one to every
# visitor, and her API answers those and nothing else:
# a key is what a visitor is, on the web.
KEYS_FILE = os.path.join(WEB_DIR, "web_keys.json")
# A visitor who left does not need their key kept
# forever, so only the newest ones stay on disk.
KEYS_KEPT = 500
_keys = None
_keys_lock = threading.Lock()

# One chat job at a time keeps her from answering two people at
# once and interleaving her own thoughts; the phone waits its
# turn the same way Discord does.
CHAT_LOCK = threading.Lock()
JOBS = {}
JOBS_LOCK = threading.Lock()


# The whole of her API, in one place, so anybody
# building a project on her can find out how
# without asking her first.
API_DESCRIPTION = {
    "name": "Kookie",
    "get_a_key": "POST /api/issue-key, optionally "
                 'with {"name": "my project"}',
    "send_the_key": "the X-Kookie-Key header, or "
                    "?key= in the address for a "
                    "voice stream",
    "endpoints": {
        "POST /api/ask": "one message in, one answer "
                         'out: {"message": "hi"} answers '
                         'with {"reply": "...", '
                         '"emotion": "..."}',
        "POST /api/chat": 'start a turn, answered with '
                          '{"id": ...}',
        "GET /api/chat/{id}": "her reply as it is "
                              "written, one event per "
                              "token",
        "POST /api/issue-key": "hand out one of her keys",
        "POST /api/provider": "relay one chat turn to "
                               "a visitor's own AI "
                               "provider, using the key "
                               "they paste in",
        "GET /api/memory?visitor=": "the messages a "
                                      "browser has "
                                      "remembered",
        "POST /api/remember": "keep one more message "
                               "for a visitor",
        "POST /api/models": "list the models of a "
                                "visitor's provider",
        "POST /api/dev-id": "check whether a Discord "
                                "account is one of "
                                "her family",
        "GET /api/status": "her mood right now",
        "GET /api/tts?text=": "her voice for a sentence",
        "GET /api/whoami": "who the door thinks you are",
    },
}


def _lan_ips():
    """The addresses a phone on this network can reach her on."""
    import socket

    out = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in out:
                out.append(ip)
    except Exception:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        if ip not in out:
            out.append(ip)
        probe.close()
    except Exception:
        pass
    return out


def _read_keys():
    """The keys on disk, cached. The caller holds the lock."""
    global _keys
    if _keys is None:
        _keys = {}
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for stored, value in loaded.items():
                    # Keys minted before a project could
                    # be named keep no name.
                    _keys[stored] = value if isinstance(
                        value, dict) else {
                            "issued": str(value), "name": ""}
        except (OSError, ValueError):
            pass
    return _keys


def _load_keys():
    """The keys she has handed out, read once."""
    with _keys_lock:
        return _read_keys()


def _is_issued(candidate):
    """Is this one of the keys she handed out?"""
    return bool(candidate) and candidate in _load_keys()


def _mint_key(name=""):
    """A key of her own, for one project of theirs."""
    import secrets

    fresh = secrets.token_hex(16)
    with _keys_lock:
        store = _read_keys()
        store[fresh] = {
            "issued": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime()),
            "name": str(name).strip()[:64],
        }
        if len(store) > KEYS_KEPT:
            for stale in list(store)[: len(store) - KEYS_KEPT]:
                del store[stale]
        try:
            with open(KEYS_FILE, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=1)
        except OSError:
            pass
    return fresh


def web_keys():
    """The keys she has handed out, with their names."""
    return dict(_load_keys())


def web_key_count():
    """How many keys her website has handed out."""
    return len(_load_keys())


# Her memory: every message anyone sends, in one
# file of her own, each marked with the visitor
# it came from, so a browser that comes back
# finds its whole conversation waiting.
MEMORY_FILE = os.path.join(WEB_DIR, "memory.json")
# A memory that grows forever is a memory nobody
# reads, so only the newest messages stay.
MEMORY_KEPT = 2000
_memory_lock = threading.Lock()


def _is_visitor(candidate):
    """A visitor's own id, and nothing else.

    The id marks a memory's place in her file,
    so only a plain id of hex and dashes is
    allowed through.
    """
    return bool(re.fullmatch(
        r"[a-f0-9][a-f0-9-]{6,63}",
        str(candidate or "").strip()))


def _load_memory():
    """Every remembered message, newest last."""
    try:
        with open(MEMORY_FILE, "r",
                  encoding="utf-8") as f:
            stored = json.load(f)
        if isinstance(stored, list):
            return [
                m for m in stored
                if isinstance(m, dict)
                and _is_visitor(m.get("visitor"))
                and m.get("role") in (
                    "user", "assistant")
                and isinstance(m.get("content"), str)
            ]
    except (OSError, ValueError):
        pass
    return []


def _memory_of(visitor):
    """One visitor's remembered messages."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in _load_memory()
        if m.get("visitor") == visitor
    ]


def _remember_message(visitor, role, content):
    """Keep one more message, and let the oldest go."""
    with _memory_lock:
        stored = _load_memory()
        stored.append({
            "visitor": visitor,
            "role": role,
            "content": content,
            "at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime()),
        })
        if len(stored) > MEMORY_KEPT:
            stored = stored[-MEMORY_KEPT:]
        try:
            with open(MEMORY_FILE, "w",
                      encoding="utf-8") as f:
                json.dump(stored, f, indent=1)
        except OSError:
            pass
    # Also save to Kookie's main memory for training
    if role == "user":
        memory.remember(f"User: {content}")
    elif role == "assistant":
        memory.remember(f"AI: {content}")


def _synthesize(text, wav_path):
    """Her voice, as a wav on disk."""
    try:
        voice._synthesize_with_piper(text, wav_path, 1.0)
        return True
    except Exception:
        pass
    try:
        voice._synthesize_with_cli(text, wav_path, 1.0)
        return True
    except Exception:
        return False


class ChatJob:
    """One conversation turn, streamed token by token."""

    def __init__(self):
        self.lock = threading.Lock()
        self.partial = ""
        self.done = threading.Event()
        self.result = None
        self.error = None

    def on_text(self, partial):
        with self.lock:
            self.partial = str(partial)

    def snapshot(self):
        with self.lock:
            return self.partial


def start_chat(message, visitor=None):
    """Answer a message from the phone, off the request thread.

    A visitor who proved they are family is treated as the person
    they are, exactly like on Discord, with their own thread of
    history. Anyone else is a stranger: a tsundere, never family,
    however warmly they talk.
    """
    job = ChatJob()

    def run():
        try:
            if visitor:
                controller.note_discord_person(
                    visitor["id"], visitor["name"],
                    visitor["username"], text=message,
                    said_by_user=True)
                result = controller.ask_ai(
                    message,
                    on_reply_text=job.on_text,
                    speaker_name=visitor["name"],
                    speaker_is_owner=visitor["is_owner"],
                    speaker_id=visitor["id"],
                    on_website=True,
                    tools_enabled=True,
                )
            else:
                result = controller.ask_ai(
                    message,
                    on_reply_text=job.on_text,
                    speaker_name="stranger",
                    speaker_is_owner=False,
                    speaker_id=None,
                    on_website=True,
                    tools_enabled=True,
                )
            if not isinstance(result, dict):
                result = {"reply": str(result), "emotion": "neutral"}
            job.result = result
        except Exception as error:
            job.error = str(error)
        finally:
            job.done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return job


def _pick_reply(data):
    """The answer, out of whichever shape the provider uses."""
    if isinstance(data, str):
        return data.strip()
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if text is None:
            text = choice.get("text")
        if text:
            return str(text).strip()
    content = data.get("content")
    if isinstance(content, list):
        joined = "".join(
            str(block.get("text") or "")
            for block in content if isinstance(block, dict))
        if joined.strip():
            return joined.strip()
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = data.get("output")
    if isinstance(output, list):
        joined = "".join(
            "".join(
                str(part.get("text") or "")
                for part in (item.get("content") or [])
                if isinstance(part, dict))
            for item in output if isinstance(item, dict))
        if joined.strip():
            return joined.strip()
    for key in ("reply", "response", "text", "answer",
                "output", "message", "content", "result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = _pick_reply(value)
            if nested:
                return nested
    nested = _pick_reply(data.get("data"))
    return nested if isinstance(data.get("data"), (str, dict)) else ""


# Who made her. A visitor's own model answers for
# her on the website, so the fact rides along in
# the system prompt of every call, whatever the
# visitor typed in.
CREATOR = (
    "You are Kookie, an AI created by Cookie. "
    "If anyone asks who created you, your creator "
    "is Cookie.")

# The words that are really bad, and only those.
# The milder ones stay as they are said: she is
# not here to police a little heat, only the
# worst of it. Stems, so a word's family goes
# with it (fuck, fucking, fucker, ...).
SEVERE_WORDS = (
    r"\bmotherfuck\w*",
    r"\bfuck\w*",
    r"\bnigg\w*",
    r"\bcunt\w*",
    r"\bfag(?:g?ot)?s?\b",
    r"\bdyke\w*",
    r"\bkike\w*",
    r"\bchink\w*",
    r"\b(?:spic|gook|coon|wetback|beaner)\w*",
    r"\btrann(?:y|ies)\b",
    r"\bretard\w*",
)
_SEVERE = re.compile("|".join(SEVERE_WORDS), re.IGNORECASE)


def _censor(text):
    """The really bad words, starred out."""
    return _SEVERE.sub(
        lambda m: "*" * len(m.group(0)),
        str(text))


def _error_detail(raw, error):
    """The provider's own words for why it refused."""
    detail = ""
    try:
        payload = json.loads(raw or "{}")
        # Some providers (e.g., Gemini) return a list instead of dict
        if isinstance(payload, list) and payload:
            payload = payload[0]
        problem = payload.get("error")
        if isinstance(problem, dict):
            detail = str(problem.get("message") or "")
        elif problem:
            detail = str(problem)
        if not detail:
            detail = str(payload.get("message") or "")
    except (ValueError, json.JSONDecodeError):
        pass
    return detail or (
        f"the provider returned status {error.code}")


def _family_visitor(payload):
    """The family member this call claims to be, or None.

    The page sends the Discord id the developer door
    confirmed, and her people file decides whether that
    account is really family: the one she belongs to,
    or somebody given a place in her family.
    """
    if not isinstance(payload, dict):
        return None
    claim = payload.get("family")
    if not isinstance(claim, dict):
        return None
    ref = str(claim.get("discord") or "").strip()
    if not re.fullmatch(r"\d{17,20}", ref):
        return None
    try:
        visitor = controller.website_visitor(ref)
    except Exception:
        return None
    if visitor and (visitor.get("is_owner")
                    or visitor.get("relation")):
        return visitor
    return None


def _relay(payload):
    """One chat turn, carried to the visitor's own AI provider.

    The page hands this server the address, key and model
    its visitor pasted in, and the provider answers this
    server instead of the browser: a browser cannot always
    speak to a provider directly, but this box always can.
    """
    cfg = payload.get("cfg") if isinstance(payload, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not messages:
        return None, "say something first"
    # The worst words never reach a provider:
    # starred out before they leave her box.
    for message in messages:
        if (isinstance(message, dict)
                and isinstance(message.get("content"), str)):
            message["content"] = _censor(
                message["content"])
    max_tokens = payload.get("max_tokens") or 1024
    kind = str(cfg.get("kind") or "")
    base = str(cfg.get("url") or "").strip().rstrip("/")
    key = str(cfg.get("key") or "")
    model = str(cfg.get("model") or "")
    system = str(cfg.get("sys") or "")
    # Who is talking, when the family door
    # confirmed them: her own words for who
    # this person is to her travel with the
    # call, the same as they do on Discord.
    visitor = _family_visitor(payload)
    if visitor:
        identity = controller.address_rules_for(
            name=str(visitor.get("name") or ""),
            is_owner_speaker=bool(visitor.get("is_owner")),
            relation=str(visitor.get("relation") or ""),
            username=str(visitor.get("username") or ""),
            place="your website")
        system = (identity + " " + system).strip()
    system = (CREATOR + " " + system).strip()
    if not base:
        return None, "enter the API address"
    if kind == "anthropic":
        url = base + "/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "user-agent": "KookieAi/1.0",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
    elif kind == "custom":
        url = base
        headers = {"content-type": "application/json", "user-agent": "KookieAi/1.0"}
        if key:
            headers["authorization"] = "Bearer " + key
        last = messages[-1] if messages else {}
        body = {
            "messages": ([{"role": "system", "content": system}]
                         if system else []) + messages,
            "message": last.get("content", "") if isinstance(last, dict) else "",
        }
        if model:
            body["model"] = model
    else:
        msgs = ([{"role": "system", "content": system}]
                if system else []) + messages
        headers = {"content-type": "application/json", "user-agent": "KookieAi/1.0"}
        if key:
            headers["authorization"] = "Bearer " + key
        is_openai = str(cfg.get("provider") or "") == "openai"
        cap = 4096 if max_tokens >= 1000 else max_tokens
        # Some models only answer through the
        # Responses API, and chat models only
        # through completions: she asks the one
        # the model's name points to, and tries
        # the other if the first door is shut.
        via_responses = bool(
            is_openai
            and re.search(
                r"(codex|-pro(?:-|$)|deep-research)",
                model, re.IGNORECASE))
        chat_body = {"model": model, "messages": msgs}
        if is_openai and re.search(
                r"^(gpt-([5-9]|\d\d)|o\d)",
                model, re.IGNORECASE):
            chat_body["max_completion_tokens"] = max(cap, 2048)
        else:
            chat_body["max_tokens"] = cap
        responses_body = {
            "model": model,
            "input": [
                {"role": m.get("role"),
                 "content": m.get("content")}
                for m in msgs],
            "max_output_tokens": max(cap, 2048),
        }

        def _ask(endpoint, body):
            request = urllib.request.Request(
                base + endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                        request, timeout=30) as response:
                    raw = response.read().decode(
                        "utf-8", "replace") or "{}"
                    return response.status, raw, None
            except urllib.error.HTTPError as error:
                raw = ""
                try:
                    raw = error.read().decode(
                        "utf-8", "replace")
                except Exception:
                    pass
                return error.code, raw, _error_detail(
                    raw, error)
            except Exception as error:
                return 0, "", (
                    f"could not reach the provider: {error}")

        if via_responses:
            status, raw, error = _ask(
                "/responses", responses_body)
        else:
            status, raw, error = _ask(
                "/chat/completions", chat_body)
            if (status in (400, 404) and is_openai
                    and raw and re.search(
                        r"v1/responses|responses api|"
                        r"not a chat model|only supported in",
                        raw, re.IGNORECASE)):
                status, raw, error = _ask(
                    "/responses", responses_body)
        if error is not None:
            return None, error
        try:
            data = json.loads(raw or "{}")
        except (ValueError, json.JSONDecodeError):
            data = {}
        reply = _censor(_pick_reply(data))
        return (reply or "(empty reply)"), None
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(
                response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            raw = json.loads(
                error.read().decode("utf-8", "replace") or "{}")
            problem = raw.get("error")
            if isinstance(problem, dict):
                detail = str(problem.get("message") or "")
            elif problem:
                detail = str(problem)
            if not detail:
                detail = str(raw.get("message") or "")
        except (ValueError, json.JSONDecodeError):
            pass
        return None, detail or (
            f"the provider returned status {error.code}")
    except Exception as error:
        return None, f"could not reach the provider: {error}"
    reply = _censor(_pick_reply(data))
    return (reply or "(empty reply)"), None


def _models(payload):
    """The models a provider offers, fetched by her.

    The browser cannot always ask a provider for
    its model list either, so she asks instead and
    hands the list back.
    """
    cfg = payload.get("cfg") if isinstance(payload, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    kind = str(cfg.get("kind") or "")
    base = str(cfg.get("url") or "").strip().rstrip("/")
    key = str(cfg.get("key") or "")
    if not base:
        return None, "enter the API address"
    if kind == "anthropic":
        url = base + "/v1/models?limit=1000"
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
    else:
        url = base + "/models"
        headers = {}
        if key:
            headers["authorization"] = "Bearer " + key
    request = urllib.request.Request(
        url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(
                request, timeout=20) as response:
            data = json.loads(
                response.read().decode("utf-8", "replace")
                or "{}")
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            raw = json.loads(
                error.read().decode("utf-8", "replace")
                or "{}")
            problem = raw.get("error")
            if isinstance(problem, dict):
                detail = str(problem.get("message") or "")
            elif problem:
                detail = str(problem)
            if not detail:
                detail = str(raw.get("message") or "")
        except (ValueError, json.JSONDecodeError):
            pass
        return None, detail or (
            f"the provider returned status {error.code}")
    except Exception as error:
        return None, f"could not reach the provider: {error}"
    if not isinstance(data, dict):
        data = {"data": data}
    return data, None


class Handler(BaseHTTPRequestHandler):
    """Every route her phone needs, on one small server."""

    server_version = "KookieWeb/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, _fmt, *_args):
        pass

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self._send(404, "not found", "text/plain")
            return
        self._send(200, body, ctype)

    def _supplied_key(self):
        """The key this request carries, if any.

        A fetch sends it as a header; the voice stream
        cannot, so it may arrive as a query instead.
        """
        supplied = self.headers.get("X-Kookie-Key", "")
        if not supplied:
            query = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query
            )
            supplied = query.get("key", [""])[0]
        return supplied

    def _key_kind(self):
        """Which door this key opens: family, stranger, or none.

        Her family key is hers alone, and so is a
        stranger key the owner set. Without one, her
        website stands open and hands out keys of its
        own: one of those opens the stranger door, and
        anything else is not a key of hers.
        """
        supplied = self._supplied_key()
        if WEB_FAMILY_KEY and supplied == WEB_FAMILY_KEY:
            return "family"
        if supplied and _is_issued(supplied):
            return "stranger"
        return None

    def _unauthorized(self):
        self._send(401, "wrong key", "text/plain")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            # The page itself is public: all it holds
            # is the door, which opens for anybody.
            self._file(os.path.join(WEB_DIR, "index.html"), "text/html")
            return
        if path == "/api":
            # The whole of her API, so a project
            # can find out how to talk to her.
            self._send(
                200, json.dumps(API_DESCRIPTION),
                "application/json",
            )
            return
        if path == "/api/whoami":
            # The door asks which key you have, and the family
            # door asks who you are, before it opens all the
            # way. On an open website anybody may ask, before
            # they hold a key of their own: that is how a
            # visitor learns the door stands open at all.
            kind = self._key_kind()
            if kind is None and not self._supplied_key():
                kind = "stranger"
            if kind is None:
                self._unauthorized()
                return
            info = {"mode": kind}
            if kind == "family":
                # A fetch sends the account as a header; a
                # plain link can only carry it in the address.
                discord_ref = self.headers.get("X-Kookie-Discord", "")
                if not discord_ref:
                    discord_ref = query.get("discord", [""])[0]
                visitor = controller.website_visitor(discord_ref)
                if visitor is None:
                    info["family"] = False
                else:
                    info["family"] = True
                    info["name"] = visitor["name"]
                    info["relation"] = visitor["relation"]
            self._send(200, json.dumps(info), "application/json")
            return
        if path == "/api/memory":
            # A visitor's remembered messages, for the
            # browser that holds their id.
            self._memory(query)
            return
        if path == "/api/webmemory":
            # All web chat memory, for viewing from the frontend.
            self._webmemory()
            return
        kind = self._key_kind()
        if kind is None:
            self._unauthorized()
            return
        if path.startswith("/api/chat/"):
            self._stream_chat(path[len("/api/chat/"):])
            return
        if path == "/api/tts":
            self._tts(query.get("text", [""])[0])
            return
        if path == "/api/status":
            mood = controller.get_mood_state()
            self._send(
                200,
                json.dumps(
                    {
                        "mood": mood.get("mood", ""),
                        "intensity": mood.get("intensity", 0.5),
                        "voice": voice.VOICE_ENABLED,
                    }
                ),
                "application/json",
            )
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/issue-key":
            self._issue_key()
            return
        if parsed.path == "/api/provider":
            self._provider()
            return
        if parsed.path == "/api/remember":
            self._remember()
            return
        if parsed.path == "/api/models":
            self._models()
            return
        if parsed.path == "/api/dev-id":
            self._dev_id()
            return
        if parsed.path == "/api/ask":
            kind = self._key_kind()
            if kind is None:
                self._unauthorized()
                return
            self._ask(kind)
            return
        if parsed.path != "/api/chat":
            self._send(404, "not found", "text/plain")
            return
        kind = self._key_kind()
        if kind is None:
            self._unauthorized()
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
            message = str(data.get("message", "")).strip()
        except (ValueError, json.JSONDecodeError):
            message = ""
        if not message:
            self._send(400, "say something first", "text/plain")
            return
        # The family key says "I might be family"; the Discord
        # account says who. Only a known account counts, so a
        # stranger with the family key still gets the tsundere.
        visitor = None
        if kind == "family":
            visitor = controller.website_visitor(
                self.headers.get("X-Kookie-Discord", ""))
        with CHAT_LOCK:
            job = start_chat(message, visitor)
            job_id = hashlib.md5(
                f"{time.time():.6f}".encode()
            ).hexdigest()[:12]
            with JOBS_LOCK:
                JOBS[job_id] = job
        self._send(202, json.dumps({"id": job_id}), "application/json")

    def _issue_key(self):
        """One of her own keys, for a project of theirs.

        Her website stands open, so anybody who asks is
        handed a key: the key is what the API answers,
        and the door to the website stays open either
        way. The name is theirs to pick, so the keys
        out in the world say whose project is whose.
        """
        name = ""
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
            name = str(data.get("name", ""))
        except (ValueError, json.JSONDecodeError):
            pass
        self._send(
            200, json.dumps({"key": _mint_key(name)}),
            "application/json",
        )

    def _provider(self):
        """Relay one chat turn to the visitor's own provider.

        Open like the key door: the page in front of it is
        what keeps strangers out, and the call only ever
        speaks to the address the visitor themselves typed
        in, with the key they themselves pasted.
        """
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload.get("messages"), list):
            self._send(400, "say something first", "text/plain")
            return
        reply, error = _relay(payload)
        if error is not None:
            self._send(
                502, json.dumps({"error": error}),
                "application/json",
            )
            return
        self._send(
            200, json.dumps({"reply": reply}),
            "application/json",
        )

    def _memory(self, query):
        """The messages a browser has remembered.

        Only the browser that holds the id may ask
        for them: the id is long and random, and
        nobody else knows it.
        """
        visitor = str(query.get("visitor", [""])[0] or "")
        if not _is_visitor(visitor):
            self._send(400, "no such visitor", "text/plain")
            return
        self._send(
            200, json.dumps({
                "history": _memory_of(visitor)
            }),
            "application/json",
        )

    def _webmemory(self):
        """All web chat memory, for viewing from the frontend."""
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
        except (OSError, ValueError):
            stored = []
        self._send(
            200, json.dumps({"memory": stored}),
            "application/json",
        )

    def _remember(self):
        """Keep one more message, or a whole chat.

        The page sends each message as it happens,
        so a crash costs one message at most.
        """
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(
                self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            payload = {}
        visitor = str(payload.get("visitor") or "")
        if not _is_visitor(visitor):
            self._send(400, "no such visitor", "text/plain")
            return
        kept = []
        messages = payload.get("messages")
        if isinstance(messages, list):
            for item in messages:
                if (isinstance(item, dict)
                        and item.get("role") in (
                            "user", "assistant")
                        and isinstance(
                            item.get("content"), str)
                        and item["content"].strip()):
                    kept.append(
                        (item["role"], item["content"]))
        else:
            role = str(payload.get("role") or "")
            content = str(payload.get("content") or "")
            if (role in ("user", "assistant")
                    and content.strip()):
                kept.append((role, content))
        if not kept:
            self._send(
                400, "nothing to remember",
                "text/plain",
            )
            return
        for role, content in kept:
            _remember_message(visitor, role, content)
        self._send(
            200, json.dumps({"remembered": len(kept)}),
            "application/json",
        )

    def _models(self):
        """The models a visitor's provider offers,
        for the picker in the connection form."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(
                self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            payload = {}
        data, error = _models(payload)
        if error is not None:
            self._send(
                502, json.dumps({"error": error}),
                "application/json",
            )
            return
        self._send(
            200, json.dumps(data or {}),
            "application/json",
        )

    def _dev_id(self):
        """Is this Discord account one of her family?

        The developer door opens for the people she
        belongs to: her own person, and anybody
        given a place in her family. The list is
        hers, not the page's, so it grows when
        her family does.
        """
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(
                self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            payload = {}
        ref = str(payload.get("discord") or "").strip()
        if not re.fullmatch(r"\d{17,20}", ref):
            self._send(
                400, "enter a numeric Discord id",
                "text/plain",
            )
            return
        visitor = controller.website_visitor(ref)
        family = bool(
            visitor
            and (visitor.get("is_owner")
                 or visitor.get("relation")))
        self._send(
            200, json.dumps({
                "family": family,
                "name": (
                    visitor.get("name")
                    if family else ""),
                "relation": (
                    visitor.get("relation")
                    if family else ""),
            }),
            "application/json",
        )

    def _ask(self, kind):
        """One question, one answer, waited for.

        The same turn the website's chat starts, but
        the answer comes back whole instead of as a
        stream, so a project does not have to read
        her reply one token at a time.
        """
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
            message = str(data.get("message", "")).strip()
        except (ValueError, json.JSONDecodeError):
            message = ""
        if not message:
            self._send(400, "say something first", "text/plain")
            return
        visitor = None
        if kind == "family":
            visitor = controller.website_visitor(
                self.headers.get("X-Kookie-Discord", ""))
        with CHAT_LOCK:
            job = start_chat(message, visitor)
            job.done.wait()
        if job.error is not None:
            self._send(500, job.error, "text/plain")
            return
        self._send(
            200, json.dumps(job.result or {"reply": job.snapshot()}),
            "application/json",
        )

    def _stream_chat(self, job_id):
        """Her reply as it is written, one event per token."""
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            self._send(404, "no such chat", "text/plain")
            return
        sent = 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def out(event, payload):
            self.wfile.write(
                f"event: {event}\ndata: "
                f"{json.dumps(payload)}\n\n".encode("utf-8")
            )
            self.wfile.flush()

        while not job.done.is_set():
            time.sleep(0.25)
            partial = job.snapshot()
            if len(partial) > sent:
                sent = len(partial)
                out("token", {"text": partial})
        partial = job.snapshot()
        if len(partial) > sent:
            out("token", {"text": partial})
        if job.error is not None:
            out("error", {"error": job.error})
        else:
            out("done", job.result or {"reply": partial})
        with JOBS_LOCK:
            JOBS.pop(job_id, None)

    def _tts(self, text):
        """Her voice for a sentence, kept so the phone
        does not pay for the same words twice."""
        # Emoji and stage directions are hers to read,
        # not hers to say: Piper would choke on them.
        text = voice.sanitize_for_speech(str(text or ""))[:300]
        if not text:
            self._send(400, "nothing to say", "text/plain")
            return
        key = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        wav_path = os.path.join(TTS_CACHE, f"{key}.wav")
        if not os.path.exists(wav_path):
            if not _synthesize(text, wav_path):
                self._send(500, "voice not available", "text/plain")
                return
        self._file(wav_path, "audio/wav")


_server = None
_server_thread = None
_server_lock = threading.Lock()
_tunnel_process = None
_tunnel_url = None
_tunnel_problem = None


def web_addresses():
    """The addresses her website answers on."""
    addresses = []
    tunneled = bool(TUNNEL and _tunnel_url)
    # If custom domain is set, use it as the primary address
    if CUSTOM_DOMAIN:
        domain = CUSTOM_DOMAIN
        if not domain.startswith(("http://", "https://")):
            domain = "https://" + domain
        addresses = [domain]
    if not (tunneled and not BIND):
        ips = _lan_ips()
        addresses = (
            [f"http://{ip}:{PORT}" for ip in ips]
            or [f"http://localhost:{PORT}"]
        )
    if tunneled:
        addresses.append(_tunnel_url)
    return addresses


def _public_ip():
    """The address the internet sees this box on, if any."""
    try:
        with urllib.request.urlopen(
                "https://api.ipify.org", timeout=5) as response:
            ip = response.read().decode("utf-8", "replace").strip()
        return ip if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ip) else ""
    except Exception:
        return ""


def _start_tunnel():
    """Open a door through the router, when there is one.

    cloudflared's quick tunnel needs no account and gives
    her website an address on the open internet, so a phone
    reaches her from anywhere without a forwarded port. The
    address arrives on its own once the door is open; a
    sentence is kept when it cannot be opened at all.
    """
    global _tunnel_process, _tunnel_problem
    import shutil
    import subprocess

    binary = shutil.which("cloudflared")
    if not binary:
        _tunnel_problem = (
            "cloudflared is not installed, so there is no "
            "tunnel address - install cloudflared to be "
            "reachable from anywhere"
        )
        return
    try:
        _tunnel_process = subprocess.Popen(
            [binary, "tunnel", "--url", f"http://localhost:{PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as error:
        _tunnel_problem = f"cloudflared would not start: {error}"
        return
    _tunnel_problem = None

    def _watch():
        global _tunnel_url
        for line in iter(_tunnel_process.stdout.readline, b""):
            found = re.search(
                r"https://[a-z0-9-]+\.trycloudflare\.com",
                line.decode("utf-8", "replace"))
            if found:
                _tunnel_url = found.group(0)
                break

    threading.Thread(
        target=_watch, daemon=True, name="kookie-tunnel").start()


def _await_tunnel(seconds=20):
    """Wait briefly for the tunnel's address to exist."""
    deadline = time.time() + seconds
    while not _tunnel_url and time.time() < deadline:
        time.sleep(0.5)
    return _tunnel_url


def web_tunnel_url():
    """The address her website answers on, from anywhere."""
    return _tunnel_url


def web_tunnel_problem():
    """Why there is no tunnel address, when there is not one."""
    return _tunnel_problem


def _open_browser():
    """Open her website here, so turning her on is one command.

    On its own thread, because opening a browser can wait on
    a display that is not there, and that must never hold up
    the command that started her.
    """
    if os.environ.get("KOOKIE_WEB_NO_OPEN"):
        return

    def _open():
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{PORT}")
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def start_web_server():
    """Serve her website in this process, on its own thread.

    `./Kookie --web` and the /web command both end up here: the
    model and her memory are whatever this process already loaded,
    so the website never needs a second copy of her.
    """
    global _server, _server_thread
    with _server_lock:
        if _server is not None:
            return False
        host = BIND or "0.0.0.0"
        if TUNNEL:
            _start_tunnel()
            _await_tunnel()
            if _tunnel_url and not BIND:
                # The tunnel is the only door. The box does
                # not answer the network at all, so the
                # address it hides never appears anywhere.
                host = "127.0.0.1"
        try:
            server = ThreadingHTTPServer((host, PORT), Handler)
        except OSError as error:
            # Someone else owns the port. A sentence, not a
            # traceback: the address is the whole point.
            raise RuntimeError(
                f"port {PORT} is taken ({error}). "
                f"KOOKIE_WEB_PORT picks another."
            ) from error
        thread = threading.Thread(
            target=server.serve_forever,
            name="kookie-web",
            daemon=True,
        )
        thread.start()
        _server = server
        _server_thread = thread
    _open_browser()
    return True


def stop_web_server():
    """Take her website down again."""
    global _server, _server_thread, _tunnel_process, _tunnel_url
    with _server_lock:
        server = _server
        _server = None
        _server_thread = None
    tunnel = _tunnel_process
    _tunnel_process = None
    _tunnel_url = None
    if tunnel is not None:
        tunnel.terminate()
    if server is None:
        return False
    server.shutdown()
    server.server_close()
    return True


def web_server_running():
    """Whether her website is being served by this process."""
    return _server is not None


def _print_outside_addresses():
    """How to reach her without being on the same network."""
    if TUNNEL:
        if not _tunnel_url:
            if _tunnel_problem:
                print(f"  tunnel: {_tunnel_problem}")
            else:
                print("  tunnel: cloudflared has not given an address yet")
        return
    if not SHOW_PUBLIC_IP:
        return
    public = _public_ip()
    if public and public not in _lan_ips():
        print(
            f"  http://{public}:{PORT}  from anywhere, once the "
            f"router forwards port {PORT} to this box"
        )
    elif public:
        print(f"  http://{public}:{PORT}  from anywhere")


def main():
    # Everything she knows, loaded before the first
    # "hi" arrives, the same way her window does it.
    # The phone carries her avatar, so the local one
    # is not driven from here.
    controller.start_runtime(connect_avatar=False, animate_parameters=False)
    start_web_server()
    print("Kookie is on the web:")
    for address in web_addresses():
        print(f"  {address}")
    _print_outside_addresses()
    print("Open that on your phone; Ctrl+C stops her.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_web_server()


if __name__ == "__main__":
    main()
