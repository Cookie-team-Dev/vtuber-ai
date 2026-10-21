"""Kookie on Discord.

She is the same person here as in the terminal: the same model, the same memory,
the same mood. Only the room changes. This file is the room - it hands what
people say to controller.ask_ai and puts her answer back into the channel.

Two things deliberately are not shared with the terminal version. The Live2D
face and the spoken voice belong to the window she is watching, not to a chat
channel, so deliver_reply is skipped and only her words are sent. And her tool
loop - the one that can open a browser, run commands and change files on this
machine - stays off unless it is switched on explicitly, because on Discord the
person sending the message is whoever happens to be in the channel.

Her token is read from the environment or from ~/.config/kookie/discord_token,
never from this file, so it never ends up in a file that gets shared or committed.
"""

print("[BOT] Starting discord_bot.py")
import sys
if __name__ == "__main__" and "discord_bot" not in sys.modules:
    sys.modules["discord_bot"] = sys.modules["__main__"]
import asyncio
import collections
import json
import logging
import os
import re
import signal
import stat
import socket
import ipaddress
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.parse

# Patch discord.ext.voice_recv.reader.AudioReader.stop so a self-stop - which
# is what the voice_recv router does when its reader finishes - does not get
# mistaken for the client dying. The original behaviour is kept; only the
# noisy traceback that used to print on every stop is gone.
import fastai
import discord.ext.voice_recv.reader as reader_mod

_original_stop = reader_mod.AudioReader.stop

def _patched_stop(self):
    return _original_stop(self)

reader_mod.AudioReader.stop = _patched_stop
print("[VOICE] Patched AudioReader.stop")

# Voice channel state: one VoiceClient per guild, plus a flag for whether her
# replies should be spoken aloud in that guild's voice channel.
_VOICE_CLIENTS = {}

class _VoiceEnabledSet(set):
    def add(self, value):
        super().add(value)
        print(f"[_VOICE_ENABLED] ADD {value} (now: {list(self)})")
    def discard(self, value):
        super().discard(value)
        print(f"[_VOICE_ENABLED] DISCARD {value} (now: {list(self)})")
    def remove(self, value):
        super().remove(value)
        print(f"[_VOICE_ENABLED] REMOVE {value} (now: {list(self)})")
    def clear(self):
        print(f"[_VOICE_ENABLED] CLEAR (was: {list(self)})")
        super().clear()

_VOICE_ENABLED = _VoiceEnabledSet()

def get_voice_enabled():
    """Get the current _VOICE_ENABLED set (initialized at module scope)."""
    return _VOICE_ENABLED

def set_voice_enabled(new_set):
    """Replace the _VOICE_ENABLED set (for reinitialization)."""
    global _VOICE_ENABLED
    _VOICE_ENABLED = new_set

import aiohttp
import discord
from discord import app_commands
from discord.abc import PrivateChannel
from discord.ext import voice_recv

import controller
import voice
from discord_commands import (
    maybe_switch,
    register_slash_commands,
    voice_listen,
    voice_speak,
)
from discord_commands import _VOICE_LISTENERS, _VOICE_SINKS

DISCORD_TOKEN_ENV = "KOOKIE_DISCORD_TOKEN"
TOKEN_FILE = os.path.expanduser("~/.config/kookie/discord_token")

# Some networks answer for discord.com and gateway.discord.gg with a sinkhole
# address instead of Discord's real one. Everything else on the internet
# resolves fine, so it is not a broken resolver - it is a targeted block, and it
# left her unable to log in at all even though the addresses Cloudflare hands out
# answer in a fifth of a second.
#
# The fix here asks Cloudflare over HTTPS for those names only, and hands the
# real address to the library. HTTPS to Cloudflare itself is not blocked, and
# doing it in this file rather than in /etc/resolv.conf keeps it to Kookie: the
# system resolver file lives on a read-only mount and there is no sudo on this
# box. Everything else, and any name that fails to resolve, falls through to the
# normal resolver untouched. Set KOOKIE_DISCORD_DOH=0 to turn it off once the
# network stops blocking Discord.
_DOH_HOSTS = frozenset(
    {
        "discord.com",
        "gateway.discord.gg",
        "discordapp.com",
        "cdn.discordapp.com",
        "media.discordapp.net",
    }
)
_DOH_ENDPOINT = "https://cloudflare-dns.com/dns-query"
_DOH_CACHE = {}
_SYSTEM_GETADDRINFO = socket.getaddrinfo


def _resolve_over_https(host):
    """Ask Cloudflare for a host's real address, or None if it cannot be reached."""
    cached = _DOH_CACHE.get(host)
    if cached is not None:
        return cached
    request = urllib.request.Request(
        f"{_DOH_ENDPOINT}?name={host}&type=A",
        headers={"accept": "application/dns-json", "user-agent": "kookie-discord"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            answers = json.loads(response.read().decode("utf-8")).get("Answer") or []
    except Exception as error:  # network, TLS, bad JSON - all mean "use the system"
        print(f"Discord bot: DNS-over-HTTPS lookup for {host} failed ({error})")
        _DOH_CACHE[host] = []
        return None
    addresses = [
        str(answer.get("data"))
        for answer in answers
        if isinstance(answer, dict) and answer.get("type") == 1
    ]
    _DOH_CACHE[host] = addresses
    if addresses:
        print(f"Discord bot: {host} -> {', '.join(addresses[:2])} (via DNS-over-HTTPS)")
    return addresses or None


def _install_doh_resolver():
    """Send Discord's own names around the resolver that blocks them."""
    if os.environ.get("KOOKIE_DISCORD_DOH", "1").strip() in ("0", "off", "false", "no"):
        return

    def getaddrinfo(host, port, *args, **kwargs):
        name = str(host or "").lower().rstrip(".")
        if name in _DOH_HOSTS:
            for address in _resolve_over_https(name) or ():
                try:
                    return _SYSTEM_GETADDRINFO(address, port, *args, **kwargs)
                except OSError:
                    continue  # one address is down; try the next one DoH gave us
        return _SYSTEM_GETADDRINFO(host, port, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo


_install_doh_resolver()

# Discord refuses messages over 2000 characters, and replies are split below a
# limit that leaves room for the rest of a sentence rather than cutting one.
MESSAGE_LIMIT = 1900
NO_PINGS = discord.AllowedMentions.none()

# How long she usually takes, and how long to wait before telling somebody she
# is still working. She is slow because the model is small and the machine is
# shared; silence with no typing indicator reads as broken.
TYPING_REFRESH_SECONDS = 8.0
NUDGE_AFTER_SECONDS = 60.0

# One turn at a time. She has one mood, one memory file and one conversation
# lock, and two people talking at her at once would interleave her answers.
QUEUE = None


def _flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _id_list(name):
    """Configured Discord ids, as a set of ints, with the empty set meaning all."""
    raw = os.environ.get(name, "")
    found = set()
    for piece in raw.replace(";", ",").split(","):
        piece = piece.strip()
        if piece:
            try:
                found.add(int(piece))
            except ValueError:
                print(f"Discord bot: ignoring {name} entry {piece!r}, which is not an id")
    return found


ALLOWED_CHANNELS = _id_list("KOOKIE_DISCORD_ALLOWED_CHANNELS")
ALLOWED_USERS = _id_list("KOOKIE_DISCORD_ALLOWED_USERS")
# On by default: a DM is two people talking, and it is how she is actually used
# day to day. Worth knowing that anyone who can reach the bot can DM it, so
# KOOKIE_DISCORD_ALLOWED_USERS is the limit if the bot is in a public server.
ALLOW_DMS = _flag("KOOKIE_DISCORD_ALLOW_DMS", "1")
# Whether a message has to be addressed to her before she answers it. Off by
# default now, and the change is not just the flag: an unaddressed message is no
# longer discarded either way. Every message the bot can see is read into the
# room transcript (controller.note_room_message), so she knows what was said to
# whom, follows threads she was not invited to, and can come back to it later on
# her own. What this flag decides is only whether she replies straight away to a
# message nobody aimed at her, which is a question about timing and noise rather
# than about whether she can see.
REQUIRE_MENTION = _flag("KOOKIE_DISCORD_REQUIRE_MENTION", "0")
# Hard gate on reading the room at all. Turning this off puts her back to seeing
# only what she was pinged about, which is the old behaviour. It exists as a
# kill switch for a server where being present in every channel is not wanted.
READ_EVERYTHING = _flag("KOOKIE_DISCORD_READ_ALL", "1")
ALLOW_TOOLS = _flag("KOOKIE_DISCORD_TOOLS")
# A tool she cannot reach should not be advertised to her either way: the list is
# ~125 prompt tokens she pays for on every single message. With tools off here
# she stops being tempted to call something that would be refused.
controller.TOOLS_ENABLED = ALLOW_TOOLS
NUDGE = _flag("KOOKIE_DISCORD_NUDGE", "0")
# She reads her replies out loud in the channel, in Discord's own voice, and the
# whole message is spoken rather than a summary of it. Off by default because a
# text-only reply from a character who is otherwise a voice is the thing that
# makes her feel like a bot. Needs the Send Message TTS permission, which she
# already has.
TTS = _flag("KOOKIE_DISCORD_TTS", "0")
# Whether she is allowed to start a conversation nobody asked for. On by default:
# she decides on her own that there is something worth saying, and this is where
# that line goes.
PROACTIVE = _flag("KOOKIE_DISCORD_PROACTIVE", "1")
# Where an unprompted line goes when she has never been spoken to. Empty means
# she stays quiet until a channel has been used, rather than picking one.
PROACTIVE_CHANNEL = os.environ.get("KOOKIE_DISCORD_PROACTIVE_CHANNEL", "").strip()

# How many messages may be waiting for her. Beyond this the oldest are dropped:
# answering a backlog an hour late is worse than admitting to it.
try:
    MAX_PENDING = max(1, int(os.environ.get("KOOKIE_DISCORD_MAX_PENDING", "8")))
except ValueError:
    MAX_PENDING = 8

_client: discord.Client = None
_worker: asyncio.Task = None
_tree = None
# The last channel somebody actually spoke to her in. It is where an unprompted
# line belongs: the conversation she was part of, rather than a channel chosen
# out of a list of every one the bot can see.
_last_channel = None
# Ids of the messages she has sent, so a reply to one of them is recognised as
# addressed to her. discord.py can usually resolve a reply reference to the real
# message, but not when the original has been deleted or has fallen out of the
# cache, and a reply she does not notice is the single most obvious way for her
# to look like she is ignoring a conversation she is part of. Bounded because
# this grows for as long as she is online.
_HER_MESSAGE_IDS = collections.deque(maxlen=400)
# What she has said in each channel lately, reduced to comparable words, so a
# whole sentence going over itself is caught before it is posted twice. Keyed by
# channel, bounded, because it grows for as long as she is online.
_HER_LAST_WORDS = {}
# Words that can come between somebody's greeting and her name and still mean
# they are talking to her: "hey kookie", "ok kookie so", "yo kookie". Kept short
# on purpose - the looser this gets, the more she answers sentences that were
# never addressed to her.
GREETING_OPENERS = (
    r"(?:hey|hi|hello|yo|ok|okay|so|um+|uh+|well|psst|look|excuse me|"
    r"mommy|mom)\b[,:]?"
)

# Anime style terms used to detect anime image prompts.
# Hoisted to module level so the draw handler can reference it
# without rebuilding the tuple on every request.
ANIME_STYLE_TERMS = (
    "anime", "manga", "waifu", "husbando", "kawaii", "senpai",
    "chan", "sama", "sensei", "shonen", "shoujo", "isekai",
    "oppai", "baka", "desu", "nya", "rawr", "monke",
    "naruto", "one piece", "dragon ball", "attack on titan",
    "demon slayer", "jujutsu kaisen", "my hero academia",
    "re zero", "re:zero", "sword art online", "sao",
    "fate", "fate/stay", "fate grand order", "fgo",
    "pokemon", "spy x family", "chainsaw man",
    "bleach", "gintama", "hunter x hunter", "hxh",
    "fullmetal alchemist", "fma", "code geass",
    "steins;gate", "cowboy bebop", "neon genesis evangelion",
    "evangelion", "one punch man", "opm", "mob psycho 100",
    "tokyo ghoul", "parasyte", "death note",
    "soul eater", "black butler", "kuroshitsuji",
    "noragami", "blue exorcist", "ao no exorcist",
    "magi", "akame ga kill", "kill la kill",
    "gurren lagann", "durarara",
    "baccano", "monogatari", "kara no kyoukai",
    "overlord", "that time i got reincarnated as a slime",
    "tensei shitara slime datta ken", "slime",
    "mushoku tensei", "jobless reincarnation",
    "the rising of the shield hero", "tate no Yuusha",
    "konosuba", "kono subarashii",
    "reincarnated as a sword",
    "tensei shitara ken deshita", "blade",
    "the wrong way to use healing magic",
    "healing magic no wrong use",
    "solo leveling", "omniscient reader",
    "orv", "tbate", "trash of the count's family",
    "tcf", "lcf", "lout of count's family",
    "survival story in a fantasy world",
    "apothecary diary", "kusuriya no hitorigoto",
    "honey lemon soda", "honey and lemon",
    "the apothecary diaries"
)

async def _voice_join(channel):
    """Join the voice channel the user is in, for the guild it belongs to."""
    guild_id = getattr(getattr(channel, "guild", None), "id", None)
    if guild_id is None:
        return None, "that is not a guild channel"
    if guild_id in _VOICE_CLIENTS:
        client = _VOICE_CLIENTS[guild_id]
        if client.is_connected():
            return client, None
        try:
            await client.disconnect(force=True)
        except Exception:
            pass
        del _VOICE_CLIENTS[guild_id]
        _VOICE_LISTENERS.pop(guild_id, None)
        _VOICE_SINKS.pop(guild_id, None)
    try:
        client = await channel.connect(cls=voice_recv.VoiceRecvClient)
        _VOICE_CLIENTS[guild_id] = client
        return client, None
    except Exception as e:
        return None, str(e)


async def _voice_leave(guild_id):
    """Disconnect from a guild's voice channel, if connected."""
    client = _VOICE_CLIENTS.get(guild_id)
    if client is None:
        return
    print(f"[_VOICE LEAVE] guild_id={guild_id} _VOICE_ENABLED={list(get_voice_enabled())}")
    try:
        await client.disconnect(force=True)
    except Exception:
        pass
    _VOICE_CLIENTS.pop(guild_id, None)
    _VOICE_LISTENERS.pop(guild_id, None)
    _VOICE_SINKS.pop(guild_id, None)


async def _voice_speak(channel, text):
    """Synthesize text with Piper and play it in the channel's voice chat."""
    if not text or not text.strip():
        return None
    guild_id = getattr(getattr(channel, "guild", None), "id", None)
    if guild_id is None or guild_id not in get_voice_enabled():
        return None
    client = _VOICE_CLIENTS.get(guild_id)
    if client is None or not client.is_connected():
        return None
    spoken = voice.sanitize_for_speech(text)
    if not spoken:
        return None
    wav_path = None
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        await asyncio.to_thread(
            voice._synthesize_with_piper, spoken, wav_path, 1.0
        )
        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            return None
        audio = discord.FFmpegPCMAudio(wav_path)
        client.play(audio)
        for _ in range(240):
            if not client.is_playing():
                break
            await asyncio.sleep(0.1)
        return True
    except Exception as e:
        print("Discord bot: voice speak error:", e)
        return None
    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def her_queue():
    """Her message queue, built inside the loop that is going to use it.

    Made at import time it belongs to whichever loop happened to exist then, and
    every other loop afterwards refuses it outright.
    """
    global QUEUE
    if QUEUE is None:
        QUEUE = asyncio.Queue()
    return QUEUE


def read_token():
    """The bot token, from the environment first and then from its own file."""
    token = os.environ.get(DISCORD_TOKEN_ENV, "").strip()
    if token:
        return token
    try:
        st = os.stat(TOKEN_FILE)
        mode = stat.S_IMODE(st.st_mode)
        if mode & 0o077:
            print(f"Discord bot: {TOKEN_FILE} is readable by group/world ({oct(mode)}); chmod 600")
            os.chmod(TOKEN_FILE, 0o600)
        with open(TOKEN_FILE, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def split_reply(text, limit=MESSAGE_LIMIT):
    """Break a long answer into pieces Discord will actually accept."""
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces, remaining = [], text
    while remaining:
        if len(remaining) <= limit:
            pieces.append(remaining)
            break
        # Prefer a paragraph, then a sentence, then a space to break on.
        window = remaining[:limit]
        for marker in ("\n\n", "\n", ". ", " "):
            cut = window.rfind(marker)
            if cut > limit // 2:
                pieces.append(window[: cut + len(marker)].strip())
                remaining = remaining[cut + len(marker):].strip()
                break
        else:
            pieces.append(window.strip())
            remaining = remaining[limit:].strip()
    return [piece for piece in pieces if piece]


def reply_text_for(result, speaker_name=None):
    """Her words, cleaned the way the terminal cleans them before showing them.

    The speaker has to be named here as well as at generation time. This runs on
    the event loop while the model ran on a worker thread, and the name she calls
    someone is held per thread, so leaving it out cleaned a stranger's reply with
    the default and put "Mommy" back into it.
    """
    try:
        reply = controller.clean_reply(result.get("reply", ""), address_name=speaker_name)
        if not result.get("singing"):
            reply = controller.tidy_reply_casing(controller.condense_reply(reply))
    except Exception as error:
        print("Discord bot: reply cleanup failed:", error)
        reply = str(result.get("reply", "") or "")
    return reply.strip() or "I thought about it and came up with nothing to say."


def ask_with_tools(text, on_reply_text=None, context_note="", speaker_name=None, speaker_is_owner=True, speaker_id=None):
    """One turn, including her tool loop when it has been switched on.

    The loop is the same one the terminal runs, including the reminder to check
    what a tool actually returned. It is off by default because a tool here can
    change files on this machine, and on Discord the sender is a channel member.

    on_reply_text is only wired into the first turn: once a tool has run she is
    answering the tool result, not the message, and streaming that into the same
    message would overwrite what she already wrote.

    context_note stays with every turn of the loop. It carries the other lines
    somebody typed in the same burst, and a tool round trip is exactly when she
    loses them: she comes back from reading a file with no idea that "no, tell
    daddy" was still waiting to be answered.
    """
    result = controller.ask_ai(
        text,
        on_reply_text=on_reply_text,
        context_note=context_note,
        speaker_name=speaker_name,
        speaker_is_owner=speaker_is_owner,
        speaker_id=speaker_id,
    )
    iterations = 0
    used_tools = set()
    while result.get("tool_call") and iterations < controller.MAX_TOOL_ITERATIONS:
        call = result.get("tool_call") or {}
        name = call.get("tool", "")
        # calculate is arithmetic and nothing else, so it runs even with the rest
        # of the tools switched off. Anything else needs KOOKIE_DISCORD_TOOLS: a
        # chat message should not be able to edit files or drive the screen.
        if not ALLOW_TOOLS and not controller.tool_is_always_allowed(name):
            return result
        value = controller._tool_arg_text(
            call.get("arg_value") or call.get("argvalue") or ""
        )
        # Asking for the same thing again is a loop, not curiosity.
        # The same guard the autonomous loop runs: without it a
        # video question watches the same clip once per turn.
        tool_signature = (name, value.strip().lower())
        if tool_signature in used_tools:
            print(f"Discord bot: [skipping repeated tool {name}]")
            # She already holds the answer: send her back in
            # with the result and a plain order to answer
            # from it, rather than returning the turn that
            # only announced the call.
            repeat_hint = (
                "You already ran this tool and have its result. "
                "Answer the message from that result now, "
                "without calling any tool again.\n\n"
            )
            result = controller.ask_ai(
                text,
                context_note=f"{context_note}\n\n{repeat_hint}" if context_note else repeat_hint,
                speaker_name=speaker_name,
                speaker_is_owner=speaker_is_owner,
                speaker_id=speaker_id,
            )
            result = dict(result)
            result["tool_call"] = None
            break
        used_tools.add(tool_signature)
        iterations += 1
        print(f"Discord bot: [tool {name}({value[:80]})]")
        outcome = controller.execute_tool_call(name, value)
        hint = (
            "Before responding to this tool result, verify it briefly: does it "
            "actually answer what you needed? If not, say so honestly and try a "
            f"different approach. Never pretend a tool result is correct if it "
            f"looks wrong or incomplete.\n\nTool result from {name}"
            f"({value[:80]}...):\n{outcome}\n\n"
        )
        result = controller.ask_ai(
            text,
            context_note=f"{context_note}\n\n{hint}" if context_note else hint,
            speaker_name=speaker_name,
            speaker_is_owner=speaker_is_owner,
            speaker_id=speaker_id,
        )
    return result


class TypingReply:
    """One message in the channel that she writes into as the answer arrives.

    Discord allows about five edits every five seconds per channel, so the
    updates are paced well under that. Without the pacing a fast token stream
    turns into a burst of rate-limit errors and the reply never appears at all,
    which is the one outcome this class exists to prevent.
    """

    EDIT_INTERVAL = 2.0
    PLACEHOLDER = "*thinking...*"

    def __init__(self, channel, speaker_name=None):
        self.channel = channel
        # Same reason as reply_text_for: this cleans on the event loop, and the
        # name she calls someone is per thread.
        self.speaker_name = speaker_name
        self.message = None
        self.shown = False
        self.latest = ""
        self.edited = ""
        self.loop = None
        self.last_edit = 0.0
        self.dropped = False

    @classmethod
    async def start(cls, channel, speaker_name=None):
        """Put the placeholder up, so a ping never looks unanswered."""
        self = cls(channel, speaker_name)
        self.loop = asyncio.get_running_loop()
        try:
            self.message = await channel.send(cls.PLACEHOLDER, allowed_mentions=NO_PINGS)
            self.shown = True
        except discord.HTTPException as error:
            # No permission, or the channel is gone: fall back to a plain send
            # later rather than failing the whole turn over a cosmetic extra.
            print("Discord bot: could not post the typing placeholder:", error)
            self.message = None
        return self

    def feed(self, partial):
        """Called from the model thread as each token lands."""
        if not self.message or not partial:
            return
        # Edits have to be awaited on the loop, but this runs on a worker, so the
        # text is parked and the event loop does the sending.
        self.latest = partial
        asyncio.run_coroutine_threadsafe(self._maybe_edit(), self.loop)

    async def _maybe_edit(self):
        if self.dropped or self.message is None:
            return
        now = time.monotonic()
        if now - self.last_edit < self.EDIT_INTERVAL:
            return
        text = self._display(self.latest)
        if len(text) < 2 or text == self.edited:
            return
        await self._edit(f"{text}...")

    def _display(self, raw):
        """The partial reply, cleaned the same way the finished one will be.

        Without this the words visibly change under the reader: clean_reply
        swaps jargon for plainer wording, so a raw stream shows "animation
        settings" and then snaps to "movement settings" in the final edit.
        """
        text = controller.clean_reply((raw or "").strip(), address_name=self.speaker_name)
        return text[:MESSAGE_LIMIT - 8]

    async def _edit(self, content):
        self.last_edit = time.monotonic()
        try:
            await self.message.edit(content=content)
            self.edited = content
        except discord.HTTPException as e:
            if getattr(e, "status", None) == 404:
                self.dropped = True
                self.message = None

    async def drain(self):
        """Make sure the last thing she wrote is actually on screen."""
        if self.dropped or self.message is None:
            return
        text = self._display(self.latest)
        if text and f"{text}..." != self.edited:
            await self._edit(f"{text}...")

    async def finish(self, final):
        if self.message is None:
            try:
                await self.channel.send(final, allowed_mentions=NO_PINGS)
            except discord.HTTPException:
                pass
            return
        await self._edit(final)
        self.message = None


async def think_and_reply(channel, label, text, user_id=None, burst_note=""):
    """Wait for her, looking busy, then put the answer in the channel."""
    started = time.monotonic()
    # Who is speaking, and what they have said to her before, goes in with the
    # question. This is the whole difference between her knowing a person and
    # knowing "a person": the words are all in her notes either way, but without
    # it she is handed each message as though it were the first one she has ever
    # been given by anybody. The newest line is skipped, because this very
    # message is already in front of her and repeating it back reads as though
    # she had been told it twice.
    context_note = controller.people_context(user_id, skip=1, message=text)
    # The lines of this same burst that were skipped below, in front of the
    # context note and not instead of it, so a person who typed three things in a
    # row gets one answer to all three.
    if burst_note:
        context_note = f"{burst_note}\n\n{context_note}" if context_note else burst_note

    async def keep_typing():
        while True:
            try:
                await channel.typing()
            except Exception:
                return
            await asyncio.sleep(TYPING_REFRESH_SECONDS)

    async def nudge():
        # A turn that is slow is not a turn that failed, and a channel full of
        # silence with a typing indicator that never resolves looks like a bot
        # that has stopped answering. One short line, once, then nothing.
        await asyncio.sleep(NUDGE_AFTER_SECONDS)
        try:
            await channel.send("Still thinking - my brain is small and this machine is busy.", allowed_mentions=NO_PINGS)
        except discord.HTTPException:
            pass

    typing = asyncio.ensure_future(keep_typing())
    waiting = asyncio.ensure_future(nudge()) if NUDGE else None

    # Show her writing. A turn takes a minute on this machine, and a channel that
    # simply goes quiet for a minute is indistinguishable from a bot that is
    # ignoring you - so the answer appears in the channel as it is written, in one
    # message that is edited over and later.
    #
    # ...unless she is going to speak it. Discord only ever reads a message aloud
    # if the message was sent that way, and an edit cannot turn into speech after
    # the fact, so a streamed message could only ever end up mute. Talking and
    # watching her type are the same slot in Discord, and voice is the one worth
    # having, so the typing indicator above carries the wait instead.
    streamer = None
    if not TTS:
        streamer = await TypingReply.start(channel, label)
    try:
        # ask_ai is a blocking call to a local model that can take a minute or
        # more, so it runs on a thread. On the event loop it would freeze every
        # other channel event - including the message that cancels it.
        feed = streamer.feed if streamer is not None else None
        result = await asyncio.to_thread(
            _one_turn, text, feed, context_note, label,
            controller.is_owner(user_id), user_id,
        )
    finally:
        typing.cancel()
        if waiting is not None:
            waiting.cancel()
        if streamer is not None:
            await streamer.drain()

    took = time.monotonic() - started
    print(f"Discord bot: {label} answered in {took:.1f}s")
    full_reply = reply_text_for(result, label)
    if user_id:
        controller.note_discord_person(
            user_id, label, text=full_reply, said_by_user=False
        )
    if full_reply:
        controller.note_room_message(
            getattr(channel, "name", None) or str(getattr(channel, "id", "")),
            _client.user.display_name if _client else "Kookie",
            full_reply,
            is_kookie=True,
            channel_id=getattr(channel, "id", None),
        )
    if streamer is not None and streamer.shown:
        if len(full_reply) > MESSAGE_LIMIT:
            full_reply = full_reply[:MESSAGE_LIMIT].rsplit(" ", 1)[0] + "..."
        await streamer.finish(full_reply)
        # Also trigger voice chat reader for own reply (when in voice channel)
        if getattr(channel, "guild", None):
            guild_id = channel.guild.id
            voice_enabled = get_voice_enabled()
            import discord_bot as self_mod
            print(f"[CHAT READER] streamer path: guild_id={guild_id} in _VOICE_ENABLED={guild_id in voice_enabled} _VOICE_ENABLED={list(voice_enabled)} module_id={id(self_mod)}")
            if guild_id in voice_enabled:
                try:
                    from discord_commands import chat_reader_speak_message
                    bot_user_id = _client.user.id if _client and _client.user else None
                    print(f"[CHAT READER] streamer path: TRIGGERING chat_reader_speak_message for bot reply")
                    asyncio.create_task(chat_reader_speak_message(guild_id, full_reply, _client.user.display_name if _client and _client.user else "Kookie", author_id=bot_user_id, bot_user_id=bot_user_id, text_channel_id=None))
                except Exception as e:
                    print(f"[CHAT READER] Error triggering for own message: {e}")
        return
    if TTS:
        if len(full_reply) > MESSAGE_LIMIT:
            full_reply = full_reply[:MESSAGE_LIMIT].rsplit(" ", 1)[0] + "..."
        await _speak_into(channel, full_reply, speak=True)
    else:
        if len(full_reply) > MESSAGE_LIMIT:
            full_reply = full_reply[:MESSAGE_LIMIT].rsplit(" ", 1)[0] + "..."
        await _speak_into(channel, full_reply, speak=True)


def _one_turn(text, on_reply_text=None, context_note="", speaker_name=None, speaker_is_owner=True, speaker_id=None):
    """The blocking half of a turn, including her conversation lock."""
    controller.schedule_idle_check()
    if not controller.conversation_lock.acquire(timeout=900):
        raise TimeoutError("Kookie was busy for a very long time")
    try:
        return ask_with_tools(
            text,
            on_reply_text=on_reply_text,
            context_note=context_note,
            speaker_name=speaker_name,
            speaker_is_owner=speaker_is_owner,
            speaker_id=speaker_id,
        )
    finally:
        controller.conversation_lock.release()
        controller.schedule_idle_check()


async def superseded_by_a_newer_one(queue, label, user_id=None):
    """True when this person has already said something newer that is still waiting.

    Someone who types three lines while she is thinking wants one answer to all
    three, not three answers to questions they have moved past. The older message
    is skipped in favour of the newer one, which is still in the queue and still
    keeps its place - and so does everyone else's.

    Matched on the account id rather than the display name, because two people
    can share a name and dropping one of their messages because a stranger wore
    the same one loses a question she was never going to answer anyway.

    The queue is drained and rebuilt to look at it, because asyncio.Queue offers
    no way to read it without emptying it. It only ever holds MAX_PENDING items,
    so the cost is nothing, and each item taken out is completed and each one put
    back is counted again, which leaves the pending count exactly as it was.
    """
    items = []
    while not queue.empty():
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
        queue.task_done()
    if user_id is not None:
        newer = any(len(item) > 3 and item[3] == user_id for item in items)
    else:
        newer = any(item[1] == label for item in items)
    for item in items:
        queue.put_nowait(item)
    return newer


# What somebody said in one burst, kept until their newest message is answered.
#
# Skipping a message because they said something better afterwards is right, and
# throwing away what they said is not. "what did you fix" followed by "no, tell
# daddy" and "come on dear tell me what mommy said" arrives as three messages
# while she is two minutes into an answer to the first one; only the last is
# answered, and it was a follow-up to a question she never saw - so the reply
# could only be about the wrong thing. The skipped lines now ride along with the
# newest one, which is what "answer all three" actually needs.
_SKIPPED_BURST = {}
BURST_NOTE_LIMIT = 6


def _burst_note_for(user_id, label):
    """Take the lines skipped for this person, in the order they were said."""
    if user_id is not None:
        skipped = _SKIPPED_BURST.get(user_id)
    else:
        skipped = _SKIPPED_BURST.get(label)
    if not skipped:
        return ""
    lines = "\n".join(f'- "{text}"' for _, text in skipped[-BURST_NOTE_LIMIT:])
    who = label or "They"
    return (
        f"{who} said these a few seconds ago, in the same burst, before the message "
        f"you are answering now:\n{lines}\n"
        f"The message you are answering is the newest one. Treat the lines above as "
        f"the context it assumes - a follow-up like 'no', 'what about it' or 'come "
        f"on' refers to something earlier in the burst - and answer the newest "
        f"message as one message."
    )


def _remember_skipped(user_id, label, text):
    """Keep a skipped line until the same person's next message is answered."""
    key = user_id if user_id is not None else label
    lines = _SKIPPED_BURST.setdefault(key, [])
    lines.append((label, text))
    # A burst is a burst, not a backlog. Anything older than this is a separate
    # conversation and belongs in the transcript, not in a note about the newest
    # thing somebody said.
    if len(lines) > BURST_NOTE_LIMIT:
        del lines[:-BURST_NOTE_LIMIT]


def _forget_skipped(user_id, label):
    key = user_id if user_id is not None else label
    _SKIPPED_BURST.pop(key, None)


async def drain_queue():
    """Answer what people say, one message at a time and in the order it came."""
    while True:
        queue = her_queue()
        channel, label, text, user_id = await queue.get()
        # Someone messaged her: interrupt any idle thinking so she answers now.
        controller.note_user_input()
        if await superseded_by_a_newer_one(queue, label, user_id):
            # They said it again, better, after this one. Answering the newer
            # message answers this one too, and two replies to the same person
            # seconds apart is what makes a bot feel like a machine.
            _remember_skipped(user_id, label, text)
            queue.task_done()
            print(f"Discord bot: skipped a message from {label} they had already moved past")
            continue
        burst_note = _burst_note_for(user_id, label)
        try:
            await think_and_reply(channel, label, text, user_id, burst_note=burst_note)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Discord bot: turn failed: {type(error).__name__}: {error}")
            # If it's a fast model error, don't show the fuzzy brain message -
            # the local model fallback already happened in controller.ask_ai
            if "fast" in str(error).lower() or "gemini" in str(error).lower() or "quota" in str(error).lower():
                # The local model already answered, this is just noise
                pass
            else:
                try:
                    await channel.send("My brain went fuzzy for a second - ask me again?", allowed_mentions=NO_PINGS)
                except discord.HTTPException:
                    pass
        finally:
            _forget_skipped(user_id, label)
            queue.task_done()


def classify(message):
    """Look at one message and decide what she should do with it.

    Returns None for anything she must not touch at all - other bots, herself, a
    channel or person she is not allowed near. Returns a verdict dict for
    everything else, and the verdict is deliberately not the same as "answer
    this". It carries whether the message was addressed to her, because that is
    now a different thing from whether she has seen it.

    The old version of this function answered one question - was she mentioned -
    and returned None for everything else, which meant every message in the
    channel that was not addressed to her never existed as far as she was
    concerned. She could not join a conversation, could not follow a running
    thread, and could not notice anything about the people she was sitting with,
    because she was only ever handed the messages that had her name in them.
    Every message is now read. Whether she answers it immediately is a separate
    decision, made below.
    """
    author = message.author
    if author.bot or author.id == _client.user.id:
        return None
    # PrivateChannel is the base of direct messages and group messages, so one
    # check covers both. ChannelType.private would miss group chats.
    is_dm = isinstance(message.channel, PrivateChannel)
    if is_dm and not ALLOW_DMS:
        return None
    # The channel allow-list names channels in a server. A direct message has a
    # channel id too, and it is never going to be in a list of server channels,
    # so applying that list to a DM quietly threw away every one of them - the
    # message was dropped without a word, which looks exactly like a bot that is
    # ignoring you. Who is allowed is decided by the user list below instead.
    if ALLOWED_CHANNELS and not is_dm and message.channel.id not in ALLOWED_CHANNELS:
        return None
    if ALLOWED_USERS and author.id not in ALLOWED_USERS:
        return None
    content = message.content or ""

    # A ping of her, by id or by the raw token, is an address. Comparing the
    # mention objects the library hands back is how "she never notices she was
    # mentioned" happens, and the raw token counts too, because a gateway that
    # gives the text without populating mentions would otherwise drop a real ping
    # on the floor and look exactly like being ignored.
    mentioned = any(
        getattr(user, "id", None) == _client.user.id
        for user in getattr(message, "mentions", []) or []
    )
    mention_tokens = [f"<@{_client.user.id}>", f"<@!{_client.user.id}>"]
    # A ping of the role she wears is a ping of her. This is how people actually
    # call a bot: nobody types the long number, they ping "Kookie".
    role_tokens = [f"<@&{role_id}>" for role_id in sorted(_HER_ROLE_IDS)]
    mention_tokens.extend(role_tokens)
    if not mentioned:
        mentioned = any(token in content for token in mention_tokens)

    # Answering somebody's reply to her is not a nicety, it is the same event as
    # being pinged: in Discord, replying to a message is how you talk to whoever
    # wrote it, and requiring a ping on top of it meant the most natural way to
    # continue a conversation with her did nothing at all. Two ways it is
    # recognised, because either one alone misses half the cases. The resolved
    # reference is a real Message object, which is authoritative but only exists
    # when the replied-to message is still in the cache. Our own record of what
    # she sent covers the rest, including when the original was deleted.
    replied_to_her = False
    reference = getattr(message, "reference", None)
    if reference is not None:
        resolved = getattr(reference, "resolved", None)
        if resolved is not None and getattr(getattr(resolved, "author", None), "id", None) == _client.user.id:
            replied_to_her = True
        elif getattr(reference, "message_id", None) in _HER_MESSAGE_IDS:
            replied_to_her = True

    # Her name in the text, without the ping symbol. People do this constantly in
    # a shared channel - "kookie what do you think" - and reading it as an
    # unaddressed line meant she sat silent while being spoken to directly.
    #
    # Position and shape both matter here, because a bare substring search is
    # exactly what makes a bot exhausting to share a channel with. It matches
    # "the kookie is good", "thanks kookie" and - the one that matters most -
    # "kookie is so cute", none of which are addressed to her, and answering
    # those is noise on top of noise.
    #
    # So her name has to appear where you would start a sentence to her (the
    # first word, or the word after a greeting) AND what follows has to be aimed
    # at her. A comma, colon or dash right after the name is the clearest
    # signal. A question mark anywhere in the rest is the next. And a question
    # word right after the name is the third - "cookie how are you doing" is
    # aimed at her just as much as "cookie, how are you", and requiring a
    # punctuation mark left it sitting in the room unanswered. What this still
    # rejects is a statement about her that never asks her anything, like
    # "kookie is so cute" - which is not addressed to her, and answering it is
    # the noise that makes a shared channel exhausting.
    # "cookie" is listed alongside her display name on purpose. It is not the
    # same word - it is how people actually call her, and matching only the
    # display name left every "cookie hello" sitting in the room unanswered
    # while the bot waited for a word it was never going to hear.
    _NAME_TOKENS = tuple(
        token
        for token in (
            _client.user.display_name, _client.user.name, "kookie", "cookie",
        )
        if token and len(token) > 2
    )
    # The question words are the ones that start a question. The forms of "to be"
    # are deliberately left out: "kookie is so cute" ends in one, and it is a
    # statement about her, not a question to her, so counting it made her reply
    # to compliments about herself. "are" stays because "are you okay" is the
    # most common way somebody actually talks to her.
    _QUESTION_WORDS = (
        "how", "what", "where", "why", "who", "whom", "when", "can", "could",
        "should", "would", "will", "shall", "do", "does", "did", "are",
    )
    named = False
    for token in _NAME_TOKENS:
        opening = (
            rf"\s*(?:{GREETING_OPENERS})\s+{re.escape(token)}\b"
            rf"|\s*{re.escape(token)}\b"
        )
        opening_match = re.match(opening, content, re.IGNORECASE)
        if opening_match is None:
            continue
        after = content[opening_match.end():]
        if after.lstrip().startswith((",", ":", "-")):
            named = True
            break
        if "?" in after:
            named = True
            break
        first_word = after.lstrip().split(None, 1)
        if first_word and first_word[0].rstrip("!?").lower() in _QUESTION_WORDS:
            named = True
            break

    if not mentioned and not named:
        # A message that pings her but gets dropped anyway is worth saying out
        # loud. Silence here is what made a broken ping look like a slow model.
        for role in getattr(message, "mention_roles", []) or []:
            if getattr(role, "id", None) in _HER_ROLE_IDS:
                print(
                    f"Discord bot: ignoring a message that pinged her role in "
                    f"#{getattr(message.channel, 'name', '?')} "
                    f"(allowed={bool(ALLOWED_CHANNELS)} allowed_ids={sorted(ALLOWED_CHANNELS) or 'all'})",
                    flush=True,
                )
                break

    # Nobody pings anyone in a private chat: it is two people talking, and the
    # whole point of a DM is that everything in it is meant for her. Holding the
    # mention rule here is what made DMs look broken - she was online, she was
    # reachable, and she answered nothing until you typed a role id at her.
    #
    # REQUIRE_MENTION is the strict setting: only a real ping of her, a real ping
    # of her role, a reply to her, or a DM counts as being spoken to, and her name
    # typed out in a channel does not. It is off by default because "kookie what
    # do you think" is a message addressed to her whether or not the author
    # remembered the @ symbol.
    addressed = is_dm or mentioned or replied_to_her or (named and not REQUIRE_MENTION)
    # With the mention requirement relaxed she is allowed to answer a message that
    # names her, but she is still not required to answer every line in the
    # channel - see _handle_message, which reads all of them and replies only to
    # the ones that are addressed to her.
    if not addressed and not READ_EVERYTHING:
        return None
    if not content.strip():
        # An attachment or a sticker with no text is still something she has read
        # and can react to, but it is not something she can answer, and the model
        # needs words to work with.
        return {
            "text": "",
            "addressed": addressed,
            "is_dm": is_dm,
            "replied": replied_to_her,
            "silent": True,
        }
    # The mention token comes off whether or not the gateway listed her, so the
    # model is never handed the raw "<@12345>" as part of the question.
    for token in mention_tokens:
        content = content.replace(token, " ")
    for prefix in _NAME_TOKENS:
        # The same length rule as the name check above, and for the same reason:
        # a bot called "a" or "ok" would otherwise have that prefix cut off the
        # front of every single message it was ever sent.
        if content.strip().lower().startswith(prefix.lower()):
            content = content.strip()[len(prefix):]
            break
    text = content.strip()
    return {
        "text": text,
        "addressed": addressed,
        "is_dm": is_dm,
        "replied": replied_to_her,
        "silent": False,
    }


def wanted(message):
    """Should she answer this one right now? Returns the cleaned text, or None.

    Answering at all and being spoken to are two different questions, and this is
    the second one. A message that was not addressed to her is still read - it
    goes into the room transcript either way - but it does not get a reply on the
    spot, because a model this size takes the better part of a minute to answer
    and a channel where she replies to every line is a channel nobody stays in.
    What she does with those messages is decide on her own, at her own pace,
    which is the point.
    """
    verdict = classify(message)
    if verdict is None:
        return None
    if verdict.get("silent") or not verdict.get("addressed"):
        return None
    return verdict["text"] or None



_HER_ROLE_IDS = set()


async def _remember_her_roles(client):
    """Note the roles she answers to, so a role ping counts as a ping.

    Nobody types a bot's long id to get its attention: they ping the "Kookie"
    role it wears. Discord files that under mention_roles rather than mentions,
    so a mention check that only looks at mentions drops every real ping and the
    bot looks broken rather than unmentioned.
    """
    global _HER_ROLE_IDS
    found = set()
    try:
        for guild in getattr(client, "guilds", []):
            me = guild.me or guild.get_member(client.user.id)
            if me is None:
                continue
            for role in getattr(me, "roles", []):
                # @everyone is not something anyone can ping.
                if getattr(role, "id", None) and role.name != "@everyone":
                    found.add(int(role.id))
    except Exception as error:
        print("Discord bot: could not read her roles:", error)
    _HER_ROLE_IDS = found
    if found:
        print(f"Discord bot: also answering pings of her roles {sorted(found)}")


def _loop():
    """Her event loop, or None if the bot is not up yet."""
    return getattr(_client, "loop", None) if _client else None


def _record_her_send(sent, text=None):
    """Remember the id of one of her own messages, so a reply to it is recognised.

    Only the id, not the words: where her words go into the room is owned by the
    turn that produced them - think_and_reply for an answer, run_idle_check for a
    line she started herself. Recording in both places meant every proactive line
    appeared in the transcript twice, and recording here alone missed her replies
    entirely whenever TTS was off, because that path finishes by editing a
    placeholder rather than sending.
    """
    if sent is None:
        return None
    message_id = getattr(sent, "id", None)
    if message_id is not None:
        _HER_MESSAGE_IDS.append(message_id)
    return sent


def channel_name_of(message):
    channel = getattr(message, "channel", None)
    return getattr(channel, "name", None) or (str(channel.id) if channel else None)


def channel_name_of_channel(channel):
    """A channel's name, or its id when Discord will not give her a name."""
    return getattr(channel, "name", None) or (
        str(getattr(channel, "id", "")) or None
    )


def _words_key(text):
    """The message reduced to comparable words, so two sends can be compared."""
    return " ".join(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def _she_already_said_that(text, channel_id):
    """True when this is word for word what she last said in this channel.

    A duplicate send is the loudest version of "she keeps saying the same
    thing": the same sentence posted twice in a row, by a model that has been
    told not to and a small box it is easy to be in a rut on. Short answers are
    allowed to repeat, because "yes" twice in a row is the right answer twice.
    A whole sentence going over itself is not.
    """
    words = _words_key(text)
    if len(words.split()) < 6:
        return False
    return words in _HER_LAST_WORDS.get(str(channel_id), set())


async def _speak_into(channel, text, speak=True):
    """Put one message in the channel, read aloud when Discord will let her.

    Discord only ever speaks a message that was sent that way - an edit cannot
    gain a voice afterwards - so a long reply is spoken from its first piece and
    the rest is left as text. Reading every piece would talk over itself.
    """
    if not text or not text.strip():
        return None
    
    # Check if the reply contains an image URL and send it
    import re as _re
    image_match = _re.search(r'https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s]*)?', text, _re.IGNORECASE)
    if image_match:
        image_url = image_match.group(0)
        caption = text.replace(image_url, "").strip()
        result, error = await _send_image(channel, image_url, caption)
        if result is not None:
            _record_her_send(result)
            return result
        # Fall through to text send if image fails
    
    cid = getattr(channel, "id", None)
    if _she_already_said_that(text, cid):
        print(f"Discord bot: not sending the same sentence twice in #{getattr(channel, "name", "?")}")
        return None
    pieces = split_reply(text)
    sent = None
    for index, piece in enumerate(pieces):
        try:
            sent = await channel.send(
                piece,
                tts=bool(speak and TTS and index == 0),
                allowed_mentions=NO_PINGS,
            )
        except discord.HTTPException as error:
            if not (speak and TTS):
                print("Discord bot: could not send:", error)
                return None
            try:
                sent = await channel.send(piece, allowed_mentions=NO_PINGS)
            except discord.HTTPException as second:
                print("Discord bot: could not send:", second)
                return None
    if cid is not None:
        recent = _HER_LAST_WORDS.setdefault(str(cid), collections.deque(maxlen=8))
        recent.append(_words_key(text))
    _record_her_send(sent)
    if speak and getattr(channel, "guild", None):
        guild_id = channel.guild.id
        voice_enabled = get_voice_enabled()
        import discord_bot as self_mod
        print(f"[_speak_into] guild_id={guild_id} in _VOICE_ENABLED={guild_id in voice_enabled} _VOICE_ENABLED={list(voice_enabled)} module_id={id(self_mod)}")
        if guild_id in voice_enabled:
            # Use chat reader to speak in voice channel (handles self mode for bot's own messages)
            try:
                from discord_commands import chat_reader_speak_message
                bot_user_id = _client.user.id if _client and _client.user else None
                asyncio.create_task(chat_reader_speak_message(guild_id, text, _client.user.display_name if _client and _client.user else "Kookie", author_id=bot_user_id, bot_user_id=bot_user_id, text_channel_id=None))
            except Exception as e:
                print(f"[CHAT READER] Error triggering for own message: {e}")
        else:
            print(f"[CHAT READER] Skipped voice reply - guild {guild_id} not in voice (run /voice first). _VOICE_ENABLED: {list(voice_enabled)}")
    return sent


def _dm_label(target, fallback=""):
    """What she calls the person she is writing to, in the confirmation.

    The relation rather than the account name: telling kazuma "sent kazuma a
    DM" is a stranger's phrasing, and she calls him Dad.
    """
    relation = str((target or {}).get("relation") or "").strip()
    if relation:
        return relation[:1].upper() + relation[1:]
    name = str((target or {}).get("display_name") or fallback or "them").strip()
    return name or "them"


async def _dm_channel_for(user_id):
    """The private channel to somebody, or None when Discord will not give her one.

    get_user only sees members who are in the cache, so a fetch is tried as well.
    Discord refuses with Forbidden when that account has server DMs turned off, or
    when she has no shared server with them at all - a wall, not a bug, and the
    caller has to be able to say so out loud.
    """
    if _client is None:
        return None, "she is not connected"
    try:
        target_id = int(user_id)
    except (TypeError, ValueError):
        return None, "that is not an account"
    user = _client.get_user(target_id)
    if user is None:
        try:
            user = await _client.fetch_user(target_id)
        except discord.HTTPException as error:
            return None, f"Discord would not look them up ({error.status})"
    if user is None:
        return None, "no account by that id here"
    try:
        return await user.create_dm(), None
    except discord.Forbidden:
        return None, "that account has server DMs turned off"
    except discord.HTTPException as error:
        return None, f"Discord said {error.status}"


async def _send_dm(channel_like, text):
    """Send one private message, out loud if she is allowed to.

    The same rule as a channel message: a message Discord will not speak is
    still a message worth sending, so the words go out either way.
    """
    try:
        return _record_her_send(await channel_like.send(text, tts=bool(TTS), allowed_mentions=NO_PINGS)), None
    except discord.Forbidden as error:
        if not TTS:
            return None, f"Discord refused it ({error.status})"
        try:
            return _record_her_send(await channel_like.send(text, allowed_mentions=NO_PINGS)), None
        except discord.HTTPException as second:
            return None, f"Discord refused it ({second.status})"
    except discord.HTTPException as error:
        return None, f"Discord refused it ({error.status})"


async def _send_image(channel_like, image_url, caption=""):
    """Download an image from a URL and send it to Discord."""
    import io
    import requests as _requests
    import urllib.parse

    # SSRF validation: only https, no loopback/private/link-local/CGNAT, no redirects
    parsed = urllib.parse.urlparse(image_url)
    if parsed.scheme != "https":
        return None, f"Only https URLs are allowed (got {parsed.scheme!r})"
    hostname = parsed.hostname
    if not hostname:
        return None, "URL has no hostname"
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError as error:
        return None, f"Could not resolve {hostname}: {error}"
    cgnat_start = ipaddress.IPv4Address("100.64.0.0")
    cgnat_end = ipaddress.IPv4Address("100.127.255.255")
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local:
            return None, f"URL resolves to {addr}, which is a loopback/private/link-local address"
        if ip.version == 4 and cgnat_start <= ip <= cgnat_end:
            return None, f"URL resolves to {addr}, which is in the CGNAT range"
    try:
        resp = _requests.get(image_url, timeout=30, headers={"User-Agent": "KookieAi/1.0"}, allow_redirects=False)
        if resp.status_code != 200:
            return None, f"Could not download image: HTTP {resp.status_code}"
        
        # Check it's actually an image
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return None, f"URL did not return an image: {content_type}"
        
        # Discord has a 25MB limit for free accounts, 8MB for most
        if len(resp.content) > 8 * 1024 * 1024:
            return None, "Image is too large for Discord (>8MB)"
        
        file = discord.File(io.BytesIO(resp.content), filename="kookie_image.png")
        if caption:
            return await channel_like.send(caption, file=file, allowed_mentions=NO_PINGS), None
        return await channel_like.send(file=file, allowed_mentions=NO_PINGS), None
    except Exception as error:
        return None, f"Could not send image: {error}"


def _compose_dm_blocking(target_name, speaker_name, topic=""):
    """Ask her for a DM to send, on a thread, under the same lock as a reply.

    Two generations at once on this box is the thing that makes her slow, so this
    waits its turn like everything else that talks to the model.
    """
    if not controller.conversation_lock.acquire(timeout=900):
        raise TimeoutError("Kookie was busy for a very long time")
    try:
        return controller.compose_dm_text(target_name, speaker_name, topic)
    finally:
        controller.conversation_lock.release()


async def _run_dm_request(channel, speaker_label, speaker_id, request):
    """Send what somebody asked her to send privately, and say that she did."""
    target = request.get("target") or {}
    target_id = request.get("target_id") or str(target.get("id") or "")
    who = _dm_label(target, request.get("target_name", ""))
    body = (request.get("body") or "").strip()
    if not body:
        # Told to write, not told what to say. She writes it, and if she cannot,
        # nothing is sent - a blank DM is worse than none.
        #
        # The asker is left out of the prompt when they are the recipient:
        # "kazuma asked me to send this to you" is a note to himself, and she
        # wrote it into a DM addressed to him.
        asker = speaker_label
        if asker and who and asker.strip().lower() == str(who).strip().lower():
            asker = ""
        body = await asyncio.to_thread(
            _compose_dm_blocking, who, asker, ""
        )
        body = (body or "").strip()
    if not body:
        await _speak_into(channel, "I couldn't think of anything to say, so I left it.")
        return None
    dm_channel, problem = await _dm_channel_for(target_id)
    if dm_channel is None:
        print(f"Discord bot: DM to {who} failed: {problem}")
        await _speak_into(channel, f"I wanted to DM {who} but {problem}.")
        return None
    sent, problem = await _send_dm(dm_channel, body)
    if sent is None:
        print(f"Discord bot: DM to {who} failed: {problem}")
        await _speak_into(channel, f"I tried to DM {who} but {problem}.")
        return None
    username = controller.person_username(target_id) or ""
    controller.note_dm_sent(target_id, str(target.get("display_name") or who), username, body)
    print(f"Discord bot: DM to {who} ({target_id}): {body[:120]}")
    await _speak_into(channel, f"Sent {who} a DM 💕")
    return sent


async def _say_on_her_own(text):
    """Post a line she decided to say, without anybody having asked.

    It lands in the conversation she was last part of. If she has never been
    spoken to there is nowhere obvious to put it, so a configured channel is the
    fallback and otherwise she keeps it to herself - a companion who opens with
    "hi" into an empty void is noise, not company.

    The channel chosen is the one people were last talking in, not the one she
    was last pinged in. Those are different places as soon as somebody pings her
    somewhere and carries on talking somewhere else, and speaking up in a
    channel nobody has used in an hour is the visible version of talking to
    yourself.
    """
    channel = _last_channel
    live_name, live_id = controller.room_last_channel()
    if live_id:
        try:
            found = _client.get_channel(int(live_id))
        except (ValueError, TypeError, AttributeError):
            found = None
        if found is not None:
            channel = found
    if channel is None and PROACTIVE_CHANNEL:
        try:
            channel = _client.get_channel(int(PROACTIVE_CHANNEL))
        except (ValueError, AttributeError):
            print(f"Discord bot: {PROACTIVE_CHANNEL!r} is not a channel id")
            channel = None
    if channel is None:
        print("Discord bot: had something to say, but nobody has talked to her yet")
        return
    text = (text or "").strip()
    if not text:
        return
    print(f"Discord bot: speaking up in #{getattr(channel, 'name', '?')}")
    await _speak_into(channel, text)
    # Asking her to look at the screen as she says it is what makes an unprompted
    # line feel like her talking rather than a log line appearing.
    try:
        await channel.typing()
    except (discord.HTTPException, AttributeError):
        pass


def _proactive_sink(result):
    """Called by controller on the idle timer's thread when she speaks up."""
    if not PROACTIVE:
        return
    text = (result or {}).get("reply", "")
    if not text.strip():
        return
    loop = _loop()
    if loop is None or loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(_say_on_her_own(text), loop)


async def _handle_ready(client):
    global _worker
    print(f"Discord bot: online as {client.user} (asking for #{client.user.id})")
    try:
        await _remember_her_roles(client)
        print("Discord bot: roles remembered")
    except Exception as e:
        print(f"Discord bot: _remember_her_roles failed: {e}")
        import traceback
        traceback.print_exc()
    try:
        controller.set_proactive_sink(_proactive_sink)
        print("Discord bot: proactive sink set")
    except Exception as e:
        print(f"Discord bot: set_proactive_sink failed: {e}")
        import traceback
        traceback.print_exc()
    try:
        controller.schedule_idle_check()
        print("Discord bot: idle check scheduled")
    except Exception as e:
        print(f"Discord bot: schedule_idle_check failed: {e}")
        import traceback
        traceback.print_exc()
    if _worker is None or _worker.done():
        _worker = asyncio.ensure_future(drain_queue())
        print("Discord bot: drain_queue worker started")
    try:
        await _tree.sync()
    except discord.HTTPException as e:
        print("Discord bot: slash sync failed:", e)


async def _maybe_switch(text, message, label):
    """Thin wrapper so callers that import _maybe_switch still work."""
    return await maybe_switch(text, message, label)


async def _handle_message(message):
    global _last_channel
    # Read first, decide second, and never the other way round. classify() is
    # what decides whether she is allowed near this message at all; everything
    # it hands back is a message she has seen, whether or not she is going to
    # answer it. Recording it happens before the reply decision so that "she has
    # read the room" is true even for a message she stays quiet about, which is
    # the entire difference between a companion sitting in a channel and a
    # helpdesk that only wakes up when summoned.
    verdict = classify(message)
    if verdict is None:
        return
    author = message.author
    channel_name = getattr(message.channel, "name", None) or str(message.channel.id)
    label = author.display_name or str(author)
    user_id = author.id
    # Her own message is not new information about the room - it is already in
    # the transcript from the moment she sent it - and writing it again would
    # make every reply she has ever given the last line of the room forever.
    is_kookie = user_id == _client.user.id
    controller.note_room_message(
        channel_name, label, verdict.get("text", ""),
        is_kookie=is_kookie, channel_id=getattr(message.channel, "id", None),
    )
    if verdict.get("silent"):
        # An attachment or an emoji-only message. She has seen it and it counts
        # as the room being alive, but there are no words to answer.
        return
# "/kookie draw: something" or "/draw something" - generate an image
    # Handled here, before the "not addressed" check, so a bare /draw works
    # without needing to ping her first.
    raw_text = verdict.get("text") or ""
    draw_match = re.match(r"^(?:/kookie\s+draw|/draw)[:\s]+(.+)$", raw_text, re.IGNORECASE)
    if draw_match:
        prompt = draw_match.group(1).strip()
        if prompt:
            print(f"Discord bot: {label} asked Kookie to draw: {prompt[:80]}")
            await message.channel.send(f"🎨 Drawing: {prompt}...", allowed_mentions=NO_PINGS)
            
            # Build the best possible prompt for character accuracy
            enhanced_prompt = prompt
            
            # Check for "character from show" format
            from_match = re.search(r"(.+?)\s+from\s+(.+)$", prompt, re.IGNORECASE)
            if from_match:
                char_name = from_match.group(1).strip()
                show_name = from_match.group(2).strip()
                enhanced_prompt = (
                    f"{char_name} from {show_name}, "
                    f"anime character portrait, official art style, "
                    f"high quality, detailed face, accurate character design, "
                    f"vibrant colors, professional artwork, "
                    f"sharp focus, studio quality, 4K detail"
                )
            
            # Auto-detect anime style for non-"from" prompts
            # Instead of a fixed keyword list, use a broader approach:
            # if the user mentions "from <show>" or uses anime-style terms,
            # boost the prompt with anime quality parameters.
            # Anime style terms (hoisted to ANIME_STYLE_TERMS at module level)
            
            is_anime = any(kw in prompt.lower() for kw in ANIME_STYLE_TERMS)
            
            # Ask the fast model whether this is an anime request. It knows
            # every series in existence, so a keyword list is the fallback
            # rather than the rule - and when the fast model is offline the
            # keyword list still catches the obvious ones.
            llm_anime = await asyncio.to_thread(fastai.is_anime_prompt, prompt)
            if llm_anime is True:
                is_anime = True
            elif llm_anime is False:
                is_anime = False
            
            # Support EVERY anime: if user says "from <anything>" or uses
            # anime-style terms, boost the prompt with quality parameters.
            # The "from" pattern already handles character+show format.
            # For other prompts, check if they look like anime requests.
            looks_anime = is_anime or from_match or "anime" in prompt.lower()
            
            if looks_anime and not from_match:
                enhanced_prompt = (
                    f"{prompt} (anime style, high quality, detailed, vibrant colors, "
                    f"professional artwork, character design, official art, "
                    f"sharp focus, studio quality, 4K detail)"
                )
            
            # Try models for best results (flux-pro/realisticVision require payment)
            models_to_try = [
                ("flux", "high"),
            ]
            
            for model_name, quality in models_to_try:
                try:
                    image_url = (
                        f"https://image.pollinations.ai/prompt/"
                        f"{urllib.parse.quote(enhanced_prompt)}"
                        f"?width=512&height=512&nologo=true&enhance=true&noWatermark=true"
                        f"&model={model_name}&quality={quality}"
                    )
                    sent, error = await _send_image(message.channel, image_url, f"Here's what I drew: {enhanced_prompt[:80]}")
                    if sent is not None:
                        print(f"Discord bot: sent image via {model_name} for '{prompt[:40]}...'")
                        break
                    print(f"Discord bot: {model_name} failed: {error}")
                except Exception as e:
                    print(f"Discord bot: {model_name} error: {e}")
                    continue
            else:
                await message.channel.send(f"Sorry, I couldn't draw that - all models failed.", allowed_mentions=NO_PINGS)
            return
    # She is not being spoken to. It is in the room now and in the thought that
    # runs next; answering it here would spend a minute of the model on a message
    # nobody asked her about, and the queue would back up behind it.
    if not verdict.get("addressed"):
        print(
            f"Discord bot: read {label} in #{channel_name} without replying"
            f"{' (replying to her)' if verdict.get('replied') else ''}"
        )
        return
    text = verdict.get("text") or ""
    _last_channel = message.channel
    if not text:
        return
    print(f"Discord bot: {label} in #{channel_name}: {text[:120]}")
    # A switch typed in the channel is a switch, not something to think about.
    # Mommy asking for /stop discord in Discord is a perfectly reasonable thing to
    # do - it is what actually happened - and it used to be handed to the model as
    # a question, which spent a generation producing a reply about stopping and
    # left her on Discord. Handled here, before anything is queued.
    if await _maybe_switch(text, message, label):
        return
    # If the speaker is in a voice channel, automatically join it so she can
    # reply by voice as well as text. This only happens when the message is
    # actually addressed to her, so she does not crash every voice channel
# she can see just by reading the room.
    if getattr(author, "voice", None) and getattr(author.voice, "channel", None):
        guild_id = getattr(getattr(author.voice.channel, "guild", None), "id", None)
        if guild_id is not None:
            client = _VOICE_CLIENTS.get(guild_id)
            joined_now = False
            if client is None or not client.is_connected():
                client, err = await _voice_join(author.voice.channel)
                if client is not None:
                    get_voice_enabled().add(guild_id)
                    joined_now = True
                    print(f"Discord bot: joined voice in guild {guild_id} because {label} is there")
                    try:
                        await message.channel.send(f"Joined {author.voice.channel.name} - I will speak there now 💕", allowed_mentions=NO_PINGS)
                    except Exception:
                        pass
                else:
                    print(f"Discord bot: auto-join failed for {label}: {err}")
            else:
                get_voice_enabled().add(guild_id)
            if guild_id in get_voice_enabled():
                voice_client = _VOICE_CLIENTS.get(guild_id)
                if voice_client is not None and voice_client.is_connected():
                    _HER_VOICE_NAME = re.compile(
                        r"(?<![a-z])(?:kookie|kuki|cookie|kook)(?![a-z])",
                        re.IGNORECASE,
                    )

                    async def _on_speech(user, text):
                        name = getattr(user, "display_name", None) or str(user.id)
                        vc = _VOICE_CLIENTS.get(guild_id)
                        if vc is None or not vc.is_connected():
                            return
                        bot_user_id = getattr(getattr(vc, "user", None), "id", None)
                        if bot_user_id is not None and user.id == bot_user_id:
                            return
                        if not _HER_VOICE_NAME.search(text or ""):
                            return
                        try:
                            reply = await asyncio.to_thread(
                                controller.ask_ai,
                                f"{name} said: \"{text}\". Reply to what they said, briefly and warmly.",
                                voice_mode=True,
                            )
                            text = (reply or {}).get("reply", "").strip()
                            if not text:
                                return
                            vc = _VOICE_CLIENTS.get(guild_id)
                            if vc is None or not vc.is_connected():
                                return
                            target_channel = getattr(vc, "channel", None) or author.voice.channel
                            await voice_speak(target_channel, text)
                        except Exception as e:
                            print("Discord bot: voice reply error:", e)
                    listen_task = asyncio.create_task(voice_listen(author.voice.channel, _on_speech))
                    try:
                        await listen_task
                    except Exception as e:
                        print(f"Discord bot: voice_listen error for auto-join guild={guild_id}: {e}")
    # Chat reader: if enabled for this channel and we're in voice, read the message aloud
    # In "self" mode, we read Kookie's own replies; in "all" mode, we read all messages; in "others" mode, we read only other users' messages
    if text and not verdict.get("silent"):
        try:
            from discord_commands import chat_reader_is_enabled, chat_reader_speak_message, chat_reader_get_mode
            guild_id = getattr(message.guild, "id", None)
            if guild_id and chat_reader_is_enabled(guild_id, message.channel.id):
                mode = chat_reader_get_mode(guild_id, message.channel.id)
                print(f"[CHAT READER] channel={message.channel.id} mode={mode} is_kookie={is_kookie}")
                # "all" mode: read all messages from users
                if mode == "all" and not is_kookie:
                    print(f"[CHAT READER] TRIGGERING for user message (mode=all)")
                    asyncio.create_task(chat_reader_speak_message(guild_id, text, label, text_channel_id=message.channel.id))
                # "others" mode: read only other users' messages (not Kookie's)
                elif mode == "others" and not is_kookie:
                    print(f"[CHAT READER] TRIGGERING for user message (mode=others)")
                    asyncio.create_task(chat_reader_speak_message(guild_id, text, label, text_channel_id=message.channel.id))
                # "self" mode is handled in _speak_into for Kookie's replies
        except Exception as e:
            print(f"[CHAT READER] Error: {e}")
    # Remember who said it before anything else can go wrong. The id travels with
    # the queued message rather than the name, so that a rename between the
    # message being accepted and being answered does not split one person's
    # history into two.
    controller.note_discord_person(
        user_id, label, getattr(author, "name", "") or "", text
    )
    # She is told the speaker is Mommy, so the line itself has to agree. Left as
    # the raw nickname the question arrives reading "cookie: im home" under a
    # header that says the speaker is Mommy, and she has to work out that the two
    # names are the same person.
    name = controller.person_display_name(user_id, label)
    if name != label:
        print(f"Discord bot: {label} is {name} to her")
    # "dm kazuma that i'm busy" is an instruction to send something, not a
    # question about it, so it never goes into the reply queue: answering it in
    # the channel would put the message on the channel screen where it was typed
    # and nowhere else. This is the only message she acts on instead of replying
    # to, and only for somebody allowed to ask - her own person, or her family.
    # A request she cannot place (no such name, or a stranger asking) falls
    # through to an ordinary reply, because "dm who?" is better than silence.
    dm_request = controller.parse_dm_request(text, user_id)
    if dm_request is not None:
        if not dm_request.get("target"):
            print(f"Discord bot: {label} asked for a DM but nobody matched: {text[:80]}")
            dm_request = None
        elif not controller.dm_request_allowed(user_id):
            print(
                f"Discord bot: {label} is not somebody she takes DM requests from - "
                "asking her anyway, so replying normally"
            )
            dm_request = None
    if dm_request is not None:
        await _run_dm_request(message.channel, name, user_id, dm_request)
        return
    queue = her_queue()
    # A busy channel can fill faster than one slow model answers it, and a queue
    # of forty messages means forty replies an hour late, each answering a
    # conversation that has already moved on. The oldest go instead.
    dropped = 0
    while queue.qsize() >= MAX_PENDING:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        queue.task_done()
        dropped += 1
    if dropped:
        print(f"Discord bot: dropped {dropped} stale message(s) rather than answer late")
    # The name is part of what she is told, and it travels beside the message
    # rather than inside it. It used to be pasted into the front of the text -
    # queue.put((channel, name, f"{name}: {text}", user_id)) - so every memory note
    # from this front end read "User: Mommy: this is not a game", the transcript
    # block rendered it as Mommy: "Mommy: this is not a game", and her prompt said
    # 'Mommy says: "Mommy: this is not a game"'. It was already in the message
    # segment as the speaker's name, so the copy bought nothing and cost the one
    # thing a memory test needs: her own words, stored exactly as she typed them.
    await queue.put((message.channel, name, text, user_id))


async def on_message_edit(message):
    """A message being edited counts as having just been said.

    Without this, correcting a typo meant the corrected version was never read:
    the original had already been recorded, and an edit is the only event that
    says the earlier line was wrong.
    """
    if getattr(message, "author", None) is None:
        return  # a partial edit payload from the gateway
    await _handle_message(message)



# discord.py dispatches every event to the attribute named after it, so these two
# names are not a style choice: register anything else and the library quietly
# runs its own built-in handler instead. The bot then logs in, joins the guild
# and discards every message it is sent, which looks exactly like being ignored.
async def on_ready():
    await _handle_ready(_client)


async def on_message(message):
    await _handle_message(message)


_stopped = False


# How long she gets to leave nicely before she is simply made to.
SHUTDOWN_GRACE_SECONDS = 15
_watchdog = None


def _arm_watchdog():
    """Make certain the process really goes, however wedged it is.

    Every other exit is cooperative, and being cooperative is not enough here. A
    turn in progress is a worker thread blocked on a lock or on a model request,
    and measured on this box one turn took 550 seconds; cancelling the task does
    not interrupt a thread, and closing the gateway can wait behind it. So SIGTERM
    was sometimes doing nothing visible at all: a bot was told to stop, went on
    reading and answering messages in the channel for minutes afterwards, and
    told whoever asked that she was off Discord. That is the exact complaint this
    exists to make impossible, and it is why "she did not stop" and "she said she
    did" happened at the same time.

    So the graceful path still runs first - she says goodbye, closes the gateway,
    stops cleanly - and this only fires if that has not finished in time. It is
    the backstop, and it exits the process outright rather than asking nicely a
    second time, because asking nicely is what just failed. Fifteen seconds is
    long enough for a clean shutdown on a quiet bot and short enough that nobody
    is still waiting on it.
    """
    global _watchdog
    if _watchdog is not None:
        return _watchdog

    def leave():
        print(
            f"Discord bot: still here {SHUTDOWN_GRACE_SECONDS}s after being told to "
            "stop, so I am leaving now."
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Deliberately not a normal exit: nothing has unwound, there is a thread
        # holding a socket open, and waiting for it is the thing being timed out.
        os._exit(0)

    _watchdog = threading.Timer(SHUTDOWN_GRACE_SECONDS, leave)
    _watchdog.daemon = True
    _watchdog.start()
    return _watchdog


def _disarm_watchdog():
    """Called once the graceful exit has finished, so it cannot fire after it."""
    global _watchdog
    if _watchdog is not None:
        _watchdog.cancel()
        _watchdog = None


async def shutdown(client):
    # Ctrl+C reaches this twice - once from the signal handler and once from the
    # finally block - and a bot that announces its own shutdown three times looks
    # like three bots dying.
    global _stopped
    if _stopped:
        return
    _stopped = True
    print("Discord bot: stopping")
    import traceback
    traceback.print_stack()
    _arm_watchdog()
    if _worker is not None:
        _worker.cancel()
    try:
        await client.close()
    except Exception:
        pass


def _take_over_from_other_bots(limit=20.0):
    """Make sure she is the only one of herself before she starts answering.

    Two bots on one token answer every message twice, and because they read the
    same prompt from the same memory they often say the same thing twice as well
    - which is what it looks like from the channel: the same sentence, posted
    over itself. They also race each other writing her mood file ("could not be
    saved: mood.json.tmp") and take turns queueing for Ollama, so a duplicate is
    also the reason a reply that should take two seconds takes a hundred.

    Nothing in the old paths caught it. controller.start_discord_bot() checks
    first, and ./Kookie --discord checks its own pid file - but a pid file left
    behind by a bot that has already gone, or a launcher started while another
    one was still shutting down, is all it takes, and after that the two simply
    lived together and answered twice.

    The newer one takes over rather than refusing, because "start her again" is
    what somebody means when they start her again. It waits for the old one to
    actually be gone before connecting, and gives up rather than risk two live.
    """
    others = [
        pid for pid in controller._all_discord_bot_pids() if pid != os.getpid()
    ]
    if not others:
        return True
    print(
        "Discord bot: another copy of me is already running (pid"
        f"{'' if len(others) == 1 else 's'} {', '.join(str(pid) for pid in others)}). "
        "Asking it to leave before I start."
    )
    for pid in others:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        still = [pid for pid in others if controller._pid_alive(pid)]
        if not still:
            print("Discord bot: it has gone. I am the only one now.")
            return True
        time.sleep(0.5)
    still = [pid for pid in others if controller._pid_alive(pid)]
    if still:
        print(
            "Discord bot: pid"
            f"{'' if len(still) == 1 else 's'} {', '.join(str(pid) for pid in still)} "
            "would not stop, so I am not starting. Two of us on one token would "
            "answer everything twice - stop that one first."
        )
        return False
    return True


def main():
    token = read_token()
    if not token:
        print(
            "Discord bot: no token found.\n"
            f"  Put it in {TOKEN_FILE}, or export {DISCORD_TOKEN_ENV}=...\n"
            "  In the Discord developer portal: Applications -> your app -> Bot ->\n"
            "  Reset Token, and turn on MESSAGE CONTENT INTENT under Privileged\n"
            "  Gateway Intents, or she will never see what people write."
        )
        return 1

    if os.environ.get("KOOKIE_DISCORD_DEBUG", "").strip() in ("1", "true", "yes", "on"):
        # Worth having when she will not come online: without it a connection
        # problem is a silent wait, because discord.py says nothing on its own.
        logging.basicConfig(level=logging.INFO, format="%(relativeCreated)6.0fms %(name)s: %(message)s")
        for noisy in ("discord.http", "aiohttp", "asyncio", "websockets"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    global _client
    intents = discord.Intents.default()
    intents.message_content = True
    _client = discord.Client(intents=intents)
    global _tree
    _tree = app_commands.CommandTree(_client)
    register_slash_commands(_tree)

    _client.event(on_ready)
    _client.event(on_message)
    _client.event(on_message_edit)

    # The headless half of startup: memory, knowledge and the model, with no
    # avatar and no animation loop, because there is no window to animate.
    # controller.start_runtime(connect_avatar=False, animate_parameters=False)

    if not _take_over_from_other_bots():
        return 1

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Ignore SIGTERM during startup to prevent premature shutdown
    import signal
    
    def signal_handler(signum, frame):
        print(f"[BOT] Received signal {signum}")
        import traceback
        traceback.print_stack(frame)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # def handle_exception(loop, context):
    #     msg = context.get("exception", context["message"])
    #     print(f"[BOT] Exception in event loop: {msg}")
    #     import traceback
    #     traceback.print_exc()
    #     loop.create_task(shutdown(_client))

    # loop.set_exception_handler(handle_exception)

    async def run():
        loop.add_signal_handler(signal.SIGINT, lambda: loop.create_task(shutdown(_client)))
        loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(shutdown(_client)))
        print("[BOT] about to start client...")
        try:
            await _client.start(token)
            print("[BOT] client.start() returned!")
        except discord.LoginFailure:
            # The usual cause is a token that was reset in the portal and not
            # updated here, which is worth saying plainly instead of dumping a
            # traceback of an HTTP 401 at whoever is setting this up.
            print(
                "Discord bot: Discord rejected that token. Reset it in the developer\n"
                "  portal (Applications -> your app -> Bot -> Reset Token) and put the\n"
                "  new one in " + TOKEN_FILE + "."
            )
            return 1
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
            print(f"Discord bot: could not reach discord.com ({type(error).__name__}).")
            print("  She needs to be online for this one; check the network or proxy.")
            return 1
        except BaseException as error:
            print(f"Discord bot: startup failed: {type(error).__name__}: {error}")
            import traceback
            traceback.print_exc()
            return 1
        return 0

    code = 1
    try:
        code = loop.run_until_complete(run()) or 0
        print(f"[BOT] run() completed with code={code}")
    except KeyboardInterrupt:
        print("[BOT] KeyboardInterrupt in main()")
        code = 0
    except BaseException as e:
        print(f"[BOT] main() caught {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        print("[BOT] main() finally block")
        try:
            loop.run_until_complete(shutdown(_client))
            print("[BOT] shutdown completed")
        except Exception as e:
            print(f"[BOT] shutdown failed: {e}")
            import traceback
            traceback.print_exc()
        loop.close()
        try:
            controller.stop_runtime()
        except Exception as error:
            print("Discord bot: shutdown was not clean:", error)
        # The graceful exit worked, so the backstop has nothing left to do. Left
        # armed it would take the process out from under whatever came next.
        _disarm_watchdog()
        print("Discord bot: stopped")
    print("[BOT] main() returning", code)
    return code


def _keepalive():
    """Keep the process alive if client is still connected."""
    import time
    while True:
        try:
            if _client and _client.is_connected():
                time.sleep(60)
            else:
                print("[BOT] keepalive: client not connected, exiting")
                break
        except Exception as e:
            print(f"[BOT] keepalive error: {e}")
            break


if __name__ == "__main__":
    sys.exit(main())
