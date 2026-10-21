from __future__ import annotations

"""Discord commands for Kookie.

Every text command, slash command, voice command and DM command lives here.
discord_bot.py is the gateway; this module is what the gateway calls.
"""

import asyncio
import audioop
import json
import os
import tempfile

import discord
from discord import app_commands
from discord.ext import voice_recv

import controller

import voice


# ---------------------------------------------------------------------------
# Text-command parsing
# ---------------------------------------------------------------------------

def parse_text_command(text: str):
    """Return (action, arg) for a Discord text command, or (None, None).

    action is one of: stop, start, status, voice, voice-off, listen, help, or None.
    """
    action, note = controller.parse_discord_switch(text)
    if action in ("voice", "listen"):
        return action, None
    return action, note


# ---------------------------------------------------------------------------
# Voice commands
# ---------------------------------------------------------------------------

async def voice_join(channel) -> tuple[str | None, str | None]:
    """Join a voice channel.

    Returns (client, None) on success, (None, error_msg) on failure.
    Uses VoiceRecvClient so we can listen as well as speak.
    """
    guild_id = getattr(getattr(channel, "guild", None), "id", None)
    if guild_id is None:
        return None, "that is not a guild channel"

    from discord_bot import _VOICE_CLIENTS, _VOICE_LISTENERS, _VOICE_SINKS, get_voice_enabled

    if guild_id in _VOICE_CLIENTS:
        client = _VOICE_CLIENTS[guild_id]
        if client.is_connected():
            print(f"[VOICE JOIN] Already connected to guild {guild_id}, channel={getattr(client, 'channel', None)}")
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
        print(f"[VOICE JOIN] Connected to guild {guild_id}, client.channel={getattr(client, 'channel', 'NO CHANNEL ATTR')}")
        return client, None
    except Exception as e:
        print(f"[VOICE JOIN] Failed to connect: {e}")
        return None, str(e)


async def voice_leave(guild_id: int):
    """Disconnect from a guild's voice channel, if connected."""
    from discord_bot import _VOICE_CLIENTS, _VOICE_LISTENERS, _VOICE_SINKS, _voice_leave as _do_voice_leave, get_voice_enabled
    print(f"[VOICE LEAVE] guild_id={guild_id} clients={list(_VOICE_CLIENTS.keys())} _VOICE_ENABLED={list(get_voice_enabled())}")
    await _do_voice_leave(guild_id)
    _VOICE_LISTENERS.pop(guild_id, None)
    _VOICE_SINKS.pop(guild_id, None)
    # Fallback: if anything is still connected under this guild, force-clear it.
    client = _VOICE_CLIENTS.get(guild_id)
    if client is not None:
        try:
            await client.disconnect(force=True)
        except Exception:
            pass
        _VOICE_CLIENTS.pop(guild_id, None)
    print(f"[VOICE LEAVE] after: _VOICE_ENABLED={list(get_voice_enabled())}")


async def voice_speak(channel, text: str) -> bool:
    """Synthesize text with Piper and play it in the channel's voice chat."""
    if not text or not text.strip():
        print("Discord commands: voice_speak called with empty text")
        return False
    guild_id = getattr(getattr(channel, "guild", None), "id", None)
    if guild_id is None:
        print("Discord commands: voice_speak no guild_id")
        return False

    from discord_bot import _VOICE_CLIENTS
    client = _VOICE_CLIENTS.get(guild_id)
    if client is None or not client.is_connected():
        print(f"Discord commands: voice_speak client missing or disconnected guild={guild_id}")
        return False

    spoken = voice.sanitize_for_speech(text)
    if not spoken:
        print("Discord commands: voice_speak sanitized to empty")
        return False

    wav_path = None
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        print(f"[VOICE] voice_speak synthesizing '{spoken[:50]}' to {wav_path}")
        await __import__("asyncio").to_thread(
            voice._synthesize_with_piper, spoken, wav_path, 1.0
        )
        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            print("[VOICE] voice_speak wav missing or empty")
            return False
        size = os.path.getsize(wav_path)
        print(f"[VOICE] voice_speak playing {size} bytes")
        audio = discord.FFmpegPCMAudio(wav_path)
        print(f"[VOICE] calling client.play() on client={client} (type={type(client)})")
        client.play(audio)
        print("[VOICE] waiting for playback to finish...")
        for i in range(480):
            if not client.is_playing():
                print(f"[VOICE] playback finished after {i} checks")
                break
            await __import__("asyncio").sleep(0.1)
        else:
            print("[VOICE] playback timeout after 48s")
        print("[VOICE] voice_speak finished playing")
        return True
    except Exception as e:
        import traceback
        print("[VOICE] voice speak error:", e)
        traceback.print_exc()
        return False
    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Voice listening
# ---------------------------------------------------------------------------

_VOICE_LISTENERS = {}
_VOICE_SINKS = {}


async def voice_listen(channel, on_speech) -> bool:
    """Start listening to a voice channel.

    Uses Vosk for local offline speech recognition. A phrase is closed when
    the audio goes quiet for about a second after speech, and the text is then
    handed to on_speech. The bot's own audio is filtered out so she does not
    answer herself.
    """
    print("[VOICE] voice_listen called")
    guild_id = getattr(getattr(channel, "guild", None), "id", None)
    if guild_id is None:
        print("Discord commands: voice_listen no guild_id")
        return False

    from discord_bot import _VOICE_CLIENTS
    client = _VOICE_CLIENTS.get(guild_id)
    if client is None or not client.is_connected():
        print("Discord commands: voice_listen client not connected guild=" + str(guild_id))
        return False
    if not hasattr(client, "listen"):
        print("Discord commands: voice_listen client has no listen method")
        return False

    # Get bot user id to filter self-audio
    try:
        bot_user = getattr(client, "user", None) or getattr(client, "_connection", None)
        bot_user_id = getattr(bot_user, "id", None) if bot_user else None
    except Exception:
        bot_user_id = None

    # Import Vosk model once
    try:
        import vosk
        from pathlib import Path
        model_path = Path(".venv/lib/python3.13/site-packages/speech_recognition/models/vosk/model")
        if not model_path.exists():
            print("Discord commands: voice_listen vosk model missing")
            return False
        model = vosk.Model(str(model_path))
    except Exception as e:
        print(f"Discord commands: voice_listen vosk init error: {e}")
        return False

    def _is_silent(pcm):
        """True when a chunk carries no speech, so Vosk can close the phrase.

        The threshold is a guess at what a quiet room sounds like in 16kHz
        mono 16-bit. It is logged rather than tuned silently, because the
        right value depends on this specific mic, this specific Discord
        client and the Opus codec in between - a number that works on a test
        file does not work on a live call, and the only way to know is to
        watch the numbers come in.
        """
        if not pcm:
            return True
        n = len(pcm) // 2
        if n == 0:
            return True
        total = 0
        for i in range(0, len(pcm) - 1, 2):
            v = int.from_bytes(pcm[i:i + 2], "little", signed=True)
            total += v * v
        rms = (total / n) ** 0.5
        _last_rms.append(rms)
        if len(_last_rms) > 400:
            _last_rms.pop(0)
        return rms < 250.0

    _last_rms = []
    _buffers = {}
    _recognizers = {}
    _silence_bytes = {}
    _last_partial = {}
    loop = __import__("asyncio").get_running_loop()
    # Vosk hands back text only when a phrase is finished, and it finishes one
    # only once it has heard silence after it: PartialResult stays empty for
    # the whole stream and FinalResult gives back everything at the end. So
    # the boundary has to be found by what the audio is doing, not by what
    # Vosk says - a second of quiet after speech is the signal to close the
    # phrase, read it out and start a new one. Measured on real 48kHz stereo
    # speech: the old per-chunk recognizer transcribed 288000 bytes to ''
    # every time; with an energy-driven finaliser the same audio comes back as
    # text.
    # The silence timer counts bytes in the 16kHz mono stream Vosk actually
    # receives, not in the 48kHz stereo chunks Discord sends: one second of
    # silence is 16000 * 2 bytes there, and 48000 * 2 * 2 here. Counting in
    # the wrong domain made the threshold twelve seconds, so a two-second
    # trailing silence never reached it and the phrase never closed.
    SILENCE_BYTES = 16000 * 2  # one second of 16kHz mono 16-bit

    def _on_audio(user, data):
        if not user or not data.pcm:
            return
        # Filter bot's own audio
        if bot_user_id is not None and user.id == bot_user_id:
            return
        uid = user.id
        name = getattr(user, "display_name", None) or str(uid)
        _buffers.setdefault(uid, bytearray()).extend(data.pcm)
        print("[VOICE] audio from " + name + ": " + str(len(data.pcm)) + " bytes (buffer " + str(len(_buffers[uid])) + ")")

        try:
            # Discord sends 48kHz stereo 16-bit PCM. Vosk wants 16kHz mono
            # 16-bit. audioop.tomono turns stereo into mono; audioop.tostereo
            # turns mono into stereo. The two are not the same operation and
            # calling tomono with the intent of making a stereo track halves
            # the audio instead - measured, it left the recognizer with a
            # half-length stream and nothing to transcribe.
            mono = audioop.tomono(data.pcm, 2, 2, 1)
            resampled = audioop.ratecv(mono, 2, 1, 48000, 16000, None)[0]
        except Exception as error:
            print(f"[VOICE] resample error: {error}")
            return

        rec = _recognizers.get(uid)
        if rec is None:
            rec = vosk.KaldiRecognizer(model, 16000)
            _recognizers[uid] = rec

        # A chunk that is all quiet lengthens the silence timer; anything with
        # speech in it resets it. The silence still has to be fed to Vosk -
        # it is what tells the recognizer the phrase is over - so AcceptWaveform
        # runs on every chunk and only the bookkeeping branches.
        rec.AcceptWaveform(resampled)
        silent = _is_silent(resampled)
        if len(_last_rms) >= 400 and len(_last_rms) % 40 == 0:
            print("[VOICE] recent rms min=" + str(round(min(_last_rms))) + " max=" + str(round(max(_last_rms))) + " mean=" + str(round(sum(_last_rms) / len(_last_rms))))
        if silent:
            _silence_bytes[uid] = _silence_bytes.get(uid, 0) + len(resampled)
        else:
            _silence_bytes[uid] = 0
            _last_partial[uid] = True
            return

        # A second of quiet after speech: close the phrase and read it out.
        if not _last_partial.get(uid):
            return
        if _silence_bytes.get(uid, 0) < SILENCE_BYTES:
            return

        _last_partial.pop(uid, None)
        _silence_bytes[uid] = 0
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        if not text:
            return
        _recognizers[uid] = vosk.KaldiRecognizer(model, 16000)
        print("[VOICE] transcribed " + name + ": " + repr(text))
        loop.call_soon_threadsafe(asyncio.ensure_future, on_speech(user, text))

    def _start():
        try:
            channel = getattr(client, "channel", None)
            if channel:
                members = [m.display_name for m in channel.members if not m.bot]
                print("[VOICE] starting listen in channel " + str(channel.name) + " with members: " + str(members))
            else:
                print("[VOICE] starting listen but no channel reference")
            sink = voice_recv.BasicSink(_on_audio)
            client.listen(sink)
            _VOICE_SINKS[guild_id] = sink
            _VOICE_LISTENERS[guild_id] = True
            return True
        except Exception as e:
            print("[VOICE] start error:", e)
            return False

    if is_listening(guild_id):
        current = _VOICE_CLIENTS.get(guild_id)
        if current is not None and getattr(current, "is_connected", lambda: False)():
            print(f"[VOICE] already listening guild={guild_id}")
            return True
        print(f"[VOICE] stale listen flag for guild={guild_id}, clearing")
        _VOICE_LISTENERS.pop(guild_id, None)
        _VOICE_SINKS.pop(guild_id, None)
    if not _start():
        return False

    async def _watchdog():
        # The watchdog only watches for the client going away. It used to
        # restart on is_listening() too, and that is what filled the log: the
        # voice_recv router stops its reader after it has delivered a frame,
        # which is normal, and every stop looked like the listener dying, so
        # listen was restarted every five seconds forever.
        check = 0
        while True:
            check += 1
            await asyncio.sleep(5)
            if not client.is_connected():
                print("[VOICE] watchdog: client disconnected guild=" + str(guild_id))
                _VOICE_LISTENERS.pop(guild_id, None)
                _VOICE_SINKS.pop(guild_id, None)
                break

    asyncio.ensure_future(_watchdog())
    return True


async def voice_stop_listen(guild_id: int):
    """Stop listening in a guild's voice channel."""
    from discord_bot import _VOICE_CLIENTS, _VOICE_SINKS
    client = _VOICE_CLIENTS.get(guild_id)
    if client is None or not hasattr(client, "stop_listening"):
        return
    try:
        client.stop_listening()
    except Exception:
        pass
    _VOICE_LISTENERS.pop(guild_id, None)
    _VOICE_SINKS.pop(guild_id, None)


def is_listening(guild_id: int) -> bool:
    """Whether this guild is currently being listened to."""
    return bool(_VOICE_LISTENERS.get(guild_id))


# ---------------------------------------------------------------------------
# Slash-command definitions
# ---------------------------------------------------------------------------

def register_slash_commands(tree: app_commands.CommandTree):
    """Register all slash commands on the given CommandTree."""

    @tree.command(name="voice", description="Make Kookie join and speak in your voice channel")
    @app_commands.describe(
        text="What she should say (leave empty and she will speak her next reply)",
        emotion="Voice emotion",
    )
    @app_commands.choices(emotion=[
        app_commands.Choice(name="neutral", value="neutral"),
        app_commands.Choice(name="happy", value="happy"),
        app_commands.Choice(name="sad", value="sad"),
        app_commands.Choice(name="excited", value="excited"),
        app_commands.Choice(name="affectionate", value="affectionate"),
        app_commands.Choice(name="angry", value="angry"),
        app_commands.Choice(name="shy", value="shy"),
        app_commands.Choice(name="curious", value="curious"),
    ])
    async def voice_command(interaction: discord.Interaction, text: str = "", emotion: str = "neutral"):
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message(
                "You need to be in a voice channel first!", ephemeral=True
            )
            return
        vc = interaction.user.voice.channel
        guild_id = getattr(getattr(vc, "guild", None), "id", None)
        if guild_id is None:
            await interaction.response.send_message("That channel is not in a guild.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        client, err = await voice_join(vc)
        if client is None:
            await interaction.followup.send(f"Could not join: {err}", ephemeral=True)
            return
        from discord_bot import get_voice_enabled
        get_voice_enabled().add(guild_id)
        print(f"[VOICE CMD] Added guild {guild_id} to _VOICE_ENABLED: {list(get_voice_enabled())}")
        listen_ok = False
        async def on_speech(user, text):
            print(f"Discord bot: heard {getattr(user, 'display_name', user.id)} say: {text}")
            try:
                vc = _VOICE_CLIENTS.get(guild_id)
                if vc is None or not vc.is_connected():
                    return
                bot_user_id = getattr(getattr(vc, "user", None), "id", None)
                if bot_user_id is not None and user.id == bot_user_id:
                    return
                reply = await __import__("asyncio").to_thread(
                    controller.ask_ai,
                    f"{getattr(user, 'display_name', 'someone')} said: \"{text}\". Reply to what they said, briefly and warmly.",
                    voice_mode=True,
                )
                text = (reply or {}).get("reply", "").strip()
                print(f"Discord bot: voice reply text={text!r}")
                if text:
                    vc = _VOICE_CLIENTS.get(guild_id)
                    if vc is None or not vc.is_connected():
                        return
                    ok = await voice_speak(vc.channel or getattr(vc, "channel", None), text)
                    print(f"Discord bot: voice_speak result={ok}")
            except Exception as e:
                print("Discord bot: voice reply error:", e)
        listen_ok = await voice_listen(vc, on_speech)
        if text:
            ok = await voice_speak(vc, text)
            if ok:
                await interaction.followup.send(f"Said: *{text}*" + (" and listening..." if listen_ok else ""), ephemeral=True)
            else:
                await interaction.followup.send("Could not synthesize speech.", ephemeral=True)
        else:
            msg = f"Joined {vc.name} - I will speak there now 💕"
            if listen_ok:
                msg += " Listening for you..."
            await interaction.followup.send(msg, ephemeral=True)

    @tree.command(name="voice-off", description="Make Kookie leave your voice channel")
    async def voice_off_command(interaction: discord.Interaction):
        guild_id = getattr(interaction.guild, "id", None)
        print(f"Discord bot: /voice-off invoked guild_id={guild_id}")
        if guild_id is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return
        from discord_bot import _VOICE_CLIENTS, _VOICE_ENABLED
        client = _VOICE_CLIENTS.get(guild_id)
        print(f"Discord bot: /voice-off client={client} connected={getattr(client, 'is_connected', lambda: False)()}")
        await voice_leave(guild_id)
        await voice_stop_listen(guild_id)
        _VOICE_ENABLED.discard(guild_id)
        try:
            await interaction.response.send_message("Left the voice channel.", ephemeral=True)
        except Exception as e:
            print(f"Discord bot: /voice-off response error: {e}")

    @tree.command(name="listen", description="Make Kookie listen and respond in your voice channel")
    @app_commands.describe(
        listen="Whether to listen (on/off)",
    )
    @app_commands.choices(listen=[
        app_commands.Choice(name="on", value="on"),
        app_commands.Choice(name="off", value="off"),
    ])
    async def listen_command(interaction: discord.Interaction, listen: str = "on"):
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message(
                "You need to be in a voice channel first!", ephemeral=True
            )
            return
        vc = interaction.user.voice.channel
        guild_id = getattr(getattr(vc, "guild", None), "id", None)
        if guild_id is None:
            await interaction.response.send_message("That channel is not in a guild.", ephemeral=True)
            return
        from discord_bot import _VOICE_CLIENTS, _VOICE_ENABLED
        client = _VOICE_CLIENTS.get(guild_id)
        if client is None or not client.is_connected():
            await interaction.response.send_message("I am not in a voice channel. Use /voice first.", ephemeral=True)
            return
        _VOICE_ENABLED.add(guild_id)
        if listen == "on":
            # Stop any listener that is already running - the auto-join path
            # may have started one with a name-filtered callback, and
            # voice_listen would otherwise return True and leave it in place.
            try:
                await voice_stop_listen(guild_id)
            except Exception:
                pass

            async def on_speech(user, text):
                name = getattr(user, "display_name", None) or str(user.id)
                print(f"Discord bot: heard {name} say: {text}")
                try:
                    reply = await __import__("asyncio").to_thread(
                        controller.ask_ai,
                        f"{name} said: \"{text}\". Reply to what they said, briefly and warmly.",
                        voice_mode=True,
                    )
                    text = (reply or {}).get("reply", "").strip()
                    print(f"Discord bot: voice reply text={text!r}")
                    if not text:
                        return
                    ok = await voice_speak(vc, text)
                    print(f"Discord bot: voice_speak result={ok}")
                except Exception as e:
                    print("Discord bot: voice reply error:", e)

            ok = await voice_listen(vc, on_speech)
            if ok:
                await interaction.response.send_message("Listening - I will respond when I hear you speak 💕", ephemeral=True)
            else:
                await interaction.response.send_message("Could not start listening.", ephemeral=True)
        else:
            await voice_stop_listen(guild_id)
            await interaction.response.send_message("Stopped listening.", ephemeral=True)

    @tree.command(name="readchat", description="Make Kookie read chat messages aloud in voice")
    @app_commands.describe(
        action="Enable, disable, or list channels to read",
        channel="Channel to read (defaults to current channel)",
    )
    @app_commands.choices(action=[
        app_commands.Choice(name="on", value="on"),
        app_commands.Choice(name="self", value="self"),
        app_commands.Choice(name="others", value="others"),
        app_commands.Choice(name="off", value="off"),
        app_commands.Choice(name="list", value="list"),
    ])
    async def readchat_command(interaction: discord.Interaction, action: str = "on", channel: discord.TextChannel = None):
        guild_id = getattr(interaction.guild, "id", None)
        if guild_id is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return
        from discord_bot import _VOICE_CLIENTS, _VOICE_ENABLED
        client = _VOICE_CLIENTS.get(guild_id)
        if client is None or not client.is_connected():
            await interaction.response.send_message("I'm not in a voice channel. Use /voice first.", ephemeral=True)
            return
        target_channel = channel or interaction.channel
        if not isinstance(target_channel, discord.TextChannel):
            await interaction.response.send_message("Please select a text channel.", ephemeral=True)
            return
        print(f"[READCHAT CMD] action={action} channel={target_channel.id}")
        if action == "on":
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id, "all")
            await interaction.response.send_message(msg, ephemeral=True)
        elif action == "self":
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id, "self")
            await interaction.response.send_message(msg, ephemeral=True)
        elif action == "others":
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id, "others")
            await interaction.response.send_message(msg, ephemeral=True)
        elif action == "off":
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id)
            if not enabled:
                await interaction.response.send_message(msg, ephemeral=True)
            else:
                await interaction.response.send_message(f"Stopped reading chat in {target_channel.mention}", ephemeral=True)
        elif action == "list":
            channels = chat_reader_get_channels(guild_id)
            if channels:
                names = ", ".join(f"<#{cid}> ({mode})" for cid, mode in channels.items())
                await interaction.response.send_message(f"Reading chat from: {names}", ephemeral=True)
            else:
                await interaction.response.send_message("Not reading any channels right now.", ephemeral=True)

    @tree.command(name="fastprovider", description="Switch fast AI provider (Gemini, Groq, OpenRouter)")
    @app_commands.describe(
        provider="Which provider to use",
    )
    @app_commands.choices(provider=[
        app_commands.Choice(name="Gemini", value="gemini"),
        app_commands.Choice(name="Groq", value="groq"),
        app_commands.Choice(name="OpenRouter", value="openrouter"),
        app_commands.Choice(name="Auto (default order)", value="auto"),
    ])
    async def fastprovider_command(interaction: discord.Interaction, provider: str = "auto"):
        if not controller.is_owner(interaction.user.id):
            await interaction.response.send_message("Not yours.", ephemeral=True)
            return
        import os
        if provider == "auto":
            if "KOOKIE_FAST_PROVIDER" in os.environ:
                del os.environ["KOOKIE_FAST_PROVIDER"]
            await interaction.response.send_message("Fast provider: auto (Gemini → Groq → OpenRouter)", ephemeral=True)
        else:
            os.environ["KOOKIE_FAST_PROVIDER"] = provider
            await interaction.response.send_message(f"Fast provider: {provider.title()} (restart to persist)", ephemeral=True)

# ---------------------------------------------------------------------------
# Text-command execution  (_maybe_switch replacement)
# ---------------------------------------------------------------------------

async def maybe_switch(text: str, message, label: str) -> bool:
    """Act on a text command if this message is one.

    Returns True when it handled the message.
    """
    action, note = parse_text_command(text)
    if action is None and note is None:
        return False

    if not controller.is_owner(message.author.id):
        print(
            f"Discord bot: {label} typed {text!r}, which is a command, but not theirs "
            "- treating it as ordinary conversation"
        )
        return False

    channel = message.channel
    _speak = None

    def _get_speak():
        nonlocal _speak
        if _speak is None:
            from discord_bot import _speak_into as _speak
        return _speak

    if action == "status":
        await _get_speak()(
            channel,
            "I'm on Discord." if controller.discord_bot_running()
            else "I'm not on Discord right now.",
            speak=False,
        )
        return True

    if action == "start":
        answer = await __import__("asyncio").to_thread(controller.discord_switch_command, text)
        await _get_speak()(channel, answer, speak=False)
        return True

    if action == "stop":
        # Prevent self-destruct on accidental triggers - only stop if explicitly requested
        # Check if it's from the actual user or just idle behavior
        twins = controller.signal_other_discord_bots(exclude_pid=os.getpid())
        print(f"Discord bot: asked {len(twins)} other cop(y/ies) to stop, then stopping self")
        try:
            await _get_speak()(
                channel,
                "Okay, getting off Discord now 💕"
                if not twins
                else f"Okay, getting off Discord now 💕 ({len(twins)} other copies too)",
            )
        except discord.HTTPException as error:
            print("Discord bot: could not say goodbye before stopping:", error)
        # Actually do stop, but be more careful
        try:
            os.kill(os.getpid(), __import__("signal").SIGTERM)
        except Exception as e:
            print(f"Discord bot: stop failed: {e}")
        return True

    if action == "voice":
        vc = getattr(message.author, "voice", None)
        if vc is None or vc.channel is None:
            await _get_speak()(
                channel,
                "You need to be in a voice channel first - join one and try again 💕",
                speak=False,
            )
            return True
        guild_id = getattr(getattr(vc.channel, "guild", None), "id", None)
        if guild_id is None:
            await _get_speak()(channel, "That voice channel is not in a guild.", speak=False)
            return True
        client, err = await voice_join(vc.channel)
        if client is None:
            await _get_speak()(channel, f"I could not join the voice channel: {err}", speak=False)
            return True
        from discord_bot import get_voice_enabled, _VOICE_ENABLED as db_ve
        import discord_bot as db
        get_voice_enabled().add(guild_id)
        print(f"[VOICE TEXT CMD] Added guild {guild_id} to _VOICE_ENABLED: {list(get_voice_enabled())}")
        print(f"[VOICE TEXT CMD] discord_bot module id: {id(db)}, _VOICE_ENABLED id: {id(db._VOICE_ENABLED)}, get_voice_enabled() id: {id(get_voice_enabled())}")
        await _get_speak()(
            channel,
            f"Joined {vc.channel.name} - I will speak there now 💕",
            speak=False,
        )
        return True

    if action == "voice-off":
        # /voice-off should work even if the user is not in voice; they just
        # want Kookie out of the guild's channel.
        guild_id = None
        channel = message.channel
        try:
            guild_id = getattr(channel, "guild", None)
            guild_id = getattr(guild_id, "id", None)
        except Exception:
            guild_id = None
        if guild_id is None:
            await _get_speak()(
                channel,
                "That only works in a server.",
                speak=False,
            )
            return True
        await voice_leave(guild_id)
        await voice_stop_listen(guild_id)
        from discord_bot import _VOICE_ENABLED
        _VOICE_ENABLED.discard(guild_id)
        await _get_speak()(channel, "Left the voice channel.", speak=False)
        return True

    if action == "listen":
        vc = getattr(message.author, "voice", None)
        if vc is None or vc.channel is None:
            await _get_speak()(
                channel,
                "You need to be in a voice channel first - join one and try again 💕",
                speak=False,
            )
            return True
        guild_id = getattr(getattr(vc.channel, "guild", None), "id", None)
        if guild_id is None:
            await _get_speak()(channel, "That voice channel is not in a guild.", speak=False)
            return True
        from discord_bot import _VOICE_CLIENTS, _VOICE_ENABLED
        client = _VOICE_CLIENTS.get(guild_id)
        if client is None or not client.is_connected():
            await _get_speak()(channel, "I am not in a voice channel. Use /voice first.", speak=False)
            return True
        _VOICE_ENABLED.add(guild_id)

        # Stop any listener that is already running - for example, the one
        # started by auto-join when she was pinged. voice_listen returns True
        # when a listener is already active, which would leave the old
        # name-filtered callback in place and make /listen a no-op.
        try:
            await voice_stop_listen(guild_id)
        except Exception:
            pass

        async def on_speech(user, text):
            name = getattr(user, "display_name", None) or str(user.id)
            print(f"Discord bot: heard {name} say: {text}")
            try:
                reply = await __import__("asyncio").to_thread(
                    controller.ask_ai,
                    f"{name} said: \"{text}\". Reply to what they said, briefly and warmly.",
                    voice_mode=True,
                )
                text = (reply or {}).get("reply", "").strip()
                print(f"Discord bot: voice reply text={text!r}")
                if text:
                    ok = await voice_speak(vc.channel, text)
                    print(f"Discord bot: voice_speak result={ok}")
            except Exception as e:
                print("Discord bot: voice reply error:", e)

        print(f"Discord bot: starting listen for guild {guild_id}")
        ok = await voice_listen(vc.channel, on_speech)
        print(f"Discord bot: voice_listen result={ok}")
        if ok:
            await _get_speak()(channel, "Listening - I will respond when I hear you speak 💕", speak=False)
        else:
            await _get_speak()(channel, "Could not start listening.", speak=False)
        return True

    if action == "readchat":
        # Usage: /readchat on|off [#channel] - defaults to current channel
        guild_id = getattr(message.guild, "id", None)
        if guild_id is None:
            await _get_speak()(channel, "That only works in a server.", speak=False)
            return True
        from discord_bot import _VOICE_CLIENTS, _VOICE_ENABLED
        client = _VOICE_CLIENTS.get(guild_id)
        if client is None or not client.is_connected():
            await _get_speak()(channel, "I'm not in a voice channel. Use /voice first.", speak=False)
            return True
        # Parse argument: "on", "off", "self", or channel mention
        arg = note.strip().lower() if note else "toggle"
        target_channel = channel
        if arg.startswith("<#") and arg.endswith(">"):
            try:
                target_channel = message.guild.get_channel(int(arg[2:-1]))
                if target_channel is None:
                    await _get_speak()(channel, "Could not find that channel.", speak=False)
                    return True
                arg = "on"
            except ValueError:
                await _get_speak()(channel, "Invalid channel.", speak=False)
                return True
        print(f"[READCHAT TEXT CMD] arg={arg} target_channel={target_channel.id if target_channel else None}")
        if arg in ("on", "enable", "start", "toggle", "all"):
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id, "all")
            await _get_speak()(channel, msg, speak=False)
        elif arg in ("self", "me", "own", "myself"):
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id, "self")
            await _get_speak()(channel, msg, speak=False)
        elif arg in ("others", "other", "people", "users"):
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id, "others")
            await _get_speak()(channel, msg, speak=False)
        elif arg in ("off", "disable", "stop"):
            enabled, msg = chat_reader_toggle(guild_id, target_channel.id)
            if not enabled:
                await _get_speak()(channel, msg, speak=False)
            else:
                await _get_speak()(channel, f"Stopped reading chat in <#{target_channel.id}>", speak=False)
        elif arg in ("list", "status"):
            channels = chat_reader_get_channels(guild_id)
            if channels:
                names = ", ".join(f"<#{cid}> ({mode})" for cid, mode in channels.items())
                await _get_speak()(channel, f"Reading chat from: {names}", speak=False)
            else:
                await _get_speak()(channel, "Not reading any channels right now.", speak=False)
        else:
            await _get_speak()(channel, "Usage: /readchat on|self|off|list [#channel]", speak=False)
        return True

    if action == "fastprovider":
        # Usage: /fastprovider gemini|groq|openrouter|auto
        if not controller.is_owner(message.author.id):
            await _get_speak()(channel, "Not yours.", speak=False)
            return True
        import os
        arg = note.strip().lower() if note else "auto"
        if arg in ("auto", "default", "reset"):
            if "KOOKIE_FAST_PROVIDER" in os.environ:
                del os.environ["KOOKIE_FAST_PROVIDER"]
            await _get_speak()(channel, "Fast provider: auto (Gemini → Groq → OpenRouter)", speak=False)
        elif arg in ("gemini", "groq", "openrouter"):
            os.environ["KOOKIE_FAST_PROVIDER"] = arg
            await _get_speak()(channel, f"Fast provider: {arg.title()} (restart to persist)", speak=False)
        else:
            await _get_speak()(channel, "Usage: /fastprovider gemini|groq|openrouter|auto", speak=False)
        return True

    # help, unknown, or a note-only command
    if note:
        await _get_speak()(channel, note, speak=False)
    return True

# ---------------------------------------------------------------------------
# Chat reading (read text messages aloud in voice channel)
# ---------------------------------------------------------------------------

_CHAT_READER_ENABLED = {}  # guild_id -> {channel_id: mode} where mode is "all" or "self"
_CHAT_READER_TASKS = {}    # guild_id -> asyncio.Task

async def _chat_reader_loop(guild_id, channel_ids, voice_client):
    """Background task that reads messages from text channels and speaks them."""
    import asyncio
    from discord_bot import _VOICE_CLIENTS, _VOICE_ENABLED
    try:
        while True:
            await asyncio.sleep(1)
            # Check if still in voice channel and enabled
            if guild_id not in _VOICE_ENABLED:
                break
            client = _VOICE_CLIENTS.get(guild_id)
            if client is None or not client.is_connected():
                break
            # The actual reading is triggered by _handle_message in discord_bot.py
            # This loop just keeps the task alive for cleanup
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[CHAT READER] Error in guild {guild_id}: {e}")
    finally:
        _CHAT_READER_TASKS.pop(guild_id, None)


def chat_reader_toggle(guild_id: int, channel_id: int, mode: str = "all") -> tuple[bool, str]:
    """Toggle chat reading for a channel in a guild. Returns (enabled, message).
    
    mode: "all" = read everyone's messages, "self" = read only Kookie's own messages, "others" = read only other users' messages
    """
    channels = _CHAT_READER_ENABLED.setdefault(guild_id, {})
    old_mode = channels.get(channel_id)
    print(f"[CHAT READER TOGGLE] guild={guild_id} channel={channel_id} requested_mode={mode} current_mode={old_mode}")
    if channel_id in channels:
        old_mode = channels[channel_id]
        if old_mode == mode:
            # Same mode - toggle off
            channels.pop(channel_id)
            if not channels:
                _CHAT_READER_ENABLED.pop(guild_id, None)
                _CHAT_READER_TASKS.pop(guild_id, None)
            print(f"[CHAT READER TOGGLE] Toggled OFF (was {old_mode})")
            return False, f"Stopped reading chat in <#{channel_id}> (was {old_mode})"
        else:
            # Different mode - switch to new mode
            channels[channel_id] = mode
            print(f"[CHAT READER TOGGLE] Switched from {old_mode} to {mode}")
            if mode == "self":
                return True, f"Switched to reading only my own messages in <#{channel_id}> aloud in voice 💕"
            elif mode == "others":
                return True, f"Switched to reading only other users' messages in <#{channel_id}> aloud in voice 💕"
            else:
                return True, f"Switched to reading all chat from <#{channel_id}> aloud in voice 💕"
    else:
        channels[channel_id] = mode
        print(f"[CHAT READER TOGGLE] Enabled {mode}")
        if mode == "self":
            return True, f"Now reading only my own messages in <#{channel_id}> aloud in voice 💕"
        elif mode == "others":
            return True, f"Now reading only other users' messages in <#{channel_id}> aloud in voice 💕"
        else:
            return True, f"Now reading all chat from <#{channel_id}> aloud in voice 💕"


def chat_reader_is_enabled(guild_id: int, channel_id: int) -> bool:
    """Check if chat reading is enabled for a channel (any mode)."""
    return channel_id in _CHAT_READER_ENABLED.get(guild_id, {})


def chat_reader_get_mode(guild_id: int, channel_id: int) -> str | None:
    """Get the reading mode for a channel: 'all', 'self', or None if disabled."""
    return _CHAT_READER_ENABLED.get(guild_id, {}).get(channel_id)


def chat_reader_get_channels(guild_id: int) -> dict:
    """Get all channels being read for a guild with their modes."""
    return _CHAT_READER_ENABLED.get(guild_id, {}).copy()


async def chat_reader_speak_message(guild_id: int, text: str, author_name: str, author_id: int = None, bot_user_id: int = None, text_channel_id: int = None) -> bool:
    """Speak a message in the guild's voice channel.
    
    If mode is "self", only speaks when the author is the bot itself.
    """
    from discord_bot import _VOICE_CLIENTS, get_voice_enabled
    print(f"[CHAT READER] chat_reader_speak_message called: guild_id={guild_id} text={text[:50]}")
    if guild_id not in get_voice_enabled():
        print(f"[CHAT READER] Guild {guild_id} not in _VOICE_ENABLED: {list(get_voice_enabled())}")
        return False
    client = _VOICE_CLIENTS.get(guild_id)
    if client is None or not client.is_connected():
        print(f"[CHAT READER] No connected voice client for guild {guild_id}: client={client} connected={client.is_connected() if client else 'N/A'}")
        return False
    channel = getattr(client, "channel", None)
    print(f"[CHAT READER] Voice client channel: {channel} (type: {type(channel)})")
    if channel is None:
        print(f"[CHAT READER] Voice client has no channel attribute for guild {guild_id}, trying client.channel")
        # VoiceRecvClient might store channel differently
        channel = getattr(client, "channel", None)
        if channel is None:
            print(f"[CHAT READER] Still no channel, skipping")
            return False
    # Check mode using the text channel ID (not voice channel ID)
    mode = _CHAT_READER_ENABLED.get(guild_id, {}).get(text_channel_id) if text_channel_id else None
    # If no text_channel_id provided (e.g., bot's own reply), skip mode check and always speak
    if text_channel_id is not None and mode is None:
        print(f"[CHAT READER] No mode set for text_channel_id={text_channel_id}")
        return False
    # If self-only mode, only speak our own messages
    if mode == "self" and author_id is not None and bot_user_id is not None:
        if author_id != bot_user_id:
            print(f"[CHAT READER] Self mode but author_id ({author_id}) != bot_user_id ({bot_user_id})")
            return False
    # Prepend author name for "all" and "others" modes
    if mode in ("all", "others"):
        spoken = f"{author_name} says: {text}"
    else:
        spoken = text
    print(f"[CHAT READER] Calling voice_speak for guild {guild_id}")
    result = await voice_speak(channel, spoken)
    print(f"[CHAT READER] voice_speak returned: {result}")
    return result
