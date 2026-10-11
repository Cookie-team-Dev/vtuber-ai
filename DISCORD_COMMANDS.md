# Discord Commands

How to talk to Kookie on Discord.

## Speaking to her

Ping her name, reply to one of her messages, or DM her. She reads every message in
channels she can see; she only replies when she is spoken to.

- `@Kookie hello`
- reply to one of her messages
- DM her directly

If you are in a voice channel when you ping her, she joins it and replies there
by voice as well as text. If you are not in voice, she stays text-only.

## Text commands

Only Mommy / Daddy / the owner can run these. Strangers get an answer instead.

| Command | What it does |
|---------|--------------|
| `/stop discord` | Takes Kookie off Discord. |
| `/stopdiscord` | Same as above, one word. |
| `/discord on` | Brings her back to Discord. |
| `/discord off` | Same as `/stop discord`. |
| `/discord status` | Tells you whether she is online. |
| `/voice` | Makes her join your voice channel. |
| `/kookie voice` | Same as `/voice`. |
| `/listen` | Start listening in your voice channel. She responds when she hears you speak. (Use `/voice` first.) |
| `/kookie listen` | Same as `/listen`. |
| `/readchat on` | Read messages from this text channel aloud in voice. |
| `/readchat self` | Read only my own messages from this channel aloud in voice. |
| `/readchat others` | Read only other users' messages from this channel aloud in voice. |
| `/readchat off` | Stop reading this channel aloud. |
| `/readchat list` | Show which channels are being read aloud. |
| `/readchat on #channel` | Read a specific channel aloud (defaults to current). |
| `/fastprovider gemini|groq|openrouter|auto` | Switch fast AI provider (owner only). |

## Slash commands

Type `/` in the message box and pick one of these.

| Slash command | What it does |
|---------------|--------------|
| `/voice [text] [emotion]` | Join your voice channel and say `text` with `emotion`. Leave `text` empty to join silently; she will speak her next reply there. |
| `/listen [on|off]` | Start or stop listening in your voice channel. When listening, she detects when someone speaks and can respond. |
| `/voice-off` | Leave the voice channel. |
| `/readchat [on|self|others|off|list] [channel]` | Enable/disable reading chat messages aloud in voice. `on` reads everyone, `self` reads only Kookie's messages, `others` reads only other users' messages, `off` stops, `list` shows active channels. |
| `/fastprovider [gemini|groq|openrouter|auto]` | Switch fast AI provider. `auto` uses default order (Gemini → Groq → OpenRouter). |

`emotion` is one of: `neutral`, `happy`, `sad`, `excited`, `affectionate`, `angry`, `shy`, `curious`.

## DM commands

Mommy / Daddy / family can ask Kookie to send a private message to somebody.

| Pattern | Example |
|---------|---------|
| `dm <person> <message>` | `dm kazuma that i'm busy` |
| `dms <person> <message>` | `dms daddy tell him i said hi` |
| `direct message <person> <message>` | `direct message astra hello` |
| `private message <person> <message>` | `private message Owen good luck` |
| `message <person> privately <message>` | `message kazuma privately see you soon` |

She knows people by their Discord display name, username, or relation words:
`kazuma`, `daddy`, `mommy`, `astra`, `owen`, `brother`, etc.

## How voice works

1. You join a voice channel in the server.
2. You ping Kookie in any text channel, or type `/voice`.
3. She joins the voice channel and starts speaking there.
4. Every text reply she sends is also spoken in voice, cleaned up by `voice.py`
   so stage directions and emoji are not read aloud.
5. Type `/listen on` to make her listen for speech in the voice channel.
6. Type `/listen off` to stop listening, or `/voice-off` to leave entirely.
7. Type `/readchat on` to have her read text messages from a channel aloud in voice (like streamers reading chat).
8. Type `/readchat self` to have her read only her own messages aloud in voice.
9. Type `/readchat others` to have her read only other users' messages aloud in voice.

If she is not in voice, replies are text only.

When listening, she detects when someone speaks and logs it. Full speech-to-text
requires an additional transcription engine; without one, she acknowledges speech
events and can still talk back using Piper TTS.

Chat reading uses the same Piper TTS (CPU-only, no GPU needed). She only reads
messages from channels you enable with `/readchat on`, `/readchat self`, or `/readchat others`, and only
while connected to a voice channel in that server. In `self` mode, only her own
messages are spoken; in `others` mode, only other users' messages are spoken; in `on` mode, everyone's messages are spoken.

## Permissions the bot needs

- **Send Messages** - to reply in text channels
- **Send Messages with TTS** - for Discord's built-in text-to-speech
- **Connect** and **Speak** - for voice channel replies
- **Message Content Intent** - to read what people write
- **Direct Messages** intent - for DM commands

## Environment variables

| Variable | Purpose |
|----------|---------|
| `KOOKIE_DISCORD_TOKEN` | Bot token from the Discord developer portal. |
| `KOOKIE_DISCORD_ALLOWED_CHANNELS` | Comma-separated channel IDs. Empty means all. |
| `KOOKIE_DISCORD_ALLOWED_USERS` | Comma-separated user IDs. Empty means all. |
| `KOOKIE_DISCORD_ALLOW_DMS` | `1` to allow DMs, `0` to block them. |
| `KOOKIE_DISCORD_REQUIRE_MENTION` | `1` to require a ping, `0` to answer her name typed out. |
| `KOOKIE_DISCORD_READ_ALL` | `1` to read every message, `0` to only read pings. |
| `KOOKIE_DISCORD_TOOLS` | `1` to enable her tool loop (`browse`, `think`, etc.). |
| `KOOKIE_DISCORD_TTS` | `1` to speak text replies with Discord TTS. |
| `KOOKIE_DISCORD_PROACTIVE` | `1` to let her start conversations on her own. |
| `KOOKIE_DISCORD_PROACTIVE_CHANNEL` | Channel ID for unprompted messages. |
| `KOOKIE_DISCORD_MAX_PENDING` | Queue limit before old messages are dropped. |
| `KOOKIE_DISCORD_NUDGE` | `1` to send a "still thinking" nudge after 60s. |
| `KOOKIE_DISCORD_DOH` | `1` to use DNS-over-HTTPS for Discord hosts. |
| `KOOKIE_DISCORD_DEBUG` | `1` for verbose connection logging. |
| `KOOKIE_FAST_PROVIDER` | Force fast AI provider: `gemini`, `groq`, `openrouter`, or `auto` (default). |
| `KOOKIE_DM_ALLOWED_IDS` | Extra user IDs allowed to use DM commands. |
