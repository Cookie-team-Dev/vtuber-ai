# Discord Commands

How to talk to Kookie on Discord.

## Speaking to her

Ping her name, reply to one of her messages, or DM her. She reads every message in
channels she can see; she only replies when she is spoken to.

- @Kookie hello
- reply to one of her messages
- DM her directly

If you are in a voice channel when you ping her, she joins it and replies there
by voice as well as text. If you are not in voice, she stays text-only.

## Text commands

Only Mommy / Daddy / the owner can run these. Strangers get an answer instead.

| Command | What it does |
|---------|--------------|
| /stop discord | Takes Kookie off Discord. |
| /stopdiscord | Same as above, one word. |
| /discord on | Brings her back to Discord. |
| /discord off | Same as /stop discord. |
| /discord status | Tells you whether she is online. |
| /activity | Tells you what Kookie is doing right now (recon, testing, reporting, etc.). |
| /doing | Same as /activity. |
| /voice | Makes her join your voice channel. |
| /kookie voice | Same as /voice. |

## Slash commands

Type / in the message box and pick one of these.

| Slash command | What it does |
|---------------|--------------|
| /voice [text] [emotion] | Join your voice channel and say text with emotion. Leave text empty to join silently; she will speak her next reply there. |
| /voice-off | Leave the voice channel. |

emotion is one of: neutral, happy, sad, excited, affectionate, angry, shy, curious.

## DM commands

Mommy / Daddy / family can ask Kookie to send a private message to somebody.

| Pattern | Example |
|---------|---------|
| dm <person> <message> | dm kazuma that i'm busy |
| dms <person> <message> | dms daddy tell him i said hi |
| direct message <person> <message> | direct message astra hello |
| private message <person> <message> | private message Owen good luck |
| message <person> privately <message> | message kazuma privately see you soon |

She knows people by their Discord display name, username, or relation words:
kazuma, daddy, mommy, astra, owen, brother, etc.

## How voice works

1. You join a voice channel in the server.
2. You ping Kookie in any text channel, or type /voice.
3. She joins the voice channel and starts speaking there.
4. Every text reply she sends is also spoken in voice, cleaned up by voice.py
so stage directions and emoji are not read aloud.
5. Type /voice-off to make her leave.

If she is not in voice, replies are text only.

## Permissions the bot needs

- **Send Messages** - to reply in text channels
- **Send Messages with TTS** - for Discord's built-in text-to-speech
- **Connect** and **Speak** - for voice channel replies
- **Message Content Intent** - to read what people write
- **Direct Messages** intent - for DM commands

## Environment variables

| Variable | Purpose |
|----------|---------|
| KOOKIE_DISCORD_TOKEN | Bot token from the Discord developer portal. |
| KOOKIE_DISCORD_ALLOWED_CHANNELS | Comma-separated channel IDs. Empty means all. |
| KOOKIE_DISCORD_ALLOWED_USERS | Comma-separated user IDs. Empty means all. |
| KOOKIE_DISCORD_ALLOW_DMS | 1 to allow DMs, 0 to block them. |
| KOOKIE_DISCORD_REQUIRE_MENTION | 1 to require a ping, 0 to answer her name typed out. |
| KOOKIE_DISCORD_READ_ALL | 1 to read every message, 0 to only read pings. |
| KOOKIE_DISCORD_TOOLS | 1 to enable her tool loop (browse, think, etc.). |
| KOOKIE_DISCORD_TTS | 1 to speak text replies with Discord TTS. |
| KOOKIE_DISCORD_PROACTIVE | 1 to let her start conversations on her own. |
| KOOKIE_DISCORD_PROACTIVE_CHANNEL | Channel ID for unprompted messages. |
| KOOKIE_DISCORD_MAX_PENDING | Queue limit before old messages are dropped. |
| KOOKIE_DISCORD_NUDGE | 1 to send a "still thinking" nudge after 60s. |
| KOOKIE_DISCORD_DOH | 1 to use DNS-over-HTTPS for Discord hosts. |
| KOOKIE_DISCORD_DEBUG | 1 for verbose connection logging. |
| KOOKIE_DM_ALLOWED_IDS | Extra user IDs allowed to use DM commands. |