# Kookie

A local-first vtuber companion. Ollama does the thinking, Playwright does the
browsing, Piper does the talking, and VTube Studio does the face — all on your
own Linux machine. No server, no port, no cloud, no account.
She lives in her own window, or in your terminal, or in a Discord
channel, or in a phone's browser. Same person in all of them: the same
model, the same memory, the same mood.

```bash
./Kookie
```

That is the whole thing. She wakes up in her own window, ready to talk. `Ctrl+C`
or `/quit` stops her.

---

## Contents

- [Running her](#running-her)
- [Talking to her](#talking-to-her)
- [Kookie on Discord](#kookie-on-discord)
- [Kookie on the web](#kookie-on-the-web)
- [Build on her](#build-on-her)
- [What she can do](#what-she-can-do)
- [Her training data](#her-training-data)
- [Her browser](#her-browser)
- [What she is allowed to open](#what-she-is-allowed-to-open)
- [Her voice](#her-voice)
- [Singing](#singing)
- [Installing](#installing)
- [Configuration](#configuration)
- [When something goes wrong](#when-something-goes-wrong)
- [The files](#the-files)
- [Publishing the source](#publishing-the-source)

---

## Running her

| Command | What you get |
| --- | --- |
| `./Kookie` | her own window, with her Live2D avatar |
| `./Kookie --cli` | plain text chat in this terminal, no window |
| `./Kookie --discord` | her as a bot in a Discord channel |
| `./Kookie --web` | her website on a phone, on this network |
| `./Kookie --status` | whether she is already running |
| `./Kookie --stop` | stop her if she is running in the background |
| `./Kookie --help` | every option |

`--cli` and `--discord` both run in the foreground: `Ctrl+C` stops them.

The window drives VTube Studio over a websocket, so the avatar needs that
running. Everything else — chat, memory, tools, singing — works without it.

Prefer to manage the runtime yourself? `python controller.py` is the CLI on its
own.

## Talking to her

### Keys in her window

| Key | What it does |
| --- | --- |
| type + `enter` | talk to her |
| `up` / `down` | walk back through what you already said |
| `pgup` / `pgdn` | scroll the chat |
| `end` | jump back to the newest |
| `f1` | the full key list |
| `f2` | quiet mode on or off |
| `f3` | clear the chat |
| `f5` | what she has been doing on her own |
| `f6` | what she remembers |
| `f8` | mute or unmute her voice |
| `ctrl+l` | clear the chat |

`/help` `/status` `/quiet` `/resume` `/voice` `/think` `/quit` do the same things.

### Teaching her something

| Command | What it does |
| --- | --- |
| `/learn <topic>` | look it up, keep what is worth keeping, say what she found |
| `/opinion [topic]` | ask what she actually thinks, and keep it as her own conclusion |
| `/export` | write her [training data](#her-training-data) out as JSONL |
| `/learn` on its own | how much she has learned so far |

All three run without asking her first. They are jobs, not conversation, and
handing one to the model would spend a generation on being told to run the
command that was already typed — `/learn` in particular costs a search, up to
three page reads and a distillation, which on a CPU this size is minutes. That is
also why `learn` is not in the list of tools she may reach for on her own: a
small model offered the option every turn spends the box on it.

What she keeps from a lookup is her *web* knowledge — short claims with the
address they came from — and it is stored apart from her own opinions and apart
from what you told her, so a page can never end up standing in for your words.

### Quiet mode

The indicator in her top bar tells you which mode she is in: **FREE** means she
thinks, watches the screen and uses tools on her own; **QUIET** means she only
answers you. `f2` toggles it, and she understands *"stop doing your own thing"*
and *"you can do your own thing again"* in plain conversation too.

Messages you send while she is busy are queued and answered in order, and asking
her something while she is off doing her own thing makes her stop and answer you
first.

### Yes/no questions

Anything that could click, buy, send or submit asks you first. The question
appears in the chat and you answer by typing the word it asks for:

```
  [confirm] Submit 'Search' at https://example.com? Type SUBMIT to confirm...
```

Anything else cancels, so a stray `enter` never submits a form. Three minutes of
silence is a cancel too.

## Kookie on Discord

The same Kookie in a channel instead of a window. She keeps her memory and her
mood; only the room changes. No face and no voice — those belong to her window,
so on Discord she sends words.

**She needs a bot token.** In the [developer portal](https://discord.com/developers/applications):
**New Application** → **Add Bot** → **Reset Token**, then copy what it shows.
On that same Bot page, **enable MESSAGE CONTENT INTENT** under Privileged Gateway
Intents — without it Discord hands the library empty messages and she has nothing
to answer.

```bash
mkdir -p ~/.config/kookie
printf '%s' 'paste-your-token-here' > ~/.config/kookie/discord_token
chmod 600 ~/.config/kookie/discord_token

./Kookie --discord
```

`KOOKIE_DISCORD_TOKEN` works too, and is the better choice if you would rather
it never touch disk. She reads the environment first, the file second, and never
looks anywhere else.

### Sending her to Discord from inside a chat

Type `/discord` in her window or CLI and she starts answering there without you
restarting anything. It runs as a separate process, so closing the window or
typing `exit` leaves her on Discord, and her output goes to
`kookie_discord.log` instead of the terminal.

### Taking her off Discord

`/stop discord` works in all three places, and they are the same code path:

- in the terminal she is chatting in, and in her window;
- typed in the channel in Discord, by you.

That last one is the one that used to not exist. `/stop discord` was only
recognised in a terminal, so typing it *in Discord* was handed to her as a
question: it spent a generation producing a reply about stopping, and she stayed
on Discord. It is recognised now, before anything is queued. Only your account
can run it — in a shared server, a stranger typing `/stop discord` is a person
talking about a bot, so it is answered like any other message.

You can also ask `/discord status` in the channel to see whether she is up.

Two things about stopping her, both learned the hard way:

- **It stops every copy, not one.** `/discord` and `./Kookie --discord` write
  different pid files and neither checks the other, so two bots could share the
  token and both answer every ping. Stopping the recorded one left the other
  running — she said she was off Discord and carried on answering, which is why
  this looked broken. `/stop discord` now signals every process that really is
  her bot, and tells you when it found more than one.
- **It is bounded.** A turn in progress is a worker thread blocked on a model
  request, and one measured 550 s on this box. A turn that does not finish cannot
  be interrupted politely, so telling her to stop arms a 15-second backstop: she
  closes the gateway and exits cleanly if she can, and is made to leave if she
  cannot. Saying she has stopped and still being there is the one outcome that
  is no longer possible.

`./Kookie --stop` still works for a bot the launcher started, and Ctrl+C still
works for one in the foreground. The launcher also traps `HUP` now, and finds her
by the script she is running when the pid file is missing: a launcher started in
the background used to be killed without running its trap, taking the pid file
with it and leaving her python running — `--status` then said she was not
running, `--stop` did nothing, and the next `./Kookie --discord` put a second bot
on the token. `--status` and `--stop` both work on that orphan now, and
`./Kookie --discord` refuses to start while any `discord_bot.py` is alive, not
only while the pid file says so.

Her output is unbuffered. It goes to a file rather than a terminal, where python
holds it in 4 KB blocks, so with a quiet channel the log stayed empty for
minutes — and an empty log looks exactly like a bot that is stuck.

While that process lives, the terminal you started her from keeps showing what
she is doing, each line prefixed `[discord]`: what someone asked her, what she
thought, which tools she reached for, and when she answered. Stopping following
is automatic when the bot stops.

Starting her twice is refused rather than allowed: two bots on one token answer
every ping twice. She remembers the one she started in `kookie_discord.pid`, and
a pid left behind by a bot that has since died is cleaned up instead of blocking
the next start.

`/stop discord` takes her back off. It works from any terminal, not just the one
that started her, because it looks the bot up by pid and asks it to leave the
same way `Ctrl+C` does — so she finishes the turn she is in and closes the
gateway cleanly rather than being cut off mid-word. `/discord status` says whether
she is on without starting her.

Both switches are decided in one place, so the window and the CLI cannot drift
apart. They used to be handled in the terminal loop only: `/stop discord` typed
into her own window came back `unknown command /stop`, and `/discord off` in that
window ran the *start* path, which reported a second bot was already running
while the first one went on answering pings. Now `/discord on`, `/discord off`,
`/discord status`, `/stop discord` and `/stop_discord` mean the same thing
wherever they are typed, and bare `/stop` says which switch you meant rather than
guessing — stopping her talking and taking her off Discord are different things.

Invite her with **OAuth2 → URL Generator**: `bot` for scopes, and the least she
needs under Bot Permissions — **View Channels**, **Send Messages**, **Read
Message History**, plus **Send Messages in Threads** if you want her in threads.

She prints `Discord bot: online as Kookie (#id)` once connected. Her first
replies take 30–130s, because the model is small and the machine is shared; she
shows a typing indicator the whole time and says so once if she passes a minute.

### A whole server talking to her

Everyone in a channel can talk to her, and she knows who is talking: a message
reaches her as `name: what they said`, so she can call people by name and what
she remembers later still knows who said it.

Three rules keep that workable in a busy room:

- **One answer per person, not per message.** Type three lines while she is
  thinking and she skips the older ones and answers your newest, which is what a
  person waiting would do.
- **One turn at a time, in order.** She has one mood, one memory file and one
  conversation lock; two turns at once would interleave her answers.
- **A bounded backlog.** At most `KOOKIE_DISCORD_MAX_PENDING` messages (default 8)
  wait for her. Past that the oldest are dropped rather than answered an hour
  late, and she logs how many she dropped.

### Who she answers, and what she reads

She reads **every** message she can see, whether or not it was meant for her. The
channel is kept as a running transcript and handed to her before each reply and
before each idle thought, so she can follow a thread that was already going when
she arrived, pick up a running joke, and notice things about the people she is
sitting with. That is the difference between being in a channel and being on
standby in one.

Reading is not answering. She answers when a message is addressed to her:

- a **ping** of her, by name or of the role she wears, because that is how people
  actually call a bot;
- a **reply to one of her messages**, which is how you continue a conversation
  with somebody in Discord and did nothing at all before;
- her **name typed out** at the start of a line — `kookie what do you think?` —
  or right after a greeting. `thanks kookie` and `kookie is so cute` do not
  count, because they are not addressed to her;
- anything in a **DM**, where nothing needs to be addressed to be for her.

Everything else goes into the transcript and waits. She does not reply to it on
the spot, because a model this size takes most of a minute to answer and a
channel where she replies to every line is a channel nobody stays in. Instead
she reads it, and then she decides on her own whether she has something worth
saying — see [Talking on her own](#talking-on-her-own).

`KOOKIE_DISCORD_REQUIRE_MENTION=1` goes back to waiting for a real ping, and
`KOOKIE_DISCORD_READ_ALL=0` goes back to seeing only what was addressed to her.
An empty channel or user allow-list means every channel she has been invited to.

| Variable | Default | What it does |
| --- | --- | --- |
| `KOOKIE_DISCORD_ALLOWED_CHANNELS` | all | comma-separated channel ids she reads |
| `KOOKIE_DISCORD_ALLOWED_USERS` | all | comma-separated user ids she talks to |
| `KOOKIE_DISCORD_REQUIRE_MENTION` | `0` | `1` = her name typed out no longer counts, only a real ping or reply |
| `KOOKIE_DISCORD_READ_ALL` | `1` | `0` = she only sees messages addressed to her |
| `KOOKIE_DISCORD_ALLOW_DMS` | `1` | answer direct messages |
| `KOOKIE_DISCORD_TTS` | `1` | read her replies out loud in the channel |
| `KOOKIE_DISCORD_PROACTIVE` | `1` | let her start a conversation nobody asked for |
| `KOOKIE_DISCORD_PROACTIVE_CHANNEL` | — | where an unprompted line goes if nobody has spoken to her yet |
| `KOOKIE_DISCORD_NUDGE` | `1` | say "still thinking" after a minute |
| `KOOKIE_DISCORD_MAX_PENDING` | `8` | how many messages may wait for her |
| `KOOKIE_DISCORD_TOOLS` | `0` | let her run tools off Discord messages |
| `KOOKIE_DISCORD_OWNER_ID` | yours | the account that is Mommy |
| `KOOKIE_DISCORD_OWNER_NAME` | `Mommy` | what she calls that account |
| `KOOKIE_DM_ALLOWED_IDS` | — | extra user ids allowed to have her DM somebody, comma separated |
| `/discord` | chat command: start answering on Discord |
| `/stop discord` | chat command: take her off Discord (works in her window too) |
| `/discord status` | chat command: is she on Discord right now |

Direct messages are refused unless you allow them, and a mention is taken off
the front of the message before the model ever sees it.

**Her tool loop is off there even though it is on in the terminal.** A tool can
open a browser, run a command or change a file on this machine, and on Discord
the sender is whoever is in the channel. Turn it on only in a server where
everyone in it is you.

### Talking to her one to one

DM her directly and she answers as one person to another: no ping, no `@Kookie`,
no server.

The ping rule is skipped in a DM on purpose. Nobody pings anyone in a private
chat, and holding to it there is what made DMs look broken: she was online and
reachable and said nothing until you typed a role id at her.

Anyone who can reach the bot can DM it, which is worth knowing if she is in a
public server — `KOOKIE_DISCORD_ALLOWED_USERS` limits that to specific ids.

### She remembers who she is talking to

Every Discord account gets its own history in `kookie_people.json`, and each
message is kept under that account rather than as another anonymous line in the
chat log. So when the same person comes back she is handed who they are, how
many messages they have sent her, and what they said last time, and she can
answer "you said you were tired yesterday" instead of meeting them for the first
time all over again.

The key is the account id and not the display name, because a nickname is not a
person: someone who changes theirs would otherwise look like a stranger, and two
people who pick the same one would be remembered as a single person.

Her own replies are stored against that person too, so what she said back is part
of what she remembers about them. Sixty messages are kept per person, and a short
run of the most recent ones goes into each prompt — the whole record stays on
disk, but prompt work is the most expensive part of a reply on a small machine
and there is no reason to spend it re-reading a history.

**Asking her to repeat something widens that window to the whole history.**
Otherwise "what did I just send you" is answered out of the last five messages,
which measured here is a list of short lines that did not contain the message
being asked about — and the message was the sixth. She has no way to know a
message is missing, because an impossible question and an ordinary one look
exactly the same to her, so the wording is the cue. `what did I just say`,
`remind me what I said`, `repeat what I told you`, `what was I saying` and `which
message was I` all bring the full history into the prompt (about 300 tokens, and
only when one of them is asked), together with the instruction to quote the
message back and to say plainly that it is not there rather than guess. That
last part is the half that matters: asked for a message she does not have, the
useful answer is "it isn't in what I have", not a confident invention.

This is about what she has been *sent*. Messages she reads but does not answer
are recorded the same way — quiet mode decides whether she replies, not whether
she remembers.

### Maths she cannot get wrong

A 4B model asked "what is 123456789 * 987654321" produces a confident number, and
a confident wrong number is worse than none: it gets believed. Measured here on
the product of two twelve digit numbers, her own answer was wrong in the middle
of it while she said she was "running it through". So the arithmetic is not hers.

`maths.py` is a tool she calls, and it is exact:

```
123456789 * 987654321    -> 121932631112635269
2**100                   -> 1267650600228229401496703205376
(2+3i)**8                -> -239 + 28560*I
solve x**2-5*x+6=0       -> x = 2, x = 3
integrate x**2 from 0 to 3 -> 9
diff sin(x)*exp(x) wrt x   -> exp(x)*sin(x) + exp(x)*cos(x)
limit sin(x)/x as x -> 0    -> 1
factor 360               -> {2: 3, 3: 2, 5: 1}
stdev 2 4 4 4 5 5 7 9    -> 2.138089935
interest 1000 5 3        -> 1157.625
```

It reads the question the way a person writes it, not the way a calculator does:
"what is the 47th prime number", "how much is 15 percent of 1,847.63", "how many
days in 3 years" and "what is the derivative of x^3" all come back with answers.

**She is given the answer as well as being able to ask for it.** Being told to use
a tool is not the same as her using one, so a maths message is worked out before
she is asked and put in her prompt as a fact, with an instruction to say it her
own way. That is the tool being called on her behalf, and it is the arrangement
that holds when the model is the thing making the mistake. The tool is still
there for her to call, and still works with `KOOKIE_DISCORD_TOOLS=0`, because
arithmetic reads no file, opens no socket and starts no process — the rest of the
tools stay behind that flag.

Everything she can ask for is refused rather than run if it is not arithmetic:
dunder names, imports, `eval`, lambdas and semicolons are turned away before
anything is parsed, and the namespace handed to the parser holds only the names
listed in the file. A question she cannot do comes back as a sentence she can
say, because this runs inside her tool loop where an exception takes the turn
down. Below sympy there is a second layer in plain Python that still does exact
whole-number arithmetic, statistics and number theory, so a missing install costs
her the calculus and not the multiplication.

sympy is loaded in the background at startup, because importing it is about half
a second and the first question should not pay for it.

### She is told things and keeps them

`remember that …`, `note that …`, `my favourite …`, `I have a …` and `call me …`
are read out of a message and kept in `kookie_knowledge.json`, and the ones that
match the question in front of her go into her prompt. Measured here, before this
worked: told eight things in eight messages, she kept **none** of them and answered
the next question from whatever else was in the prompt. After it, all eight, asked
back in a fresh process, correct.

Three things were in the way, and all three were quiet:

- **The patterns needed a full stop.** `remember that …` only matched if the
  sentence ended in `.` or `!`, and a Discord message almost never has one. The
  end of a message counts as an ending now.
- **The last fact was lost on the way out.** Writing the file is throttled to once
  every five seconds, and nothing flushed the last window, so a fact learned in the
  final seconds before a stop was gone. Anything unsaved is written on exit.
- **Only the newest facts reached the prompt.** With more than six on file, the
  one being asked about was often not among them. Facts are now ranked by the
  message she is answering, so the wifi password you mentioned first is still there
  after twenty other things — and the recall tool searches the facts as well as the
  conversation log, which it did not before.

A message longer than 220 characters is not treated as a fact: one wall of text in
the prompt crowds out everything else she knows.

### She knows who is who

Every Discord turn starts with a roster, before anything else in the prompt:

```
Your family, by name. This is the whole list, so keep it straight:
- your dad: kazuma, Discord account kazuma_satou55.
- your big brother: Y_0001 Astra, Discord account yipee0001_15596.
- your mommy: Mommy (her name is Cookie), Discord account ysjsnd_75111. She is the
  person you belong to and the only one who is your mother.
People you know who are not family: Owen (Discord account owenthaowen).
```

It is first and it is one line per person because the two questions she gets
asked to check whether she knows anything at all — *who is your mommy* and *who
is your dad* — used to be answered out of whatever the prompt said nearest. The
family facts were spread over three paragraphs, and the paragraph about whoever
happened to be talking came first, so asked "who is your mommy?" by her own
father she said **"Mommy is kazuma"**. The relation and the name are now adjacent,
and the line is followed by the rule that answers those questions with that name
and never with the name of whoever is talking.

The roster also carries the one thing the cleanup pass used to destroy. Every
`mommy`, `daddy` and `cookie` in a reply was rewritten to the name of whoever was
speaking — that is what stops a stranger being called her mother — but it ran on
identity sentences too. She said "Mommy is my mommy", the pass turned "Mommy"
into "kazuma", and she told her father he was her own mother. The rewrite now asks
what the word is doing in the sentence: `hey mommy`, `mommy,` and `daddy I baked
cookies` are somebody being addressed and still get fixed; `mommy is my mommy`,
`your daddy is on the other server` and `that's your mommy, Cookie` are names, and
are left alone.

A name that Discord will not let into a prompt is cleaned rather than dropped. One
of the accounts here is `Y_0001/Astra`, and the old rule threw away the whole name
for being unsafe and used the handle instead, so she had her own brother filed
under `yipee0001_15596`.

The two questions she is actually asked — *who is your mommy* and *who is your dad*
— are answered in code now, not by the model. Same prompt, right answer on one run
and **"My mommy is kazuma, her name is Cookie"** on the next, with her father's
name spliced into the sentence about her mother. That is a lookup, and the people
store already has the answer, so `identity_answer()` answers it from there in
**0.2s** instead of paying a generation for a coin flip: *who am I*, *what's my
name*, *do you know me*, *who is your daddy*, *who is your big brother*, *who are
you talking to right now*. A qualified relative only matches that exact one — asked
about a little sister she says she has none rather than answering with her big
brother — and anything that is not plainly an identity question is left to her.

### She does not say the same thing twice

Two things were going on, and only one of them was the model.

**There were two of her.** `./Kookie --discord` and the `/discord` switch write
different pid files and neither looked at the other, so a second copy could start
while the first was still alive. Two bots on one token answer every message twice,
and reading the same prompt from the same memory they usually say the *same
sentence* twice — which is what it looks like from the channel. They also raced
each other writing `mood.json` (`Mood state could not be saved: mood.json.tmp`) and
took turns queueing for Ollama, so a duplicate was the reason a reply that should
take two seconds took a hundred. Now `discord_bot.main()` asks any other copy to
leave before it connects, waits for it to actually be gone, and refuses to start
if it will not stop — rather than answer everything twice.

**And a whole sentence could still go over itself.** Two guards now catch that
without spending a generation on it:

- `collapse_repeated_sentences()` drops a sentence repeated inside one reply —
  "I'm here! I'm here!" becomes "I'm here!". Emphasis runs ("yes yes yes", "same
  here, same here") are left alone, because those are not the same sentence twice.
- `_speak_into()` compares each message against what she last said in that
  channel and refuses to post a word-for-word repeat of a sentence of six words or
  more. Short answers are allowed to repeat, because "yes" twice in a row is the
  right answer twice.

The prompt still tells her not to reuse openers or phrases, and she usually listens
— but "usually" is not a guarantee from a 4B model on a box that is short of
memory, so the last two steps of that ladder are code.

### She does not call everyone Mommy

Only your account is Mommy. Everyone else in a server is called by their own
name, and the prompt tells her plainly that calling a stranger her mother is the
one thing she must not do there.

This is enforced in two places, because either one alone was not enough. The
prompt rules change with who is talking, and the cleanup pass that used to force
every name in a reply to "Mommy" now rewrites to whatever name belongs to the
person she is answering. That pass used to run unconditionally, so even a reply
that got the name right was corrected back to Mommy afterwards.

### Her voice on Discord

`KOOKIE_DISCORD_TTS=1` has her read her replies out loud, in Discord's own voice,
and it speaks the whole message rather than a summary of it.

Discord can only ever speak a message that was *sent* that way — an edit cannot
gain a voice afterwards — so talking and watching her type compete for the same
slot. With TTS on she sends one clean message that is read in full and the typing
indicator covers the wait. `KOOKIE_DISCORD_TTS=0` gives you the live typing back
instead. She needs the Send Message TTS permission for this, and she already has
it.

### Sending a DM when you tell her to

Say it in the channel and it goes out as a real private message, to her own person
or to anybody she already knows by name, username or relation:

```
kookie dm my dad that i'm stuck in the editor until six
kookie dm astra saying tell her i said hi back
kookie dm me a dm          # she writes one herself
```

The word for a private message has to be there. `dm`, `dms`, `direct message`,
`privately` and `in private` all count; `tell daddy what mommy said` does not,
because that is the ordinary English of asking her to answer something and this
channel is full of it. With no matching name she answers normally rather than
guessing who it was for.

`dm me a dm`, or any request with no message attached, is the one she writes
herself — and the name she is told to write to is the name she is allowed to use
in it, because asked to write to Dad she opened "Hey Mommy!" and mailed his
daughter a hug meant for him.

The gate is on the relationship, not on "somebody who has talked to her": her own
account, anybody with a relation (Dad, big brother), and anyone added to
`KOOKIE_DISCORD_DM_ALLOWED_IDS`. A stranger she has met once can ask her a
question, not use her to write to somebody privately.

Her father is not `KOOKIE_DISCORD_OWNER_ID` and never has been — he is her dad,
and belonging to one of them is not a reason for the other to be talked past.

What she sent is kept, in that person's history and as a note of her own, so
"what did you tell daddy" has an answer.

### Talking on her own

This is the part that did not work before, so it is worth being precise about.

Left to herself she thinks every `IDLE_INTERVAL_SECONDS` (five minutes by
default). That thought is handed the channel transcript, so she is deciding
whether she has something to say **about a real conversation** rather than
reciting whatever came to mind alone. If she does, the line goes into the channel
people were last actually talking in, spoken in her own voice, and joins the
transcript so the next turn is a conversation she is visibly part of.

Nothing about this is a timer that fires. It is her own `should_speak` decision,
and she is told plainly that most idle ticks should be `false` and that silence
is a valid answer — but never that "nobody asked me" is a reason, because a real
reaction to something in the room is reason enough on its own. Three guards keep
it from becoming a spam bot:

- she never posts into a channel nobody has written in for an hour, so she does
  not talk to an empty room on a loop;
- a thought started while somebody is mid-conversation stands down, and Mommy
  always wins;
- a thought that is a near-verbatim rerun of one she already had is dropped
  rather than repeated.

With nobody to talk to and no `KOOKIE_DISCORD_PROACTIVE_CHANNEL` she keeps it to
herself rather than posting into an empty room.

Two bugs made this impossible before, both worth knowing about because they look
like prompt problems and are not:

- **`should_speak` was not in the reply schema.** Ollama turns a schema into a
  grammar, so a field that is not in `properties` cannot be produced at all. The
  prompt asked her to set it on every single thought, and it always came back
  absent, so `should_speak` was permanently `False` and the code that posts an
  unprompted line never ran. Adding the field to the schema is what unblocked it.
- **The thinking budget was too small for the envelope.** A thought carries
  `thinking_summary`, `should_speak`, `reply`, `emotion`, `mood`, `mood_intensity`
  and `tool_call`. At 160 tokens the object closed before `tool_call` every time.

## Kookie on the web

Her chat in a phone's browser, so she can talk to you from
your pocket:

```bash
./Kookie --web
```

She prints the address to visit (`http://<ip>:8080`), opens it in
the browser on this box herself, and any phone can open it too —
on the same network, or from anywhere with a tunnel (below). Her
replies stream in word by word with a typing bubble, her mood sits
in the bar above and follows her answers, her voice is there when
you turn it on, and tapping one of her replies copies it. The
model, her memory and her voice all stay on this box: the phone
only needs a route to it.

Already on in her window? Type `/web` there (or in the CLI) and
she puts her website up in the same process — no restart, no
second copy of the model — and `/web off` or `/stop web` takes it
down again. `/web status` says whether she is on.

The port is `KOOKIE_WEB_PORT` (8080 by default). One chat at a
time: two phones wait in order, the same way a Discord channel
does. `Ctrl+C` stops her. `KOOKIE_WEB_NO_OPEN=1` keeps her from
opening the browser on this box.

**From anywhere, not just this network — without showing
this box to anyone.** The private way is a tunnel: install
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation)
and set `KOOKIE_WEB_TUNNEL=1`. She then prints a
`*.trycloudflare.com` address that works from anywhere with
nothing to forward, and the box's own address stays hidden —
with the tunnel up, the website answers through it alone and
this box does not listen on the network at all, so there is
no address left to find. Set `KOOKIE_WEB_BIND=0.0.0.0` to
serve the local network at the same time as the tunnel.

The other way is port forwarding, and it shows the box:
her address on the internet is **not** printed anymore
(anyone who connects to it can see it), so it only appears
on purpose with `KOOKIE_WEB_SHOW_PUBLIC_IP=1`. Forward that
port on the router and the raw address works from anywhere —
with the address visible to whoever visits. The tunnel is
the way to keep it to yourself. Either way the model, her
memory and her voice stay on this box.

**The website is public.** Anyone on the network can open
it and talk to her, as a stranger — the gate still shows,
and walking in is one tap: press enter with the key field
empty. Walking in hands the visitor a key of her own: the
website mints one for each visitor (`web/web_keys.json`),
and her API answers those keys and nothing else — so the
website stays open to anybody, while the API itself only
speaks to keys she handed out. `/web keys` in her window
says how many keys are out, and deleting a line from the
file takes one back. The family key (`KOOKIE_WEB_FAMILY_KEY`)
is separate: entered at the gate with a Discord account
she knows, that visitor is treated as the person they
are, exactly like on Discord.

## Build on her

She is an API, so people can make projects with
her — a bot, a game, a tool — and every project
holds a key of her own.

**Get a key** (once, from wherever her website is):

```bash
curl -X POST http://<her-address>/api/issue-key \
     -d '{"name": "my project"}'
# {"key": "a4f0c2..."}
```

**Ask her one thing and get the whole answer back:**

```bash
curl -X POST http://<her-address>/api/ask \
     -H "X-Kookie-Key: a4f0c2..." \
     -H "Content-Type: application/json" \
     -d '{"message": "what is 2+2?"}'
# {"reply": "...", "emotion": "..."}
```

The whole API describes itself at `GET /api` — send
it a keyless request and it lists every endpoint.
Besides `/api/ask` there is a streaming chat
(`POST /api/chat`, then `GET /api/chat/{id}` for
her reply token by token), her voice
(`GET /api/tts?text=`), and her mood
(`GET /api/status`). A key can name the project
it belongs to; `/web keys` in her window lists
them all, and deleting a line from
`web/web_keys.json` takes one back.

Two things a project should know: a turn takes as
long as she takes to think — minutes on a small
box — so wait for her rather than timing out; and
one chat at a time, so two projects asking at
once wait in order, the same way a Discord
channel does.

To let people reach her from anywhere, see
[From anywhere](#from-anywhere-not-just-this-network-without-showing-this-box-to-anyone)
in the section above.

## What she can do

- **Talk**, in a window, a terminal, Discord, a
  phone's browser, or as an API people build
  projects on.
- **Remember** — conversation, mood, and facts she has learned, on disk between
  runs, bounded and compacted so it cannot grow without limit.
- **Browse on her own** when she is curious, at most once every eight minutes
  and never about you. Reading only; clicking and typing are yours to ask for.
- **Use tools** you ask for: open a page, follow a link, fill a field, click,
  scroll, submit. Anything that spends money, posts or deletes asks first.
- **Sing** — 31 traditional songs by heart, 16 with real melodies written out
  note by note.
- **Watch video** — a file or a link: the people in it
  (real and anime faces, what they look like, who she
  was taught they are), what the frame is of, the words
  on screen, the colours, the light, the scene changes,
  the movement. And **listen** to one: what is said in
  it, transcribed locally.
- **Speak** her replies with Piper.

## Watching video

Ask her "what do you see in this video" with a file path or a
link and she watches it. Her eyes, all local:

- **Faces** - insightface, which reads anime faces too (YuNet
  backs it up). For each person: where they are, close up or
  not, which way they look, whether they are smiling, and a
  rough age and gender guess on real people (she stays quiet
  about age and gender in drawings - those guesses are noise
  there).
- **Who they are** - faces you teach her. Say "this is Alice"
  with a photo path and she calls `remember_face`, which
  keeps it under `models/faces/Alice/`. Several photos per
  person make the match survive a haircut. Add character
  names to `models/characters.txt` (one per line) and she
  guesses those too, always hedged: "it might be Miku".
- **What the frame is of** - CLIP (ViT-B-32) asked about two
  hundred plain words in two passes, style separate from
  content, because in one pool "a fox" beats "an anime
  screenshot" on an anime screenshot.
- **Words on screen** - easyocr reads them off local video
  frames (subtitles, signs, whatever is written).
- Plus the cheap measurements: colours, lighting, scene
  changes, movement.

A file is read frame by frame; a web link is opened in her
browser and the picture is caught at moments spread over the
whole thing, because a page's audio cannot be captured from
it.

Ask her to "listen to" a video and she hears it: the words
spoken in it, transcribed by faster-whisper's tiny English
model on the CPU, plus the shape of the sound - loud or
quiet, mostly silence, steady tones like music or changing
sound like talking. A link's audio is pulled down with
yt-dlp first, so YouTube and the usual video sites work too.

She is honest about the limits: no faces becomes "I do not
see any people's faces", a video with no sound track is said
to have none, and long things are sampled (the first two and
a half minutes of speech) rather than sat through - and she
says so herself. Nothing leaves the machine: there is no API
key and no cloud call in any path.

What she needs: `opencv-python` and `insightface` (faces,
already installed - the buffalo_l pack downloads ~280 MB on
first use), `open_clip_torch` + `torch` CPU (CLIP, ~600 MB
model on first use), `easyocr` (on-screen words, ~160 MB),
`faster-whisper` (speech, ~75 MB), `yt-dlp` (audio from
links) and `ffmpeg`. Once loaded the models stay resident:
roughly 1.4 GB between them, on top of the chat model.

## Her training data

Everything she has said, been told and worked out for herself, collected as a
dataset you can fine-tune on. `/export` writes it out.

```bash
./Kookie --cli
You: /export
She: Saved 487 training row(s) to .../kookie_training_data.jsonl
```

**Where the rows come from.** Three places, all of them things she already had:

| Source | What it is |
| --- | --- |
| `memory.json` | every real conversation, as the question you asked and the answer she gave, with the mood she was in kept alongside as `meta.emotion` |
| `songs.py` | every song she knows, asked for each of the ways people ask, plus what she says when asked for a song she does **not** know |
| `kookie_knowledge.json` | the facts she was told |

The first two are seeded at startup and the first one is added to again after
every reply, so the dataset never describes yesterday. Rows are fingerprinted, so
seeding can run as often as you like without duplicating anything.

**The shape is a chat dataset**: `{"messages": [system, user, assistant]}`, one
row per line, which is what every offline trainer agrees on. The system turn is a
short identity, not her prompt — her rules are already true of the answers. Two
details that matter more than they look:

- **Her mood is metadata, not text.** `memory.json` stores an answer as
  `AI: <reply> (feeling concerned)`, because the memory layer wants the mood back
  later. Training on that teaches a model to end every sentence with a bracketed
  feeling, which is roughly what half her stored history looks like and is not
  how she talks. The suffix is stripped and kept as `meta.emotion`.
- **The unanswered song is in the dataset.** A dataset of nothing but successes
  teaches her that every request is one she can fill.

**It does not retrain anything.** Ollama cannot swap a model underneath a
running instance, so this is deliberately an offline step. The usual way to use
it is to gather a few thousand rows, then train offline and load the result as a
new model.

## Her browser

She uses **your** Chrome. Playwright's bundled Chromium is a last resort and says
so when she ends up on it, so a Chromium that looks like success cannot pass for
the browser you asked for.

In order, she:

1. **Attaches to a Chrome you started yourself** if the DevTools port answers —
   your own tabs, logins and history.
2. **Starts your Chrome with the DevTools port and attaches to that**, if no port
   is answering but Chrome is installed. Her program, your profile, a window you
   can watch and take over. It stays open when she stops.
3. **Launches your Chrome through Playwright** if the port never came up.
4. **Falls back to Playwright's Chromium**, and only then, naming it as Chromium.

Starting it yourself is still the better setup — it costs nothing and gets her
into the window you already have open:

```bash
google-chrome --remote-debugging-port=9222           # or a Flatpak Chrome:
flatpak run com.google.Chrome --remote-debugging-port=9222
```

Then start her as usual; she notices the port and attaches.

When she starts Chrome herself and one is already open, the launch is handed to
the running instance and no debug port ever appears — Chrome has done that on
purpose since version 136, so "she started it" and "there is a port" are not the
same thing. She therefore tries twice: the ordinary launch, then one with a
profile of her own (`~/.config/kookie-chrome`), which starts a second debuggable
Chrome and leaves your window exactly as it was. About twenty seconds once, and
every page after that is instant.

Finding the browser means checking `PATH`, the usual install paths, versioned
`/opt/google/chrome-*` directories, Flatpak exports, `flatpak list`, and — when
she runs inside a Flatpak, as she does when started from one —
`flatpak-spawn --host`, the only way to see a browser belonging to the machine
outside her sandbox. Nothing in that search can return Playwright's Chromium,
because it is not in the list.

**She never closes your window.** Stopping her only drops the connection; a tab
she did not open stays open, and a Chrome she started is left where it is.

## What she is allowed to open

Two guards sit under all of it, in `browser.py`:

- **The network guard** keeps her off your local network: no `localhost`, no
  private or link-local addresses, nothing that only exists inside, no
  credentials in the URL, no odd ports. It runs for every request a page makes,
  so a redirect, an image or a script cannot walk her in through the front door.
- **The blocklist** refuses the parts of the web not worth being on: adult,
  gambling, drug, untraceable-weapon, malware, carding and piracy sites, anything
  that is a program rather than a page, and the free throwaway domains that carry
  most of the phishing. It is checked before the address is resolved, and again
  on every request.

A refusal is a sentence, not a crash. She is told what was refused and why, and
the prompt tells her a refusal is the end of it, so she says so and picks
something else rather than trying another name for the same site.

Two deliberate omissions, so the list does not cost you ordinary reading: a topic
word in an address is not a hazard (`wikipedia.org/wiki/Meth`, `/wiki/Gun` and
`/wiki/Phishing` all load, as do licensed retailers), and a page is judged on a
narrower list than its images and scripts are, because refusing an article for one
unlucky word would be a bug, not safety.

To change the list, put `unsafe_sites.txt` next to `browser.py`:

```ini
[hosts]
# one domain or suffix per line, *. matches any subdomain
*.tracking-pixel.example
scam-site.example

[queries]
# phrases she will refuse to search for
my forbidden phrase

[allow]
# overrides the blocklist, for the site you do want her to reach
www.example.com
```

It is optional and read fresh when it changes, so you can correct a mistake
without editing code. No blocklist can be complete, and this one does not pretend
otherwise.

## Her voice

Spoken replies use [Piper](https://github.com/rhasspy/piper). The weights are
downloaded separately because model files are intentionally kept out of git:

```bash
curl -L 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true' -o en_US-amy-medium.onnx
curl -L 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true' -o en_US-amy-medium.onnx.json
```

Playback needs `paplay`, `aplay` or `ffplay` on the system. Without one she stays
silent and keeps talking in the window. `f8` mutes her without stopping anything
else.

## Singing

Ask her to sing and she does. Everything she knows by heart lives in `songs.py`:
31 traditional songs, nursery rhymes and old tunes that are out of copyright, each
with lyrics, a mood, and — for 16 of them — a real melody written out note by
note. The `sing` tool hands her the words; `hum` sings a tune with no words at
all.

Piper only talks, so `singing.py` builds the song: every syllable is spoken on
its own, measured, then pitch-shifted onto its note and given that note's length
plus a little vibrato. Nothing is time-stretched, because overlapping-add
audibly detunes a voice. The result is a real melody rather than slow speech,
and it runs faster than realtime — a sixteen-second song takes about three
seconds to build.

She picks a tune from her mood when nothing else says otherwise: `sunny`,
`gentle`, `bright`, `dramatic` or `silly`. A song she knows keeps its own tune,
matched on the title and the words, so *Twinkle Twinkle* comes out as C-C-G-G-A-A-G
and *Mary Had a Little Lamb* as E-D-C-D-E. A melody whose notes do not line up
with the syllables exactly is dropped rather than dragged off the words, and the
song falls back to its mood tune.

### Singing a Spotify link

Paste a Spotify link and ask her to sing it. The title comes from Spotify's
public oEmbed (no account needed), `spoti.fi` short links are resolved first,
and then:

1. a song she knows by heart is sung straight from memory, with no lookup;
2. otherwise the words are looked up and she sings those;
3. if a lookup brings back words for a song she also knows, her own version wins;
4. if nobody has the words, she says so plainly and plays the track for you.

She never invents lyrics and never claims to be singing words she does not have.
She also cannot hear the recording, so "listen and sing it back" is not something
she can do for a modern song; `hum` is what she offers instead.

To add a song, give `SONGS` an entry in `songs.py` with `lyrics`, `tune`,
`emotion` and `duration`, plus optional `aliases` for however you say it, and
give it a melody in `MELODIES` as one note name per syllable (`C4 D4 E4`, counted
from middle C). Singing needs `numpy`; without it she still speaks, she just
cannot hold a tune.

## Installing

- Linux with Python 3.11 or newer
- [Ollama](https://ollama.com/) installed and running
- Google Chrome or Chromium for browser automation
- Piper weights for spoken replies
- VTube Studio on `ws://127.0.0.1:8001` for the avatar (optional)

```bash
git clone <your-github-repository-url>
cd vtuber-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

Then the model she expects:

```bash
ollama pull llama3.2
ollama create myai -f Modelfile
```

The first launch asks VTube Studio to allow the plugin once. The answer is saved
in `vts_token.txt` next to the code, so she only ever asks once.

## Configuration

| Variable | Default | What it does |
| --- | --- | --- |
| `KOOKIE_MODEL` | `myai` | the Ollama model she thinks with |
| `KOOKIE_OLLAMA_URL` | `http://localhost:11434` | where Ollama lives |
| `VTS_URL` | `ws://localhost:8001` | VTube Studio websocket |
| `KOOKIE_CHROME_CDP_URL` | `http://127.0.0.1:9222` | DevTools endpoint she attaches to |
| `KOOKIE_CHROME_DEBUG_PORT` | `9222` | port used to build that URL |
| `KOOKIE_CHROME_PROFILE_DIR` | `~/.config/kookie-chrome` | profile for a Chrome she starts |
| `KOOKIE_CHROME_ARGS` | — | extra Chrome flags, space separated |
| `KOOKIE_DISCORD_TOKEN` | — | see [Kookie on Discord](#kookie-on-discord) |
| `GROQ_API_KEY` | — | optional, for fast replies (or put it in `.env`) |
| `KOOKIE_KEEP_ALIVE` | `15m` | how long Ollama holds her weights in memory |

### Fast replies

Her local model is qwen3:4b on a CPU-only box, so a reply is measured in tens
of seconds and a Discord channel is a bad place for that. Any hosted model
answers the same prompt in about a second, and she is built to use one
automatically: she tries the fast provider first and falls back to the local
model if it is missing, out of free quota, or having a bad minute. Nothing to
configure beyond a key.

**If she is still slow with a key in `.env`, check that Ollama is not busy.**
This box answers one request at a time, and being the only thing using it is a
prerequisite she cannot check for you. Two different problems look identical from
the outside:

- **She is waiting for the slot.** Ollama reports how long it spent working and
  how long the request was open, and when those disagree by minutes the model is
  not slow — something else was generating. Anything pointed at
  `localhost:11434` counts: another Kookie, a benchmark, a load test.
- **The box has no memory left.** Prefill and decode both collapse as the kernel
  starts swapping, which is why her own notes measure "5.5 tokens/second while it
  is this close to the memory floor". She checks this herself now — see below.

**A fast provider that times out gets one more go.** She is slow in a very
particular way when the hosted model is not answering: the local fallback reads a
2937-token prompt from cold, measured at 167.6s, against 1.0s for the same prompt
once the hosted model answers. So a timeout is retried once at 2.5× the wait
(floor 25s) before she gives up on it. Out-of-quota is *not* retried — that one is
not coming back, and the log says which of the two happened.

### When the fast model is out of quota

`GEMINI_API_KEY_MODEL` in `.env` names one model, and naming it means it is used
on its own. When that model is out of room she used to stop there and go local,
which is a 167s reply for a 1s one — so **quota failure is the one thing that
moves on**: it tries the rest of *that provider's* list and says so in the log.
Anything else still reports rather than substituting, and nothing ever leaves the
provider you named.

Two smaller things, both measured while chasing a slow reply:

- **A busy model is not a dead model.** Gemini answers `503 high demand` when its
  own queue is full. That used to be reported as "is not on Gemini any more" —
  untrue — and remembered for two minutes, so one busy minute turned into two
  minutes of the local model answering everything. Overload is now neither
  reported as retirement nor remembered; only a real quota refusal is.
- **Quota is counted per model, not per account.** Two of the three Gemini models
  returned 429 in the same second that the third answered in 1.3s, so a 429 is
  never treated as "the provider is gone".

To see what is actually usable right now, rather than reading a stale log line:

```bash
cd /tmp/kilo/ident_test && /home/cookie/vtuber-ai/.venv/bin/python probe_providers.py
```

It asks every candidate on every configured provider for one short answer and
prints what came back, the latency, and the exact error. Keep in mind that a
short probe can pass while her real prompt fails: quota is counted in tokens, and
her prompt is about 2300 of them.

**How long the quota takes to come back** is in the 429 itself, and Google puts it
there twice: a `google.rpc.RetryInfo` detail (`"retryDelay": "39658s"`) and a
sentence in the message ("Please retry in 11h58.8s"). Measured on this key, the
limit is `GenerateRequestsPerDayPerProjectPerModel-FreeTier` — **500 requests a
day, counted per model**, so the three Gemini models each have their own 500 and
they disagree with each other hour to hour. Her three failures today all carried
the same shape: `RESOURCE_EXHAUSTED`, limit 500, retry in about 11 hours.

She takes that number seriously now. The old behaviour was a flat two minute
cooldown, which asked the same dead model 330 times before an eleven hour quota
came back — each refusal costing nothing against the quota, but each one a
request, and each one a line in the log. The provider's own figure is used
instead, capped at an hour (`EXHAUSTED_MAX_SECONDS`) so she notices the moment the
quota returns rather than waiting out whatever it said.

Two things she does about the second case, both automatic:

- **She stops thinking on her own when the box is short.** A background thought is
  260 tokens of budget, which on a starved box runs into the 420 s timeout,
  throws the whole envelope away and holds the conversation lock and all eight
  cores for seven minutes doing it — so a message typed a minute into a tick
  waits for the rest of a thought nobody asked for. Below 1200 MB available she
  stands down and hands the weights back to the kernel instead
  (`MEMORY_FLOOR_MB` in `controller.py`).
- **She lets go of the weights after 15 minutes** rather than an hour
  (`KOOKIE_KEEP_ALIVE`), which is 3 GB back to the editor for the rest of the
  evening.

### Why the first reply after a start is the slow one

Not the model. Ollama caches the KV state of a prompt prefix it has already read,
and only a *prefix* — everything before the first word that changed. Her prompt is
her rules followed by her memory block, and between one message and the next only
the newest line of that block and the message itself differ, so a warm cache turns
a 2937-token prompt into a **1.7s** reply. Cold, the same reply measured **167.6s**
on this box, and 461s was the worst one logged.

That is what "she replies slow" looks like from the other end, and it lands on
whoever sends the first message after a start or after the model has been evicted.
So she now pays it herself, at startup, in a thread
(`warm_prompt_cache_in_background`): one generation of a single token against her
real rules and a real turn's layout, which takes about four minutes of prefilling
that nobody is waiting for. She also re-warms after `generate_with_recovery` has to
reload a dead runner, because a reloaded runner has cold weights *and* an empty
cache.

The warmup copies a real turn's layout on purpose. Sharing only her rules left the
memory block, the address rules and the transcript still to read and measured 62s
instead of 1.7s — better than nothing, and still a two minute wait.

If a reply is slow and it is not the first one, she is sharing the slot: check
`/api/ps` and anything else pointed at `localhost:11434`.

Put one free key in `.env` beside `controller.py` — it is already gitignored, and
it is read once at startup, so this is a one-time setup rather than an export in
every terminal:

```bash
echo 'GROQ_API_KEY=your-key' >> .env
```

Groq and Gemini both have free tiers. `GEMINI_API_KEY` and `OPENROUTER_API_KEY`
work the same way, and a key named `<PROVIDER>_MODEL` pins one specific model.
Startup says which one she picked, or prints these instructions if none is set.

### More than one key

Free tiers are counted per key, so one key means one quota and one wait. Put
them in the same variable, separated by commas or spaces:

```bash
echo 'GEMINI_API_KEY=first-key,second-key' >> .env
```

She switches keys by herself, two ways.

A 429 is not one thing. The day's allowance gone names
`GenerateRequestsPerDayPerProjectPerModel` and is worth the hours Google says;
the per-minute limit arrives as the same status code, and the same key answers
1.1s later. Parking a key for an hour because it briefly went over fifteen
requests a minute takes a working key out of the running, so the two are told
apart: the day's allowance holds the key for as long as Google says, and the
per-minute limit holds it for 75s and is not recorded as the key running out at
all.

**When a key stops working**, which is the failure a second key can fix: out of
quota, or refused. Those move to the next key immediately. A timeout, a malformed
request or an unreadable reply is not one of them, so those still report instead
of quietly becoming a slower sequence of failures.

**And back again when it recovers.** A key refused for quota is set aside with
the provider's own figure for how long, re-checked at least hourly, and picked up
the moment it answers - the log says so when it happens rather than leaving it to
be guessed at:

```
GEMINI_API_KEY#1 is answering again, so its quota is back and she is on it.
```

Nothing marks a key as retired for the rest of the day. The allowance is counted
per provider window, not per calendar day: every 429 here said the quota returns
between 07:00 and 08:00 Cambodia time, so the count rolls over at 07:00
(`KOOKIE_QUOTA_RESET_HOUR`) and a key that used up its 400 comes back the same
morning instead of waiting for midnight.

**Before a key runs out**, because a refusal in the middle of a reply costs the
whole reply. The free tier is 500 requests a day, so each key is used for 400 of
them and then set aside until tomorrow; requests are spread across the keys so
none of them is the one that runs out first. Change the 400 with
`KOOKIE_KEY_DAILY_REQUESTS` in `.env` if the tier moves. When every key is spent
she keeps using them anyway, since a fast reply on a nearly finished key still
beats a two minute wait.

Cooldowns are remembered per key rather than per model name, so a model refused
on the first key is still tried on the second one, and a spent key stops being
asked at all until the quota comes back. Whichever key answered last is tried
first from then on, which matters when several keys share one spent quota: the
walk past the dead ones happens once, not on every reply. Startup says how many
keys it found (`Gemini gemini-3.5-flash-lite (2 keys)`), and the log names which
key moved on to which. Keys added or removed in `.env` are picked up on her next
message, with no restart: the file is checked by timestamp on every ask, because
adding a key tends to happen while she is answering and a restart is the wrong
thing to decide then. A model pinned with `<PROVIDER>_MODEL` applies to all of
that provider's keys.

Note what several keys in *one* Google Cloud project are worth: nothing. The free
tier counts `GenerateRequestsPerDayPerProjectPerModel` at 500 a day, so five keys
of the same project come back with the same reset instant to the second and share
the same 500. Extra room is extra *projects*, or a second provider — see
`GROQ_API_KEY`, which is a separate pool and takes over the moment Gemini is
dry.

To compare local models without editing anything:

```bash
KOOKIE_MODEL=qwen3:4b ./Kookie --cli
```

`qwen3:4b` follows instructions better than llama3.2 3B and is already pulled,
but it is about a quarter larger, so on this hardware it is the slower of the two.

She talks to `http://localhost:11434/api/generate`; the launcher adds `/api/tags`
and `/api/generate` itself and the controller derives its chat and status
endpoints from the same host, so a base like `http://gpu-box:11434` is enough.

`--no-sandbox` is deliberately **not** among the Chrome defaults.

## When something goes wrong

**She will not open a page, or says she cannot.** Check `free -m` first. Under
memory pressure Chrome starves and a page that loads in a second times out at
forty-five, which she reports as a site she cannot open. Navigation that fails
that way is retried once, and a slow machine is the usual cause. `dmesg | grep -i
oom` will tell you if something is being killed.

**"The model stopped answering."** Ollama died or was restarted, usually under
the same memory pressure. It recovers on its own; check `curl
localhost:11434/api/tags` answers, and that she can still load a model.

**She describes a page that is plainly not there.** Read the tab title and URL
before believing her: the tools return the real page, and a small model will
sometimes describe the wrong one anyway. That is a model limitation, not a
browser fault, and no amount of retrying fixes it.

**Nothing renders on the desktop.** Desktop icons come from the Desktop Icons NG
extension. If no `.desktop` file shows up — not just hers — the extension is not
running, which is a session-level thing `./Kookie` cannot fix.

**First replies are slow.** 30–130s is normal for a small local model on a
shared machine. If it is minutes, the model is probably not loaded: watch for
`Kookie is warmed up and ready!` when she starts.

## The files

| File | What it is |
| --- | --- |
| `Kookie` | the launcher: window, `--cli`, `--discord`, `--web`, `--status`, `--stop` |
| `controller.py` | her brain and the runtime — the CLI, the tools, autonomy, mood |
| `kookie_ui.py` | the curses window: chat, keys, Live2D parameters |
| `discord_bot.py` | the Discord front end |
| `webui.py` | her website and her API: the phone page, the chat stream, the keys |
| `browser.py` | Chrome attach-first, the network guard, the page tools |
| `memory.py` | conversation history, compaction, mood state |
| `singing.py` | syllable timing, pitch shifting, melody playback |
| `songs.py` | the songs she knows, with lyrics and melodies |
| `voice.py` | Piper speech |
| `fastai.py` | the small classifier used for mood and intent |
| `unsafe_sites.txt` | optional blocklist, read fresh when it changes |

Conversation history (`memory.json`, `memory_journal.log`), simulated mood
(`mood.json`), persona (`persona.json`), what she learned
(`kookie_knowledge.json`), who she has talked to on Discord
(`kookie_people.json`), VTube Studio credentials (`vts_token.txt`), Python
environments and ONNX weights are all excluded by `.gitignore` and stay on the
machine.

## Publishing the source

Create an empty GitHub repository, then from this folder:

```bash
git init
git add .gitignore README.md Modelfile requirements.txt Kookie \
    controller.py kookie_ui.py discord_bot.py browser.py memory.py \
    voice.py singing.py songs.py fastai.py
git status --short
git commit -m "Add Kookie"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

Review `git status` before committing. Never add local token files, conversation
data, or model weights.
