"""Kookie's own window - a native curses interface for her.

No browser, no port, no app: this is her running in your terminal, on your
machine, for you. Start it with ./Kookie.
"""

import codecs
import curses
import re
import sys
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import controller

KOOKIE_TZ = getattr(controller, "KOOKIE_TZ", "Asia/Phnom_Penh")
MAX_LINES = 2000
HISTORY_LIMIT = 60

HELP_TEXT = [
    ("type + enter", "talk to her"),
    ("up / down", "previous messages"),
    ("pgup / pgdn", "scroll the chat"),
    ("end", "jump back to the newest"),
    ("f2", "stop or resume her own thing"),
    ("f3", "clear the chat"),
    ("f5", "what she has been doing on her own"),
    ("f6", "what she remembers"),
    ("f8", "mute or unmute her voice"),
    ("ctrl+l", "clear the chat"),
    ("type + enter", "answer her yes/no question"),
    ("/help", "this list"),
    ("/discord", "let her answer on Discord"),
    ("/stop discord", "take her off Discord"),
    ("/web", "put her on the website for a phone"),
    ("/stop web", "take her off the website"),
    ("/learn <topic>", "look something up and keep what is worth keeping"),
    ("/opinion [topic]", "ask what she actually thinks"),
    ("/export", "write her training data out"),
    ("/quit", "stop Kookie"),
]

SLASH_COMMANDS = {
    "help", "quit", "exit", "clear", "status", "quiet", "resume", "voice", "think",
    "learn", "export", "opinion", "discord", "stop", "web",
}

# Browser actions ask "Type CLICK to confirm"; the word is what she expects back.
CONFIRMATION_PATTERN = re.compile(r"type\s+([A-Za-z]+)\s+to confirm", re.IGNORECASE)
CONFIRMATION_TIMEOUT = 180.0


class _QuietStdout:
    """Swallow Kookie's prints so they cannot corrupt the window."""

    def write(self, _text):
        return 0

    def flush(self):
        return None

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


class _WindowStdout:
    """Route Kookie's prints into her window instead of onto the screen.

    While curses owns the terminal, anything printed to the real stdout would
    scribble over the chat frame, so the text is turned into system lines. The
    reply itself is skipped because the window already draws it as a chat line.
    """

    def __init__(self, sink):
        self._sink = sink

    def write(self, text):
        for line in str(text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("Kookie: "):
                continue
            self._sink(line)
        return len(text or "")

    def flush(self):
        return None

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def _safe(text, fallback="?"):
    """Keep only characters the terminal can actually draw."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode("ascii", "replace").decode("ascii") or fallback


def _char_width(char):
    """Terminal cells one character takes up.

    ncurses writes cells, not codepoints, and her prompt always asks for emoji:
    counting codepoints gave an emoji a width of 1, so every wide or astral
    character pushed the row past its budget and the end of it was clipped away.
    """
    if unicodedata.combining(char):
        return 0
    if unicodedata.east_asian_width(char) in ("W", "F"):
        return 2
    if ord(char) > 0xFFFF:
        # Emoji and the other symbols outside the BMP are drawn double wide.
        return 2
    return 1


def _width(text):
    return sum(_char_width(char) for char in _safe(text, ""))


def _take_cells(text, limit):
    """Split text at the last character that still fits in `limit` cells."""
    used = 0
    for index, char in enumerate(text):
        used += _char_width(char)
        if used > limit:
            # Never hand back an empty head: a caller looping on this would spin
            # forever on a single character wider than the whole budget.
            if index == 0:
                return text[:1], text[1:]
            return text[:index], text[index:]
    return text, ""


def _clip(text, limit):
    text = _safe(text)
    if limit <= 0:
        return ""
    if _width(text) <= limit:
        return text
    # Cut by cells, so the last wide character is never sliced in half.
    head, _rest = _take_cells(text, limit - 1)
    return head + "…"


def _window_to_cursor(text, cursor, cells):
    """Window one long line so the caret stays inside a `cells`-wide field.

    Returns the slice to draw and the part of that slice before the caret, so
    the caller can put the cursor on it without counting anything twice.
    """
    before = text[:cursor]
    total = _width(before)
    if total <= cells:
        return text, before
    # Characters are dropped off the front until what is left of the caret's
    # side of the line fits, so the caret and the text around it stay on screen
    # and the tail of what you typed is not the part that disappears.
    dropped = 0
    for index, char in enumerate(before):
        if total - dropped <= cells:
            return text[index:], before[index:]
        dropped += _char_width(char)
    return "", ""


class KookieWindow:
    def __init__(self, screen):
        self.screen = screen
        self.lines = []
        self.input = ""
        self.cursor = 0
        self.history = []
        self.history_offset = None
        self.scroll = 0
        self.follow = True
        self.status_note = ""
        self.status_note_until = 0.0
        self.busy = False
        self.busy_label = ""
        # The queue and the flag that says who is draining it are one decision:
        # split apart, a message could be accepted in the gap between the drain
        # loop finding the queue empty and clearing the flag, and then nothing
        # would ever come back for it.
        self.pending = deque()
        self.pending_lock = threading.Lock()
        self.draining = False
        self.stopping = False
        self.muted = False
        self.lock = threading.Lock()
        self.dots = 0
        self.last_draw = 0.0
        self._status_cache = {}
        self._status_cache_time = 0.0
        # getch() hands over one byte at a time, so this is where a character
        # that needs two or four of them is assembled.
        self._utf8 = codecs.getincrementaldecoder("utf-8")()
        # Wrapped-chat cache. _chat_version is the single thing that says the
        # chat changed; _wrap_per_line then keeps the rows of the lines that did
        # not change, so one new reply only wraps one line.
        self._chat_version = 0
        self._wrap_width = None
        self._wrap_version = -1
        self._wrap_src = []
        self._wrap_per_line = []
        self._wrap_flat = None
        self.confirm_prompt = ""
        self.confirm_word = ""
        self.confirm_answer = ""
        self.confirm_lock = threading.Lock()
        self.confirm_event = threading.Event()
        self.setup_colors()

    # ------------------------------------------------------- confirmations

    def ask_confirmation(self, message, timeout=CONFIRMATION_TIMEOUT):
        """Ask a yes/no question in the chat and wait for the answer here.

        The browser tools run on the agent thread while this window owns the
        terminal, so the answer travels back through an event instead of stdin.
        Anything that is not the exact word cancels, and the wait is bounded so
        a forgotten prompt can never wedge the agent forever.
        """
        match = CONFIRMATION_PATTERN.search(str(message or ""))
        word = match.group(1) if match else ""
        with self.confirm_lock:
            self.confirm_event.clear()
            self.confirm_prompt = str(message)
            self.confirm_word = word
            self.confirm_answer = ""
        self.system(f"[confirm] {message}")
        deadline = time.time() + timeout
        while not self.confirm_event.wait(0.2):
            if self.stopping or time.time() > deadline:
                break
        with self.confirm_lock:
            answer = self.confirm_answer
            self.confirm_prompt = ""
            self.confirm_word = ""
            self.confirm_answer = ""
        if not self.confirm_event.is_set():
            self.system("[confirm] no answer in time, so nothing happened")
        return answer

    def _resolve_confirmation(self, message):
        """Hand the typed line to the waiting agent thread."""
        with self.confirm_lock:
            expected = self.confirm_word
            self.confirm_answer = message
        self.confirm_event.set()
        self.system(f"[confirm] {message}" + ("" if message == expected else " (cancelled)"))
        return True

    # ---------------------------------------------------------------- colours

    def setup_colors(self):
        curses.start_color()
        curses.use_default_colors()
        try:
            curses.curs_set(1)
        except curses.error:
            pass

        def pair(index, fg, bg=-1):
            try:
                curses.init_pair(index, fg, bg)
                return curses.color_pair(index)
            except curses.error:
                return curses.A_NORMAL

        self.c_title = pair(1, curses.COLOR_GREEN) | curses.A_BOLD
        self.c_dim = pair(2, curses.COLOR_CYAN)
        self.c_kookie = pair(3, curses.COLOR_GREEN) | curses.A_BOLD
        self.c_you = pair(4, curses.COLOR_YELLOW) | curses.A_BOLD
        self.c_accent = pair(5, curses.COLOR_YELLOW) | curses.A_BOLD
        self.c_bad = pair(6, curses.COLOR_RED) | curses.A_BOLD
        self.c_clock = pair(7, curses.COLOR_MAGENTA) | curses.A_BOLD
        self.c_rule = pair(8, curses.COLOR_GREEN)
        self.c_meta = pair(9, curses.COLOR_BLUE)
        self.c_help = pair(10, curses.COLOR_WHITE)
        self.c_think = pair(11, curses.COLOR_CYAN) | curses.A_BOLD

    # ---------------------------------------------------------------- messages

    def note(self, text, seconds=4.0):
        self.status_note = text
        self.status_note_until = time.time() + seconds

    def add_line(self, role, text, emotion=""):
        stamp = datetime.now(ZoneInfo(KOOKIE_TZ)).strftime("%H:%M")
        with self.lock:
            self.lines.append((role, str(text), stamp, emotion))
            if len(self.lines) > MAX_LINES:
                del self.lines[: len(self.lines) - MAX_LINES]
            self._chat_version += 1
        if self.follow:
            self.scroll = 0

    def clear_chat(self):
        """Drop the whole transcript. Used by ctrl+l, f3 and /clear."""
        with self.lock:
            self.lines = []
            self._chat_version += 1
        self.scroll = 0

    def system(self, text):
        self.add_line("system", text)

    # ---------------------------------------------------------------- drawing

    def wrapped(self):
        """Every chat line, wrapped to the current width."""
        height, screen_width = self.screen.getmaxyx()
        width = max(20, screen_width - 4)

        # Wrapping the whole chat again on every redraw was by far the most
        # expensive thing this window did, and it grew without bound: at the
        # 2000-line cap it rebuilt 2000 wrapped strings about eight times a
        # second. The wrapped chat only depends on the chat and the width, so
        # it is cached against those. The "thinking" row is deliberately not
        # part of that key, so its ticking dots do not throw the cache away.
        if self._wrap_width != width:
            # A resize re-wraps every line, so nothing cached survives it: the
            # per-line rows and the flattened list are both dropped, and the
            # version is forced stale so the next call rebuilds.
            self._wrap_width = width
            self._wrap_src = []
            self._wrap_per_line = []
            self._wrap_flat = None
            self._wrap_version = -1
        if self._wrap_flat is None or self._wrap_version != self._chat_version:
            self._wrap_flat = self._wrapped_chat(width)
            self._wrap_version = self._chat_version
        rows = self._wrap_flat
        if self.busy:
            # A copy, because the live thinking row is appended below.
            rows = list(rows)
            label = self.busy_label or "thinking"
            ticks = "." * (1 + self.dots % 3)
            rows.append((" kookie  ····  ", f"{label}{ticks}", self.c_think))
        visible = max(1, height - 7)
        if self.follow:
            self.scroll = max(0, len(rows) - visible)
        elif self.scroll > len(rows) - visible:
            self.scroll = max(0, len(rows) - visible)
        return rows[self.scroll: self.scroll + visible]

    def _wrapped_chat(self, width):
        """Wrap every stored chat line, reusing rows for lines already wrapped.

        Only newly added lines are actually wrapped, so a reply arriving at the
        2000-line cap costs one line of work instead of re-wrapping the whole
        transcript.
        """
        # A reply can land on the list while this runs, so the length is taken
        # from a copy: iterating the live one could run off its end, and the
        # MAX_LINES trim could shift it under the loop.
        with self.lock:
            lines = list(self.lines)
        src = self._wrap_src
        per_line = self._wrap_per_line
        if len(lines) < len(src):
            # Lines were dropped off the front (the MAX_LINES trim) or cleared.
            del src[len(lines):]
            del per_line[len(lines):]
        elif len(lines) > len(src):
            src.extend([None] * (len(lines) - len(src)))
            per_line.extend([None] * (len(lines) - len(per_line)))

        for index, item in enumerate(lines):
            # The row cache is sized from that same length, so it is grown here
            # too. Indexing it blindly raised IndexError out of draw() whenever
            # another thread appended a line in between, which took the window
            # down with it.
            while len(per_line) <= index:
                src.append(None)
                per_line.append(None)
            if per_line[index] is None or src[index] != item:
                src[index] = item
                per_line[index] = self._wrap_one(item, width)
        self._wrap_src = src
        self._wrap_per_line = per_line
        # Always rebuilt from the per-line rows rather than reused, so clearing
        # the chat can never leave the old transcript on screen.
        rows = []
        for chunks in per_line:
            rows.extend(chunks)
        return rows

    def _wrap_one(self, item, width):
        """Wrap one stored chat line into (prefix, chunk, style) rows."""
        role, text, stamp, emotion = item
        if role == "system":
            prefix, style = "  · ", self.c_dim
        elif role == "user":
            prefix, style = f" you  {stamp}  ", self.c_you
        else:
            prefix, style = f" kookie {stamp} ", self.c_kookie
        body = text
        if emotion and emotion != "neutral":
            body = f"{text}  ({emotion})"
        chunks = self._wrap(body, width - _width(prefix))
        padding = " " * _width(prefix)
        return [
            (prefix if index == 0 else padding, chunk, style)
            for index, chunk in enumerate(chunks)
        ]

    @staticmethod
    def _wrap(text, width):
        width = max(8, width)
        out = []
        line = ""
        line_len = 0
        # The running width in cells is tracked instead of rebuilding and
        # re-stripping "line + word" once per word: this is the innermost loop
        # of every redraw, and the discarded candidate string was most of it.
        for word in str(text).split():
            word_len = _width(word)
            if line_len:
                if line_len + 1 + word_len <= width:
                    line = f"{line} {word}"
                    line_len += 1 + word_len
                    continue
                out.append(line)
                line, line_len = "", 0
            while word_len > width:
                head, word = _take_cells(word, width)
                out.append(head)
                word_len = _width(word)
            line, line_len = word, word_len
        out.append(line)
        return out or [""]

    def draw(self):
        height, width = self.screen.getmaxyx()
        now = datetime.now(ZoneInfo(KOOKIE_TZ))

        now_time = time.time()
        if now_time - self._status_cache_time > 0.2:
            mood = controller.get_mood_state()
            autonomy = controller.autonomy_state()
            self._status_cache = {
                "mood": mood,
                "autonomy": autonomy,
                "vts": controller.vts is not None,
            }
            self._status_cache_time = now_time
        mood = self._status_cache.get("mood", {})
        autonomy = self._status_cache.get("autonomy", {})
        mood_text = f"mood {mood.get('mood', 'neutral')} {int(mood.get('intensity', 0) * 100)}%"
        mode = "FREE" if autonomy.get("enabled") else "QUIET"
        mode_style = self.c_kookie if autonomy.get("enabled") else self.c_accent
        avatar = "connected" if self._status_cache.get("vts") else "no avatar"

        # ---- everything that goes on screen is computed first
        brand = " Kookie "
        clock = now.strftime("%a %d %b · %H:%M Cambodia ")
        middle = (
            f"{mood_text}   avatar {avatar}   she is {mode}   "
            f"{'voice on' if not self.muted else 'voice muted'} "
        )
        bar_width = max(0, width - len(middle) - len(brand) - len(clock) - 2)
        rows = self.wrapped()
        label = "confirm" if self.confirm_prompt else ("send" if self.input.strip() else "talk to her")
        text = self.input.replace("\t", " ")
        # The composer is a single line, so a long message has to scroll: what
        # is visible is windowed around the caret, which used to be clipped at
        # the field width while the cursor column kept counting on past it.
        field = max(0, width - 6)
        visible, before = _window_to_cursor(text, self.cursor, field)
        cursor_x = 4 + _width(before)
        hint = "f1 help · f2 quiet · f3 clear · ctrl+l clear"
        note = self.status_note if time.time() < self.status_note_until else ""
        status_line = f" {note or self.default_status(autonomy)}"

        # This used to skip the repaint whenever the frame matched the last one,
        # which is nearly every wake of this loop. Nothing can tell this process
        # that something else wrote to the terminal - the launcher probes Ollama
        # in the background with the real stdout still attached - so an
        # unchanged frame is exactly when the scribbled text is left sitting
        # there forever. Every frame is erased and drawn again; the expensive
        # part, wrapping the chat, is still cached above.
        self.screen.erase()

        # ---- top bar
        self._put(0, 0, brand, self.c_title, width, height)
        self._put(0, max(len(brand), width - len(clock) - 1), clock, self.c_clock, width, height)
        self._put(0, len(brand), middle, mode_style, width, height, limit=bar_width)
        self._rule(1, width, height)

        # ---- chat
        for index, (prefix, chunk, style) in enumerate(rows):
            self._put(2 + index, 2, prefix + chunk, style, width, height)

        # ---- composer
        self._rule(height - 5, width, height)
        self._put(height - 4, 2, "› ", self.c_accent, width, height)
        self._put(height - 4, 4, visible, curses.A_REVERSE if self.busy else curses.A_NORMAL, width - 6, height)
        # The label sits right of the text so it never shifts the cursor column.
        self._put(height - 4, min(6 + _width(visible), width - len(label) - 1), label, self.c_dim, width, height)
        try:
            self.screen.move(height - 4, cursor_x)
        except curses.error:
            pass

        self._put(height - 3, max(2, width - len(hint) - 2), hint, self.c_dim, width, height)

        # ---- status line
        self._put(height - 2, 0, status_line, self.c_dim if not note else self.c_accent, width, height)
        self.screen.noutrefresh()
        curses.doupdate()

    def _put(self, row, col, text, style, width, height, limit=None):
        if row < 0 or row >= height:
            return
        # _clip already sanitises, so _safe is not run a second time here.
        text = _clip(text, limit if limit is not None else max(0, width - col - 1))
        if not text:
            return
        try:
            self.screen.addstr(row, col, text, style)
        except curses.error:
            pass

    def _rule(self, row, width, height):
        if 0 <= row < height:
            try:
                # ACS_HLINE is a single terminal cell, unlike a box-drawing glyph.
                self.screen.hline(row, 0, curses.ACS_HLINE, max(0, width - 1), self.c_rule)
            except curses.error:
                pass

    # ---------------------------------------------------------------- status

    def mood_text(self):
        try:
            state = controller.get_mood_state()
            return f"mood {state.get('mood', 'neutral')} {int(state.get('intensity', 0) * 100)}%"
        except Exception:
            return "mood unknown"

    def default_status(self, autonomy):
        if self.busy:
            return f"she is {self.busy_label or 'thinking'}…"
        if not autonomy.get("enabled"):
            until = autonomy.get("resume_at") or 0
            if until:
                minutes = max(1, round((until - time.time()) / 60))
                return f"quiet mode on - she stays quiet for another {minutes} min"
            return "quiet mode on - she only answers when you speak to her"
        return "she is free to think, watch and use tools on her own"

    # ---------------------------------------------------------------- replies

    def reply_to(self, message):
        """Queue one exchange with her and answer it without freezing the window."""
        # Now, not when the worker thread picks this up: until the turn has the
        # conversation lock she has no other way of knowing Mommy is waiting.
        controller.note_user_input()
        with self.pending_lock:
            self.pending.append(message)
            if self.draining:
                waiting = len(self.pending)
                self.note(f"{waiting} message{'' if waiting == 1 else 's'} waiting")
                return
            self.draining = True
        self.busy = True
        self.busy_label = "thinking"
        threading.Thread(target=self._drain_queue, daemon=True).start()

    def _drain_queue(self):
        try:
            while not self.stopping:
                with self.pending_lock:
                    if not self.pending:
                        # Clearing the flag in the same breath as the last pop is
                        # what makes this safe: a message arriving now either
                        # lands in a queue this loop has not given up on, or sees
                        # the flag down and starts a drain of its own.
                        self.draining = False
                        break
                    message = self.pending.popleft()
                self._turn(message)
        finally:
            with self.pending_lock:
                self.draining = False
            self.busy = False
            self.busy_label = ""

    def _turn(self, message):
        """Run one exchange, holding the conversation so her own thinking waits."""
        acquired = False
        try:
            # She is told to stop, and if she really will not let go the message
            # waits here instead of being thrown away. It used to be dropped with
            # "try again in a moment", which lost whatever had been typed while a
            # tool call or a long thought held the lock.
            announced = False
            while not controller.acquire_user_turn():
                if not announced:
                    self.system("she has not let go yet - holding your message")
                    announced = True
                if self.stopping:
                    return
                time.sleep(1.0)
            acquired = True
            result = controller.ask_ai(message) or {}
            reply = controller.clean_reply(controller.condense_reply(result.get("reply", "")))
            result["reply"] = reply
            if reply:
                self.busy_label = "speaking"
                self.add_line("kookie", reply, result.get("emotion", ""))
                controller.deliver_reply(result, play_voice=not self.muted)
            else:
                self.system("she had nothing to say")
        except Exception as error:
            self.system(f"something went wrong: {error}")
        finally:
            if acquired:
                controller.release_user_turn()
            controller.schedule_idle_check()

    # ---------------------------------------------------------------- input

    def handle_key(self, key):
        if key in (curses.KEY_UP,):
            self.recall(1)
        elif key in (curses.KEY_DOWN,):
            self.recall(-1)
        elif key in (curses.KEY_PPAGE,):
            self.follow = False
            self.scroll = max(0, self.scroll - 5)
        elif key in (curses.KEY_NPAGE,):
            self.follow = False
            self.scroll += 5
        elif key == curses.KEY_HOME:
            self.cursor = 0
        elif key == curses.KEY_END:
            self.cursor = len(self.input)
            self.follow = True
        elif key == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor = min(len(self.input), self.cursor + 1)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if self.cursor:
                self.input = self.input[: self.cursor - 1] + self.input[self.cursor:]
                self.cursor -= 1
        elif key == curses.KEY_DC:
            if self.cursor < len(self.input):
                self.input = self.input[: self.cursor] + self.input[self.cursor + 1:]
        elif key == 21:  # ctrl+u
            self.input, self.cursor = "", 0
        elif key == 12:  # ctrl+l
            self.clear_chat()
        elif key == curses.KEY_F1:
            self.show_help()
        elif key == curses.KEY_F2:
            self.toggle_quiet()
        elif key == curses.KEY_F3:
            self.clear_chat()
            self.note("chat cleared")
        elif key == curses.KEY_F5:
            self.show_activity()
        elif key == curses.KEY_F6:
            self.show_memory()
        elif key == curses.KEY_F8:
            self.muted = not self.muted
            self.note("voice muted" if self.muted else "voice on")
        elif key == 27:  # escape clears the line
            self.input, self.cursor = "", 0
        elif key in (curses.KEY_ENTER, 10, 13):
            self.submit()
        elif 32 <= key <= 126:
            self.insert(chr(key))
        elif 128 <= key <= 255:
            self._insert_byte(key)

    @staticmethod
    def _is_text_key(key):
        """True for a key that means the user is typing a character."""
        return 32 <= key <= 126 or 128 <= key <= 255

    def insert(self, text):
        """Put text into the composer where the caret is."""
        self.input = self.input[: self.cursor] + text + self.input[self.cursor:]
        self.cursor += len(text)

    def _insert_byte(self, byte):
        """Turn the bytes curses hands over above ASCII into real characters.

        In a UTF-8 terminal getch() delivers one byte per call, so an accent, a
        curly quote or an emoji arrived as two or four keypresses and every one
        of them used to be dropped. The incremental decoder holds the partial
        character until the rest of it arrives, and gives up on a byte that
        cannot start one, so a terminal that is not speaking UTF-8 does not
        swallow the keys typed after it.
        """
        try:
            decoded = self._utf8.decode(bytes((byte,)))
        except UnicodeDecodeError:
            self._utf8.reset()
            return
        if decoded:
            self.insert(decoded)

    def recall(self, direction):
        """Walk back and forward through what you already said."""
        if not self.history:
            return
        if self.history_offset is None:
            # The first Up press lands on the most recent message, not past it.
            self.history_offset = max(0, direction)
        else:
            self.history_offset = max(0, min(len(self.history), self.history_offset + direction))
        if self.history_offset == 0:
            self.input, self.cursor = "", 0
        else:
            self.input = self.history[-self.history_offset]
            self.cursor = len(self.input)

    def submit(self):
        message = self.input.strip()
        self.input, self.cursor = "", 0
        if not message:
            return
        if self.confirm_prompt:
            # A browser action is waiting on this line, so it is the answer.
            self._resolve_confirmation(message)
            return
        if message in self.history:
            self.history.remove(message)
        self.history.append(message)
        del self.history[:-HISTORY_LIMIT]
        self.history_offset = None

        if message.startswith("/"):
            self.slash(message)
            return
        self.add_line("user", message)
        self.reply_to(message)

    def slash(self, message):
        parts = message[1:].split()
        if not parts:
            self.system("type a command, like /help")
            return
        command = parts[0].lower()
        if command in ("quit", "exit"):
            self.stopping = True
        elif command == "help":
            self.show_help()
        elif command == "clear":
            self.clear_chat()
            self.note("chat cleared")
        elif command == "status":
            self.system(f"model {controller.MODEL} · {self.mood_text()} · "
                        f"{'avatar connected' if controller.vts else 'no avatar'} · "
                        f"web {'on' if controller.web_server_running() else 'off'} · "
                        f"voice {'muted' if self.muted else 'on'}")
        elif command == "quiet":
            minutes = None
            for part in parts[1:]:
                if part.replace(".", "").isdigit():
                    minutes = float(part)
            state = controller.set_autonomy(False, minutes=minutes, reason="/quiet")
            self.system("quiet mode on" + (f" for {int(minutes)} min" if minutes else ""))
            if not state:
                self.system("could not change her mode")
        elif command == "resume":
            controller.set_autonomy(True, reason="/resume")
            self.system("she is free again")
        elif command == "voice":
            self.muted = not self.muted
            self.system("voice muted" if self.muted else "voice on")
        elif command == "think":
            self.show_activity()
        elif command == "discord":
            # Every /discord variant is decided in one place, so the window and
            # the terminal cannot disagree about what "off" means. It used to be
            # decided here as "start", which is how `/discord off` in this window
            # started a second bot and reported a conflict while the first one
            # went on answering pings.
            self.system(controller.discord_switch_command(message))
        elif command == "web":
            # Same rule as /discord: one place decides what
            # "off" means, so this window and the terminal
            # cannot disagree about it.
            self.system(controller.web_switch_command(message))
        elif command == "stop":
            what = " ".join(parts[1:]).strip().lower()
            if not what:
                self.system("stop what? /stop discord takes me off Discord,"
                            " /stop web takes her off the website")
                return
            if what == "web":
                self.system(controller.web_switch_command(message))
                return
            self.system(controller.discord_switch_command(message))
        elif command == "learn":
            topic = " ".join(parts[1:]).strip()
            if not topic:
                self.system(controller.learning.stats_line())
            else:
                self.system(controller.learn_about(topic))
        elif command == "export":
            self.system(controller.export_training_data())
        elif command == "opinion":
            topic = " ".join(parts[1:]).strip()
            self.system(controller.decide_for_herself(topic))
        else:
            self.system(f"unknown command /{command} - try /help")

    def toggle_quiet(self):
        if controller.is_autonomous_allowed():
            controller.set_autonomy(False, reason="f2")
            self.system("quiet mode on - she will only answer you now")
        else:
            controller.set_autonomy(True, reason="f2")
            self.system("she is free again")

    # ---------------------------------------------------------------- extras

    def show_help(self):
        self.system("keys")
        for key, meaning in HELP_TEXT:
            self.system(f"    {key:<12} {meaning}")

    def show_activity(self):
        history = controller.get_autonomous_history()
        if not history:
            self.system("she has not done anything on her own yet")
            return
        self.system("what she has been doing on her own")
        for entry in history[-6:]:
            self.system(f"    {_clip(entry, 96)}")

    def show_memory(self):
        entries = controller.get_memory()
        notes = [e for e in entries if isinstance(e, dict) and controller.memory_kind(e.get("text", ""))]
        self.system(f"she remembers {len(notes)} notes")
        for entry in notes[-4:]:
            self.system(f"    {_clip(entry.get('text', ''), 96)}")

    # ---------------------------------------------------------------- loop

    def run(self):
        self.system("she is awake - say something whenever you like")
        self.note("f1 for the keys", seconds=6)
        while not self.stopping:
            try:
                key = self.screen.getch()
            except KeyboardInterrupt:
                break
            if key == -1:
                time.sleep(0.05)
            else:
                if not self.follow and (
                    key in (curses.KEY_NPAGE, curses.KEY_END) or self._is_text_key(key)
                ):
                    # Scrolling down, jumping to the newest line, or typing means
                    # the user is done reading the backlog: follow the chat again.
                    self.follow = True
                self.handle_key(key)
            now = time.time()
            if now - self.last_draw > 0.12:
                self.last_draw = now
                self.dots += 1
                self.draw()
        return 0


def run_window():
    """Start Kookie and hand the terminal over to her window."""
    real_stdout = sys.stdout
    sys.stdout = _QuietStdout()
    try:
        controller.start_runtime(connect_avatar=True, animate_parameters=True)
        return curses.wrapper(_boot)
    finally:
        sys.stdout = real_stdout
        try:
            controller.stop_runtime()
        except Exception:
            pass


def _boot(screen):
    # Pulled in here rather than at the top: Playwright costs 150ms to import and
    # this window should be on screen before that.
    import browser

    window = KookieWindow(screen)
    sys.stdout = _WindowStdout(window.system)
    # Browser confirmations have to be asked here: this window owns the
    # terminal, so an input() on the agent thread would never be seen.
    browser.set_confirmation_hook(window.ask_confirmation)
    screen.nodelay(True)
    screen.keypad(True)
    try:
        return window.run()
    finally:
        browser.set_confirmation_hook(None)
        sys.stdout = _QuietStdout()  # keep stop_runtime()'s prints off the frame


if __name__ == "__main__":
    raise SystemExit(run_window())