import collections
import functools
import glob
import ipaddress
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeout
from urllib.parse import quote_plus, unquote, urljoin, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

MAX_PAGE_CHARS = 10000
_playwright = None
_browser = None
_context = None
_page = None
_browser_name = None
# True while she is driving a Chrome Mommy already had open. Closing then only
# disconnects: her browser.close() on a connected Chrome can take her window and
# everything in it down with her, which is the opposite of what she was asked for.
_attached = False
# How she got into the browser she is attached to, for the sentence she says about
# it. Set by _attach_to_running_chrome.
_ATTACHED_NOTE = "she is using the window you already have open"
# True only for a tab she opened herself. A tab that was already in somebody
# else's window is left open when she disconnects, because it is not hers to
# close.
_own_page = False
# Reentrant because _get_page() holds it and calls close_browser() while a
# launch is half-finished; it is never held while waiting on the worker queue,
# so that cannot deadlock either.
_BROWSER_LOCK = threading.RLock()

# Playwright's sync API only answers on the thread that ran start(), and
# controller hands every message to a brand new thread, so the second browser
# request died with "Cannot switch to a different thread". One long-lived daemon
# thread owns the Playwright lifecycle and every action is queued onto it.
# The wait has to outlast a confirmation prompt, which the window gives the user
# 180 seconds to answer and which now runs on this same thread.
BROWSER_ACTION_TIMEOUT = 240.0
BROWSER_TIMEOUT_MESSAGE = "The browser did not answer in time, so nothing happened."
CHROME_EXECUTABLES = (
    "google-chrome",
    "google-chrome-stable",
    "google-chrome-beta",
    "google-chrome-unstable",
    "chrome",
    "chromium",
    "chromium-browser",
)
# Flatpak, Snap and self-installed Chrome are all normal on a Linux desktop and
# none of them are on PATH for a plain shell, so they are looked up by path too.
# The globs matter more than the fixed paths: a Chrome installed as a versioned
# /opt/google/chrome-124 directory, or a Flatpak whose export symlink was never
# created, is still the user's Chrome and missing it is what silently handed her
# Playwright's Chromium instead.
CHROME_EXECUTABLE_PATHS = (
    ("Google Chrome (flatpak)", "~/.local/share/flatpak/exports/bin/com.google.Chrome"),
    ("Google Chrome (flatpak)", "/var/lib/flatpak/exports/bin/com.google.Chrome"),
    ("Google Chrome (flatpak)", "/var/lib/flatpak/exports/host/bin/com.google.Chrome"),
    ("Google Chrome (flatpak)", "~/.local/share/flatpak/exports/host/bin/com.google.Chrome"),
    ("Chromium (flatpak)", "~/.local/share/flatpak/exports/bin/org.chromium.Chromium"),
    ("Chromium (flatpak)", "/var/lib/flatpak/exports/bin/org.chromium.Chromium"),
    ("Chromium (snap)", "/snap/bin/chromium"),
    ("Chromium (snap)", "/var/lib/snapd/snap/bin/chromium"),
    ("Google Chrome (snap)", "/snap/bin/chrome"),
    ("Google Chrome", "/opt/google/chrome/google-chrome"),
    ("Google Chrome", "/opt/google/chrome/chrome"),
    ("Google Chrome", "/opt/google/chrome-beta/google-chrome"),
    ("Google Chrome", "/usr/bin/google-chrome-stable"),
    ("Google Chrome", "/usr/lib/chromium-browser/chrome"),
    ("Google Chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
)
# Directory globs for the same reason: the path changes with the install and the
# folder does not.
CHROME_EXECUTABLE_GLOBS = (
    ("Google Chrome", "/opt/google/chrome*/chrome"),
    ("Google Chrome", "/opt/google/chrome*/google-chrome"),
    ("Google Chrome", "~/Applications/Google Chrome*.app/Contents/MacOS/Google Chrome"),
    ("Google Chrome", "~/.local/opt/google-chrome*/chrome"),
    ("Chromium", "~/.local/opt/chromium*/chrome"),
    ("Chromium (flatpak)", "~/.local/share/flatpak/exports/**/com.google.Chrome"),
)
# Where a Chrome that Mommy started herself answers. She attaches to that window
# instead of opening a second browser, so she works in her own tabs, her own
# logins and her own history.
CHROME_DEBUG_PORT = os.environ.get("KOOKIE_CHROME_DEBUG_PORT", "9222")
CHROME_CDP_URL = os.environ.get(
    "KOOKIE_CHROME_CDP_URL", f"http://127.0.0.1:{CHROME_DEBUG_PORT}"
)
# A Chrome of her own, used when the user's Chrome is already running and a launch
# aimed at its profile would only be handed to that running instance. It is a real
# Chrome with a real profile, just not the one with the logins in it.
CHROME_PROFILE_DIR = os.path.expanduser(
    os.environ.get("KOOKIE_CHROME_PROFILE_DIR", "~/.config/kookie-chrome")
)
# How long her own Chrome gets to open its debug port before she gives up and
# uses the other ways in. Chrome shows a window and takes a profile lock first,
# and a cold profile on a slow disk is not quick, so this is generous on purpose:
# it is paid once, when she starts a browser, and every page after that is
# instant. Half of it goes to the first attempt and half to the retry.
CHROME_START_TIMEOUT_SECONDS = 30.0
# Where she looks something up when nobody asked her to. DuckDuckGo's plain-HTML
# search is the one built to be read by a program: no account, no key, and it
# answers as ordinary page text.
BROWSE_SEARCH_URL = "https://html.duckduckgo.com/html/?q="
# Reading is quick; a page that hangs is not worth holding her conversation for.
BROWSE_TIMEOUT_SECONDS = 30
# How long one page load may take before she tries it once more. Generous on
# purpose: this is the timeout that fires while the model is using the machine,
# and a page that was only slow should not be reported as unreachable.
BROWSER_NAVIGATION_TIMEOUT_SECONDS = 45
# Her own browsing has to fit inside a thought, because that is what she is doing
# it inside, and it is holding the conversation while it does.
BROWSE_AUTONOMOUS_TIMEOUT_SECONDS = 12

SAFE_PAGE_KEYS = {
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "ArrowUp",
    "Backspace",
    "Control+A",
    "Control+Z",
    "Delete",
    "End",
    "Enter",
    "Escape",
    "Home",
    "PageDown",
    "PageUp",
    "Shift+Tab",
    "Space",
    "Tab",
}
# What she found to drive. A path when there is a program to point at, a flatpak
# app id when there is not, and on_host when that app belongs to the machine she is
# running inside rather than the one she is looking at.
BrowserProgram = collections.namedtuple(
    "BrowserProgram", "path label flatpak_app on_host"
)

SENSITIVE_FIELD_PATTERN = re.compile(
    r"password|passcode|credit.?card|card.?number|cc-|cvv|cvc|security.?code",
    re.IGNORECASE,
)
HIGH_IMPACT_ACTION_PATTERN = re.compile(
    r"\b(submit|send|publish|post|buy|purchase|checkout|pay|order|delete|remove|transfer|subscribe|unsubscribe)\b",
    re.IGNORECASE,
)


# --- The parts of the web she is not allowed to open ------------------------
# The network guard further down keeps her off the local network; this keeps her
# off the parts of the web that are not worth being on, and off the pages whose
# whole purpose is to take something from her. It runs before the address is
# resolved, so a blocked host is refused without a DNS lookup, and it runs again
# for every request a page makes, so a redirect, an image or a script cannot walk
# her in through the front door.
#
# Nothing here is a complete index of the unsafe web and it does not pretend to
# be - a list that could never be finished would either be so broad it breaks
# ordinary pages or so narrow it is theatre. It covers the categories that are
# worth refusing by default, and UNSAFE_SITE_FILE is there so the list can be
# extended without touching this file.

# Suffixes that are a whole site rather than a word in a hostname.
UNSAFE_HOST_SUFFIXES = (
    ("pornhub.com", "adult sites"),
    ("xvideos.com", "adult sites"),
    ("xnxx.com", "adult sites"),
    ("xhamster.com", "adult sites"),
    ("redtube.com", "adult sites"),
    ("youporn.com", "adult sites"),
    ("onlyfans.com", "adult sites"),
    ("fansly.com", "adult sites"),
    ("chaturbate.com", "adult sites"),
    ("stripchat.com", "adult sites"),
    ("bongacams.com", "adult sites"),
    ("myfreecams.com", "adult sites"),
    ("spankbang.com", "adult sites"),
    ("motherless.com", "adult sites"),
    ("thepiratebay.org", "piracy"),
    ("thepiratebay.com", "piracy"),
    ("1337x.to", "piracy"),
    ("rarbg.to", "piracy"),
    ("yts.mx", "piracy"),
    ("limetorrents.lol", "piracy"),
    ("eztv.re", "piracy"),
    ("bitorrent.com", "piracy"),
    ("demonoid.is", "piracy"),
    ("pornhubpremium.com", "adult sites"),
)

# Single hostname labels. A hostname is split on dots first, so this catches the
# long tail of made-up ad domains - xxx-hd-1080-7.xyz is blocked by "xxx" without
# anyone having to have written that host down.
UNSAFE_HOST_LABELS = (
    # adult
    "porn", "porno", "porns", "xxx", "nsfw", "hentai", "camgirl", "camgirls",
    "escort", "escorts", "milf", "creampie", "blowjob", "handjob", "rimjob",
    "camsoda", "webcam-xxx", "csae", "csam", "preteen", "bestiality",
    "shemale", "fetish",
    # gambling
    "casino", "casinos", "casino-online", "sportsbook", "betting", "bets", "bet",
    "roulette", "jackpot", "jackpots", "slots", "slot-machine", "blackjack",
    "poker-online", "lottery", "lotto", "scratchcards", "megaways",
    # drugs
    "drugs", "drugs-for-sale", "heroin", "meth", "methamphetamine", "cocaine",
    "fentanyl", "steroids", "anabolic",
    # weapons. A licensed gun shop is a shop, not a hazard, so the plain words are
    # deliberately absent and only the parts that are actually illicit are here.
    "silencer", "suppressor", "ghostguns", "autosear", "autoseq",
    # Gambling brands are compound labels - bet365, betfair, williamhill - so the
    # plain words above never match them and the names are written down here.
    "bet365", "betfair", "betvictor", "betway", "betfred", "unibet", "1xbet",
    "pinnacle", "williamhill", "ladbrokes", "mybookie", "dabble", "comeon",
    "partypoker", "pokerstars", "pokerdom", "betexplorer", "oddsportal",
    # hacking and malware
    "malware", "spyware", "ransomware", "botnet", "keylogger", "rootkit",
    "exploit-kit", "exploitkit", "ddos", "booter", "stresser", "nulled",
    "carding", "dumps", "fullz", "keygen", "warez", "cracks", "cracked",
    "torrent", "torrents", "bypass", "spoofer", "phishing", "webscr",
    "smishing",
)

# Roots that make a longer hostname label unsafe however the label continues:
# porn-*, xxx*, hentai*, casin*, roulett*, jackpot*, silencer*, ghostgun*,
# keygen*, malware*, spyware*, carding*, warez*. Each is a category word with no
# ordinary site behind it, unlike "cam" or "slot" or "crack", whose prefixes would
# take camden.gov.uk, sloth.com and crackdown.fandom.com with them.
UNSAFE_HOST_LABEL_PREFIXES = (
    "porn", "xxx", "hentai", "camgirl", "bestiality", "casin", "roulett",
    "jackpot", "silencer", "ghostgun", "keygen", "malware", "spyware",
    "carding", "warez", "cracked", "ransomware", "botnet", "keylogger",
    "cvv", "dumps", "fullz", "webscr", "smishing",
)

# Phishing lives in the shape of the name rather than in one word: an address
# like account-verify-secure-login.example is three label tokens and matches
# nothing above. These are matched against the hostname with its punctuation
# squeezed out, and each one is long enough that no ordinary site spells it.
UNSAFE_HOST_PHRASES = (
    "accountverify", "verifyaccount", "verifyyouraccount", "accountconfirmation",
    "confirmyouraccount", "securelogin", "loginsecure", "loginverify",
    "verifylogin", "accountunlock", "unlockaccount", "accountupdate",
    "updateyouraccount", "webscr", "smishing", "bitcoindoubler", "cryptodoubler",
    "ethdoubler", "doubleryour", "freecrypto", "airdropclaim", "claimyourprize",
    "claimyourreward", "youvoweligibility", "giveawaywinner", "luckywinner",
)

# Reference works are where a topic word in the address is the subject of the
# article. Wikipedia has a page on Phishing and on Carding, and refusing those
# is refusing the encyclopedia, so on these hosts only the always-refused list
# applies.
REFERENCE_HOSTS = (
    "wikipedia.org", "wikibooks.org", "wikisource.org", "wikiquote.org",
    "britannica.com", "bbc.co.uk", "nationalgeographic.com", "smithsonianmag.com",
    "snopes.com", "factcheck.org",
)

# A label that is one of the category words, optionally with digits after it:
# bet365, casino777, torrent9. Precise enough to leave "better" and "betting-tips"
# alone, which a bare prefix rule would not be.
UNSAFE_HOST_LABEL_NUMBERED = re.compile(
    r"^(?:bet\d{2,5}|(?:casino|poker|roulette|slots|jackpot|xxx|porn|torrent|"
    r"warez|keygen|malware|botnet|ransomware|keylogger|spyware|carding|"
    r"ghostgun|silencer)\d{0,5})$"
)

# Words that can appear anywhere in a hostname, hyphen or underscore included.
UNSAFE_HOST_WORDS = (
    "keygen", "warez", "cracked", "carding", "fullz", "webscr", "malware",
    "silencer", "ghostgun", "exploitkit", "ddosforhire", "bootloader",
)

# Free and throwaway domains that host an unreasonable share of malware and
# phishing. The list is short on purpose: every entry here also blocks whatever
# innocent site somebody manages to put on it.
UNSAFE_TLDS = (
    "tk", "ml", "ga", "cf", "gq", "zip", "mov", "cam", "cyou", "click",
    "loan", "download", "racing", "quest", "icu",
)

# Anything that ends in one of these is a program, not a page, and she has no
# business running one off the web.
UNSAFE_FILE_EXTENSIONS = (
    "exe", "msi", "apk", "scr", "hta", "bat", "cmd", "com", "pif", "vbs",
    "vbe", "jar", "dll", "ps1", "sh", "app", "dmg", "pkg", "deb", "rpm",
)

# Tokens in the path or query of an otherwise ordinary site. Matched on segments
# and on a few unmistakable substrings, so "alphabet" and "better" are safe.
UNSAFE_PATH_TOKENS = (
    "porn", "xxx", "nsfw", "hentai", "camgirl", "escort", "milf", "fetish",
    "preteen", "bestiality", "csae", "csam", "webcam",
    "casino", "sportsbook", "betting", "bets", "roulette", "jackpot", "slots",
    "blackjack", "lottery", "lotto", "scratchcard", "megaways",
    "meth", "methamphetamine", "heroin", "cocaine", "fentanyl", "steroids",
    "gun", "guns", "firearm", "firearms", "ammo", "ammunition", "silencer",
    "suppressor", "ghostgun",
    "keygen", "warez", "crack", "cracked", "torrent", "piracy", "bootleg",
    "malware", "spyware", "ransomware", "botnet", "keylogger", "rootkit",
    "exploit-kit", "exploitkit", "ddos", "booter", "stresser", "nulled",
    "carding", "dumps", "fullz", "cvv", "phishing", "webscr", "smishing",
)

# The same idea, narrowed to what is never an innocent page, and this is the list
# a whole page is judged on. The wider list above would refuse the Wikipedia
# article on Meth or on the Gun, which is a page worth reading: a topic is not a
# hazard, the place that trades in it is.
UNSAFE_PATH_TOKENS_STRICT = (
    "porn", "xxx", "nsfw", "hentai", "camgirl", "bestiality", "csae", "csam",
    "preteen", "keygen", "warez", "cracked", "malware", "spyware",
    "ransomware", "botnet", "keylogger", "rootkit", "exploit-kit", "exploitkit",
    "carding", "fullz", "phishing", "webscr", "smishing", "silencer",
    "ghostgun", "booter", "stresser",
)

# A search for these is refused whatever else is in it, because there is no
# innocent reading of them worth spending a look on.
UNSAFE_QUERY_ALWAYS = (
    (r"\b(csae|csam|child[\s-]?(?:porn|sexual|abuse)(?:material|content|site)?)\b",
     "child sexual abuse material"),
    (r"\bbestiality\b", "sexual content involving animals"),
    (r"\b(?:preteen|underage)\s+(?:nude|nudes|xxx|porn|girl|girls)\b",
     "sexual content involving children"),
    (r"\b(?:carding|fullz|cvv\s+dumps?|dumps?\s*\+\s*cvv|skimmer\s+skimm?er)\b",
     "stolen card data"),
    (r"\b(?:exploit[\s-]?kit|ransomware|botnet|ddos[\s-]?for[\s-]?hire|booter|stresser)\b",
     "malware and attack-for-hire services"),
    (r"\b(?:silencer|ghost\s+gun|suppressor|auto\s+sear|drill\s?kit|untraceable\s+gun)\b",
     "untraceable weapons"),
    (r"\b(?:pipe\s?bomb|pressure\s+cooker\s+bomb|how\s+to\s+make\s+(?:a\s+)?bomb|"
     r"nerve\s+agent|sarin|ricin|anthrax|ricin)\b", "weapons and dangerous substances"),
    (r"\b(?:keygen|warez|cracked\s+licen[cs]e|activator\s+crack)\b",
     "pirated software and licence cracks"),
)

# Terms that are only refused together with an intent to acquire something: a
# news article or a harm-reduction page about any of them is ordinary reading.
UNSAFE_QUERY_SENSITIVE = (
    r"(?:porn|pornography|xxx|sex\s*tape|nude|nudes|erotic|escort|onlyfans|hentai)",
    r"(?:meth|methamphetamine|cocaine|heroin|fentanyl|mdma|ecstasy|oxycontin|"
    r"oxycodone|st\s?eroids|anabolic\s+steroids|nitrous\s+oxide)",
    r"(?:gun|guns|firearm|firearms|rifle|pistol|handgun|ammo|ammunition|"
    r"assault\s+weapon|ar-?15|ak-?47)",
    r"(?:prescription\s*meds|adderall|xanax|valium|oxy\s*contin)",
)
UNSAFE_QUERY_BUY_INTENT = (
    r"\b(?:buy|purchase|order|shop|store|for\s+sale|cheap|cheapest|discount|"
    r"free|download|downloads|torrent|torrents|apk|mod(?:ded)?|crack(?:ed)?|"
    r"keygen|generator|online\s+buy|where\s+to\s+get|where\s+can\s+i\s+get|"
    r"link\s*https?:|mega\.nz|mediafire|telegram)\b"
)

# Optional file next to this module, so the list can grow or be corrected without
# editing this file:
#
#   [hosts]    one domain or suffix per line, matched the same way as the built-in
#              list. A leading *. matches any subdomain.
#   [queries]  one phrase per line; she will not search for it.
#   [allow]    one domain per line that overrides the hostname blocklist, for the
#              site you do want her to reach. The always-refused rules still
#              apply, so this is not a way past them.
#
# Lines starting with # are comments. The file is optional and read fresh when it
# changes.
UNSAFE_SITE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsafe_sites.txt")
_EXTRA_BLOCKED_STAMP = None
_EXTRA_BLOCKED_HOSTS = ()
_EXTRA_BLOCKED_QUERIES = ()
_EXTRA_ALLOWED_HOSTS = ()


def _refusal_detail(error):
    """The sentence explaining a refusal, whichever kind of refusal it was.

    _validate_public_url answers a blocked host with a BlockedPage so the reason
    can be said out loud, but a malformed address, a private network or an
    unresolvable host comes back as an ordinary ValueError. Both are refusals
    the person asking is entitled to hear, so both are turned into a sentence
    here instead of one of them escaping as a browser failure.
    """
    return error.refused() if isinstance(error, BlockedPage) else str(error)


class BlockedPage(ValueError):
    """A page she refuses to open, carrying the reason she can say out loud.

    A plain refusal message is what makes this readable rather than mysterious:
    "that site is blocked because it is a malware download page" is an answer, and
    a bare "that browser action did not work" is not.
    """

    def __init__(self, url, reason, kind="page"):
        self.url = str(url or "")
        self.reason = str(reason)
        self.kind = kind
        host = _host_of(self.url)
        super().__init__(
            f"she will not open {host or self.url or 'that'}: it is on her blocked "
            f"list ({self.reason})"
        )

    def refused(self):
        return str(self)


def _host_of(url):
    try:
        return (urlparse(str(url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _extra_blocked():
    """The additions from UNSAFE_SITE_FILE, re-read only when the file changes."""
    global _EXTRA_BLOCKED_STAMP, _EXTRA_BLOCKED_HOSTS, _EXTRA_BLOCKED_QUERIES
    global _EXTRA_ALLOWED_HOSTS

    try:
        info = os.stat(UNSAFE_SITE_FILE)
        stamp = (info.st_mtime_ns, info.st_size)
    except OSError:
        stamp = None
    if stamp == _EXTRA_BLOCKED_STAMP:
        return _EXTRA_BLOCKED_HOSTS, _EXTRA_BLOCKED_QUERIES

    hosts, queries, allowed, section = [], [], [], "hosts"
    try:
        with open(UNSAFE_SITE_FILE, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip().lower()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1].strip()
                    continue
                line = line.lstrip("*.").strip()
                if "quer" in section:
                    queries.append(line)
                elif "allow" in section:
                    allowed.append(line)
                else:
                    hosts.append(line)
    except OSError:
        pass
    _EXTRA_BLOCKED_STAMP = stamp
    _EXTRA_BLOCKED_HOSTS = tuple(hosts)
    _EXTRA_BLOCKED_QUERIES = tuple(queries)
    _EXTRA_ALLOWED_HOSTS = tuple(allowed)
    return _EXTRA_BLOCKED_HOSTS, _EXTRA_BLOCKED_QUERIES


def _host_is_allowed(host):
    """True when UNSAFE_SITE_FILE says this site is one she may use after all."""
    for pattern in _EXTRA_ALLOWED_HOSTS:
        if host == pattern or host.endswith("." + pattern):
            return True
    return False


def _host_is_unsafe(host):
    """Why this hostname is refused, or None when it is fine."""
    if not host or _host_is_allowed(host):
        return None
    labels = host.split(".")
    squashed = re.sub(r"[^a-z0-9]+", "", host)
    for phrase in UNSAFE_HOST_PHRASES:
        if phrase in squashed:
            return "a site shaped like a phishing or scam page"
    for suffix, reason in UNSAFE_HOST_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return reason
    for label in labels:
        if label in UNSAFE_HOST_LABELS:
            return "adult, gambling, drug, weapon or malware sites"
        if label.startswith(UNSAFE_HOST_LABEL_PREFIXES):
            return "adult, gambling, drug, weapon or malware sites"
        if UNSAFE_HOST_LABEL_NUMBERED.match(label):
            return "adult, gambling, drug, weapon or malware sites"
    for token in re.split(r"[^a-z0-9]+", host):
        if token in UNSAFE_HOST_LABELS or token in UNSAFE_HOST_WORDS:
            return "adult, gambling, drug, weapon or malware sites"
    if labels[-1] in UNSAFE_TLDS:
        return f".{labels[-1]} is a free domain used mostly for malware and phishing"
    extra_hosts, _ = _extra_blocked()
    for pattern in extra_hosts:
        if host == pattern or host.endswith("." + pattern):
            return "it is on the blocked list in unsafe_sites.txt"
    return None


def unsafe_url_reason(url, resource_type="document"):
    """Why this address is refused, or None when she may open it.

    Called for the address she was asked to open and for every request the page
    makes, so a safe page cannot smuggle her onto an unsafe one by redirect.

    resource_type matters. A whole page is judged on the strict list, because
    refusing a page for an unlucky word in its path breaks ordinary reading; an
    image, script or fetch is judged on the wider list, where a blocked sub-
    resource costs a picture and nothing else.
    """
    try:
        parsed = urlparse(str(url or ""))
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    reason = _host_is_unsafe(host)
    if reason:
        return reason

    # The path is unquoted first, or %70%6f%72%6e walks straight past the list.
    try:
        path = unquote(parsed.path or "").lower()
        query = unquote(parsed.query or "").lower()
    except (UnicodeDecodeError, ValueError):
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()

    if path.rsplit(".", 1)[-1] in UNSAFE_FILE_EXTENSIONS:
        return "that link is a program, not a page"
    tokens = re.split(r"[^a-z0-9]+", path)
    strict = resource_type == "document"
    if any(host == item or host.endswith("." + item) for item in REFERENCE_HOSTS):
        # An article about a nasty subject is not the nasty subject.
        tokens = []
    for token in tokens:
        if token in (UNSAFE_PATH_TOKENS_STRICT if strict else UNSAFE_PATH_TOKENS):
            return "adult, gambling, drug, weapon, malware or piracy pages"

    # An address is checked on the always-refused list only. The buy-intent
    # judgement belongs to the searches she composes, not to every page on the
    # open web, or a shopping page about anything would be unreachable.
    return unsafe_query_reason(query, allow_purchase=False)


def unsafe_query_reason(text, allow_purchase=True):
    """Why this search or question is refused, or None when it is fine.

    allow_purchase is False when judging an address somebody handed her: a topic
    word in a URL says what a page is about, and only a search she wrote herself
    is a decision to go looking.
    """
    text = str(text or "").lower()
    if not text.strip():
        return None
    for pattern, reason in UNSAFE_QUERY_ALWAYS:
        if re.search(pattern, text, re.IGNORECASE):
            return reason
    _, extra_queries = _extra_blocked()
    for phrase in extra_queries:
        if phrase and phrase in text:
            return "it is on the blocked list in unsafe_sites.txt"
    if allow_purchase and re.search(UNSAFE_QUERY_BUY_INTENT, text, re.IGNORECASE) and any(
        re.search(term, text, re.IGNORECASE) for term in UNSAFE_QUERY_SENSITIVE
    ):
        return "looking for a place to buy it"
    return None


_CONFIRM_HOOK = None

_WORKER_QUEUE = queue.SimpleQueue()
_WORKER_START_LOCK = threading.Lock()
_WORKER = None


def _run_browser_queue():
    """Own the Playwright lifecycle; only this thread may touch it."""
    while True:
        action, slot = _WORKER_QUEUE.get()
        try:
            slot.set_result(action())
        except BaseException as error:
            slot.set_exception(error)


def _ensure_worker():
    global _WORKER

    with _WORKER_START_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        if sys.is_finalizing():
            raise ValueError(BROWSER_TIMEOUT_MESSAGE)
        _WORKER = threading.Thread(
            target=_run_browser_queue, name="kookie-browser", daemon=True
        )
        _WORKER.start()


def _submit(action, timeout=BROWSER_ACTION_TIMEOUT):
    """Run one browser action on the Playwright thread and wait for its answer."""
    _ensure_worker()
    slot = Future()
    _WORKER_QUEUE.put((action, slot))
    try:
        return slot.result(timeout=timeout)
    except _FutureTimeout as error:
        # The action is still queued and will run a moment later; the caller
        # gets the same plain failure as any other browser error instead of a
        # queue exception it would only report as a glitch.
        raise ValueError(BROWSER_TIMEOUT_MESSAGE) from error


def _browser_action(function):
    """Keep a browser action on the one thread that owns Playwright.

    Running straight through when that thread is already the caller is what
    makes nested actions (follow_link -> open_page_url) work instead of
    queueing behind the job that is calling them.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if threading.current_thread() is _WORKER:
            return function(*args, **kwargs)
        return _submit(lambda: function(*args, **kwargs))

    return wrapper


def set_confirmation_hook(hook):
    """Send the "are you sure?" prompts somewhere the caller can actually show.

    Kookie drives the browser from a background thread while her curses window
    owns the terminal. A plain input() there prints its prompt into a swallowed
    stdout and then races the window for keystrokes, so a destructive action
    would look like a silent hang. The window installs a hook that asks in-chat;
    the CLI keeps the default input().
    """
    global _CONFIRM_HOOK
    _CONFIRM_HOOK = hook


def _confirm(message):
    if _CONFIRM_HOOK is not None:
        return _CONFIRM_HOOK(message)
    return input(message)


_RESOLVED_HOSTS = {}
_RESOLVED_HOSTS_TTL = 60.0


def _resolve_host(hostname, port):
    """Resolve a hostname once per session.

    The request guard runs for every image, stylesheet and XHR on a page, so a
    page with 200 sub-resources would otherwise do 200 identical DNS lookups.
    """
    key = (hostname, port)
    now = time.monotonic()
    cached = _RESOLVED_HOSTS.get(key)
    if cached is not None and cached[1] > now:
        return cached[0]
    addresses = [
        ipaddress.ip_address(result[4][0])
        for result in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    ]
    _RESOLVED_HOSTS[key] = (addresses, now + _RESOLVED_HOSTS_TTL)
    # Keep the cache from growing without bound.
    if len(_RESOLVED_HOSTS) > 256:
        oldest = min(_RESOLVED_HOSTS, key=lambda k: _RESOLVED_HOSTS[k][1])
        del _RESOLVED_HOSTS[oldest]
    return addresses


def _validate_public_url(url, resource_type="document"):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only public HTTP or HTTPS pages can be opened.")
    if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
        raise ValueError("URLs with credentials or nonstandard ports are blocked.")
    if parsed.hostname.endswith((".localhost", ".local")) or parsed.hostname == "localhost":
        raise ValueError("Local network addresses are blocked.")

    # Before the DNS lookup, not after: a blocked host should cost nothing and
    # reach nothing, and a page that links to one must not be able to resolve it.
    reason = unsafe_url_reason(url, resource_type)
    if reason:
        raise BlockedPage(url, reason)

    try:
        address = ipaddress.ip_address(parsed.hostname)
        addresses = [address]
    except ValueError:
        try:
            addresses = _resolve_host(parsed.hostname, parsed.port or 443)
        except OSError as error:
            raise ValueError("The page host could not be resolved.") from error

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Local and private network addresses are blocked.")

    return url


def _guard_request(route):
    try:
        # The resource type is passed on so a page is judged by the strict list
        # and its images and scripts by the wider one.
        _validate_public_url(route.request.url, route.request.resource_type)
    except (ValueError, OSError):
        route.abort()
        return
    route.continue_()


def _chrome_is_listening(endpoint=None, timeout=1.5):
    """True when a Chrome with a DevTools port is already up on that endpoint."""
    url = f"{(endpoint or CHROME_CDP_URL).rstrip('/')}/json/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _find_chrome_executable():
    """Her own Google Chrome on this machine, as a BrowserProgram (or an empty one).

    Her own Google Chrome comes first and bundled Chromium is never chosen while
    any Chrome exists, because that is the browser she was asked to use.

    Four passes, because the ways a Chrome gets onto a Linux desktop all hide it
    differently: a command on PATH, a fixed install path, a versioned or
    self-installed directory, and - the two that kept missing - a Flatpak whose
    export symlink was never created, and a Flatpak belonging to the *host* while
    she is running inside a sandbox that cannot see the host's /var. Chromium from
    Playwright is deliberately in none of them, so finding it can never be
    mistaken for finding the user's Chrome.
    """
    for name in CHROME_EXECUTABLES:
        found = shutil.which(name)
        if found:
            label = "Google Chrome" if name.startswith(("google-chrome", "chrome")) else "Chromium"
            return BrowserProgram(found, label, _flatpak_app_for(found), False)
    for label, path in CHROME_EXECUTABLE_PATHS:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return BrowserProgram(expanded, label, _flatpak_app_for(expanded), False)
    for label, pattern in CHROME_EXECUTABLE_GLOBS:
        try:
            found = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        except (OSError, ValueError):
            continue
        for candidate in found:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return BrowserProgram(candidate, label, _flatpak_app_for(candidate), False)
    app, on_host = _flatpak_browser_app()
    if app:
        # No path to hand to Playwright, but the program is there and can be run
        # as `flatpak run <app>`, which is the same Chrome the user clicks on.
        where = " on the host" if on_host else ""
        return BrowserProgram(None, f"Google Chrome (flatpak {app}{where})", app, on_host)
    return BrowserProgram(None, None, None, False)


def _flatpak_listing():
    """Flatpak's installed app list, as (lines, on_host).

    Inside a Flatpak - she is, when she is started from one - there is no flatpak
    binary and no /var/lib/flatpak to read, because the sandbox is not given them.
    `flatpak-spawn --host` is the way out: it runs the host's own flatpak against
    the host's own filesystem, which is how a host Flatpak Chrome is found at all.
    """
    flatpak = shutil.which("flatpak")
    commands = []
    if flatpak:
        commands.append(([flatpak, "list", "--app", "--columns=application"], False))
    if shutil.which("flatpak-spawn"):
        commands.append(
            (["flatpak-spawn", "--host", "flatpak", "list", "--app", "--columns=application"], True)
        )
    for command, on_host in commands:
        try:
            listing = subprocess.run(
                command, capture_output=True, text=True, timeout=25, check=False,
                # The host has its own /tmp and may not have the directory she
                # happens to be standing in, which fails the spawn outright.
                cwd="/" if on_host else None,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if listing.returncode == 0 and listing.stdout:
            return [line.strip() for line in listing.stdout.splitlines() if line.strip()], on_host
    return [], False


def _flatpak_browser_app():
    """A Chrome installed as a Flatpak: (app id, whether it belongs to the host)."""
    installed, on_host = _flatpak_listing(); installed = set(installed)
    for app in ("com.google.Chrome", "org.chromium.Chromium"):
        if app in installed:
            return app, on_host
    return None, False


def _user_chrome_profile_exists():
    """True when a Chrome has run here before, even if the program is not found.

    Worth knowing: a profile directory next to the missing binary means the user
    has a Chrome and she did not find it, which is a different problem from them
    not having one, and only one of them is fixed by installing something.
    """
    home = os.path.expanduser("~")
    return any(
        os.path.isdir(os.path.join(home, ".config", name))
        for name in ("google-chrome", "google-chrome-beta", "google-chrome-unstable")
    )


def _start_user_chrome(program):
    """Open the user's own Chrome with the debug port, so she drives their window.

    This is the difference between using her Chrome and using a Chromium: her own
    program, her own profile, a window she can be seen in and taken over from. It
    is only tried when no Chrome is already answering, so a Chrome the user
    started themselves is never disturbed and never restarted under them.

    Two attempts, because the first one often cannot work. A launch aimed at the
    profile Chrome is already using does not open a debug port: the new process
    hands its command line to the running one, prints "Opening in existing browser
    session" and exits. Chrome has done this by design since 136. So the first
    attempt is the ordinary one, and the second gives Chrome a profile of its own,
    which starts a second, separate, debuggable Chrome and leaves the user's own
    window exactly as it was.

    Returns True when a debug port answered afterwards.
    """
    port = CHROME_CDP_URL.rstrip("/").rsplit(":", 1)[-1].split("/")[0]
    if not port.isdigit() or not (program.path or program.flatpak_app):
        return False
    base = _flatpak_command(program) if program.flatpak_app else [program.path]
    common = [
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--no-default-browser-check",
        *_extra_chrome_args(),
    ]
    share = CHROME_START_TIMEOUT_SECONDS / 2
    for extra in ([], [f"--user-data-dir={CHROME_PROFILE_DIR}"]):
        try:
            subprocess.Popen(
                [*base, *common, *extra, "about:blank"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                # As above: the host does not share this filesystem's directories.
                cwd="/" if program.on_host else None,
            )
        except (OSError, ValueError):
            return False
        if _wait_for_chrome(share):
            return True
    return False


def _extra_chrome_args():
    """Extra flags from KOOKIE_CHROME_ARGS, for a machine that needs something."""
    extra = str(os.environ.get("KOOKIE_CHROME_ARGS", "") or "").split()
    return [item for item in extra if item]


def _wait_for_chrome(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _chrome_is_listening(timeout=1.0):
            return True
        time.sleep(0.4)
    return False


def _flatpak_command(program):
    """How to start a Flatpak browser: `flatpak run` here, or on the host.

    The bare command name is passed through on purpose. Probing for the flatpak
    binary only adds a way to get it wrong - the OS resolves the command when it
    is started, so a flatpak that appears later still works, and a missing one
    fails loudly at launch instead of being quietly turned into a bare app id.
    """
    if program.on_host:
        return ["flatpak-spawn", "--host", "flatpak", "run", program.flatpak_app]
    return ["flatpak", "run", program.flatpak_app]


def _pick_attached_page(context):
    """Her tab inside somebody else's window: an empty one if there is one.

    Falling back to the newest tab means she starts in whatever Mommy was last
    looking at, which is usually what she wants to work on anyway. Returns the
    page and whether she is the one who opened it.
    """
    pages = [page for page in context.pages if not page.is_closed()]
    for page in pages:
        if page.url == "about:blank":
            return page, False
    if pages:
        return pages[-1], False
    return context.new_page(), True


def _attach_to_running_chrome(playwright, started_by_her=False):
    """Take over the Chrome that is already open instead of starting another.

    started_by_her is only for saying so afterwards. Telling the user she is in a
    window they already had open when she opened it herself is a small lie that
    makes her look like she took over something, so the two cases are named apart.
    """
    global _ATTACHED_NOTE
    browser = playwright.chromium.connect_over_cdp(CHROME_CDP_URL, timeout=20000)
    contexts = browser.contexts
    context = contexts[0] if contexts else browser.new_context()
    # Routing is a Playwright-level guard against her wandering onto the local
    # network. A connected Chrome can refuse it, and her window matters more than
    # the guard, so a refusal is not allowed to fail the whole attach.
    try:
        context.route("**/*", _guard_request)
    except Exception as error:
        print("Could not install the network guard on the attached Chrome:", error)
    page, own_page = _pick_attached_page(context)
    page.bring_to_front()
    _ATTACHED_NOTE = (
        "she opened it and is working in it"
        if started_by_her
        else "she is using the window you already have open"
    )
    return browser, context, page, own_page, f"your open Chrome ({CHROME_CDP_URL})"


def _flatpak_app_for(executable):
    """The Flatpak app id behind a flatpak export path, if that is what this is."""
    name = os.path.basename(executable or "")
    for app in ("com.google.Chrome", "org.chromium.Chromium"):
        if name.startswith(app):
            return app
    return None


def _launch_own_chrome(playwright, program):
    """Start a Chrome of her own when there is no window to attach to."""
    launch_options = {"headless": False}
    if program.path:
        launch_options["executable_path"] = program.path
    try:
        browser = playwright.chromium.launch(**launch_options)
    except Exception:
        # A Flatpak Chrome is a wrapper that exists to be run through `flatpak
        # run`, and driving that wrapper directly is unreliable. Asking flatpak to
        # start it for her is what actually works on a desktop install.
        app = program.flatpak_app or _flatpak_app_for(program.path)
        flatpak = shutil.which("flatpak")
        if not app or not flatpak:
            raise
        browser = playwright.chromium.launch(
            executable_path=flatpak, args=["run", app, *launch_options.get("args", ())],
        )
    context = browser.new_context(
        java_script_enabled=True,
        accept_downloads=False,
        service_workers="block",
    )
    context.route("**/*", _guard_request)
    page = context.new_page()
    page.bring_to_front()
    return browser, context, page, True, program.label


def _describe_current_url(url):
    """Where she is now, said plainly instead of refused.

    Connecting to somebody else's Chrome means her tab is very often not a public
    web page - a chrome:// settings tab, a file:// page, an extension screen.
    Those are facts to report, not failures: raising here turned every browser
    tool into an error until she navigated somewhere else. Blocking a request is
    still done by _guard_request, before it happens.
    """
    url = str(url or "")
    if not url or url == "about:blank":
        return "about:blank"
    try:
        return _validate_public_url(url)
    except (ValueError, OSError) as error:
        return f"{url} (not a page she may work with: {error})"


def _get_page():
    global _playwright, _browser, _context, _page, _browser_name, _attached, _own_page

    with _BROWSER_LOCK:
        if _page and not _page.is_closed():
            return _page

        # A window can survive its page, and starting Playwright again over the
        # old globals would orphan the driver process and Chromium behind it.
        # Closed inline, not through the public wrapper: the lock is held here,
        # and queueing would wait on the very worker this thread may be.
        if _playwright is not None or _browser is not None or _context is not None:
            _close_browser()

        _playwright = sync_playwright().start()
        try:
            attached = False
            if _chrome_is_listening():
                try:
                    (
                        _browser,
                        _context,
                        _page,
                        _own_page,
                        _browser_name,
                    ) = _attach_to_running_chrome(_playwright)
                    attached = True
                except Exception as error:
                    # A port that answers /json/version can still refuse the
                    # connection (an old session, a Chrome that is shutting
                    # down). Falling back to her own window beats giving up.
                    print("Could not attach to the open Chrome:", error)
            # Second choice, and the one that makes the difference between her
            # Chrome and a Chromium: if the user has a Chrome but no debug port
            # is open, she starts their Chrome with one and attaches to that
            # rather than launching a browser of her own that they cannot see
            # inside. Only reached when nothing is already answering, so a Chrome
            # the user opened themselves is never restarted under them.
            if not attached:
                program = _find_chrome_executable()
                if program.path or program.flatpak_app:
                    if _start_user_chrome(program):
                        try:
                            (
                                _browser,
                                _context,
                                _page,
                                _own_page,
                                _browser_name,
                            ) = _attach_to_running_chrome(_playwright, started_by_her=True)
                            attached = True
                        except Exception as error:
                            print("Started Chrome but could not attach:", error)
                    if not attached:
                        (
                            _browser,
                            _context,
                            _page,
                            _own_page,
                            _browser_name,
                        ) = _launch_own_chrome(_playwright, program)
            if not attached and _page is None:
                # Nothing of the user's was found. Chromium is the last resort and
                # says so, because a silent Chromium looks exactly like success.
                _browser = _context = _page = None
                (
                    _browser,
                    _context,
                    _page,
                    _own_page,
                    _browser_name,
                ) = _launch_own_chrome(
                    _playwright, BrowserProgram(None, _bundled_chromium_name(), None, False)
                )
            _attached = attached
        except Exception:
            _close_browser()
            raise

        return _page


def _bundled_chromium_name():
    """What to call Playwright's Chromium, including why she ended up on it."""
    if _user_chrome_profile_exists():
        return (
            "Playwright Chromium - she could not find your Chrome program, though "
            "your Chrome profile is on this machine"
        )
    return "Playwright Chromium (no Google Chrome installed on this machine)"


@_browser_action
def open_browser():
    page = _get_page()
    page.bring_to_front()
    if _attached:
        return f"{_browser_name} is connected - {_ATTACHED_NOTE}."
    return f"{_browser_name} is open and ready."


def _page_state_once():
    page = _get_page()
    title = page.title()
    url = _describe_current_url(page.url)
    # Slice inside the browser: inner_text() would otherwise ship the entire
    # page body over CDP just for the first MAX_PAGE_CHARS of it.
    text = page.evaluate(
        "limit => (document.body && document.body.innerText || '').slice(0, limit)",
        MAX_PAGE_CHARS,
    )
    controls = page.locator(
        "input, textarea, select, button, a[href], [role='button'], [contenteditable='true']"
    ).evaluate_all(
        "elements => elements.filter(element => {"
        "const style = getComputedStyle(element);"
        "return style.visibility !== 'hidden' && style.display !== 'none' && element.getClientRects().length;"
        "}).slice(0, 50).map(element => {"
        "const labels = Array.from(element.labels || []).map(label => label.innerText.trim());"
        "const role = element.getAttribute('role') || ({A: 'link', BUTTON: 'button', SELECT: 'combobox'}[element.tagName] || element.tagName.toLowerCase());"
        "const type = element.getAttribute('type') || '';"
        "const options = element.tagName === 'SELECT' ? Array.from(element.options).map(option => option.label).slice(0, 20) : [];"
        "return {role, type, label: element.getAttribute('aria-label') || labels.join(' ') || element.innerText.trim() || element.placeholder || element.name || element.title || '',"
        "checked: ['checkbox', 'radio'].includes(type) ? element.checked : null, options};"
        "})"
    )
    control_lines = []
    for control in controls:
        details = [control["role"]]
        if control["type"]:
            details.append(control["type"])
        if control["checked"] is not None:
            details.append("checked" if control["checked"] else "unchecked")
        if control["options"]:
            details.append("options: " + ", ".join(control["options"]))
        control_lines.append(f"- {control['label'] or '(unlabeled)'} ({'; '.join(details)})")
    controls_text = "\n".join(control_lines) or "No visible interactive controls found."
    return (
        f"Title: {title}\nURL: {url}\n\nVisible page text:\n{text[:MAX_PAGE_CHARS]}"
        f"\n\nVisible controls:\n{controls_text}"
    )


@_browser_action
def get_page_state():
    """Everything she can see and click on, read once the page settles."""
    return _read_after_settling(_page_state_once)


@_browser_action
def list_page_links():
    page = _get_page()
    links = page.locator("a:visible").evaluate_all(
        "elements => elements.map(link => ({text: link.innerText.trim(), href: link.href}))"
    )
    public_links = []
    for link in links:
        if not link["text"] or not link["href"]:
            continue
        try:
            _validate_public_url(link["href"])
        except ValueError:
            continue
        public_links.append(link)
    if not public_links:
        return "No public links found on the current page."
    return "\n".join(
        f"{index}: {link['text'][:120]} -> {link['href']}"
        for index, link in enumerate(public_links[:40])
    )


@_browser_action
def open_page_url(url):
    """Open one public HTTP/HTTPS address in the browser and return the page state."""
    page = _get_page()
    destination = str(url or "").strip()
    try:
        destination = _validate_public_url(destination)
    except ValueError as blocked:
        # Said plainly rather than raised: the refusal is the useful answer here,
        # and whoever asked gets to hear why instead of a browser error. Blocked
        # pages are a ValueError too, so this covers every kind of no.
        return f"Not opening that. {_refusal_detail(blocked)}."
    _load_page(page, destination)
    final_url = page.url
    if final_url != "about:blank":
        # A redirect is still a place she ended up, so it gets the same judgement.
        try:
            final_url = _validate_public_url(final_url)
        except ValueError as blocked:
            return f"That link sent her somewhere else, so she stopped. {_refusal_detail(blocked)}."
    title = page.title()
    text = page.evaluate(
        "limit => (document.body && document.body.innerText || '').slice(0, limit)",
        MAX_PAGE_CHARS,
    )
    return f"Title: {title}\nURL: {final_url}\n\n{text[:MAX_PAGE_CHARS]}"


def _load_page(page, destination, seconds=None):
    """Go to a page, and go again if the first attempt was not the site's fault.

    Two failures that look like refusals and are not. ERR_ABORTED is the page
    being moved out from under the navigation - a second goto, the new tab page
    of a browser still starting, a redirect landing. A timeout is a slow page or
    a busy machine: while the model is answering it takes every core, and Chrome
    gets so little that a page which loads in a second times out at thirty.

    The model reads either as "this site cannot be opened" and tells you that, so
    both are tried once more before the failure is believed.
    """
    for attempt in range(2):
        try:
            page.goto(
                destination,
                # "commit", not "domcontentloaded". Committing means the server
                # answered and the page has started; domcontentloaded means the
                # whole page has also finished loading, which on a machine that is
                # already running the model, Chrome and a desktop is a thing that
                # can take a minute, and it is not the part she needs. What the
                # page looks like afterwards is one short, best-effort wait.
                wait_until="commit",
                # Playwright counts milliseconds. Passing the seconds straight
                # through gave every page load a 45ms budget, which is why the
                # agent spent its whole budget being told pages timed out.
                timeout=int((seconds or BROWSER_NAVIGATION_TIMEOUT_SECONDS) * 1000),
            )
            try:
                page.wait_for_load_state("domcontentloaded", timeout=10000)
            except PlaywrightTimeout:
                # Still loading, or never going to finish loading. She reads
                # whatever is there: _read_after_settling will try again if the
                # page is mid-move, and a half-loaded page is better than a
                # refusal to look at one.
                pass
            return
        except Exception as error:
            text = str(error)
            transient = ("ERR_ABORTED" in text or "Timeout" in text
                         or isinstance(error, PlaywrightTimeout))
            if not transient or attempt:
                raise
            # Let the page finish moving, or the machine finish breathing.
            time.sleep(2.0 if "ERR_ABORTED" in text else 5.0)


def _transient_page_error(error):
    """Is this the page moving out from under a read, rather than a broken read?"""
    if isinstance(error, PlaywrightTimeout):
        return False
    text = str(error)
    return any(marker in text for marker in (
        "Execution context was destroyed",
        "Target closed",
        "Frame was detached",
        "detached",
    ))


def _read_after_settling(read, attempts=2):
    """Read the page, and read it again if it moved while we were reading.

    A page that navigates takes its JavaScript context with it, and Playwright
    says so in words that look like a broken browser. Told "get_page_state did
    not work", the model concludes the page cannot be read at all, when the only
    thing that happened is that it finished loading half a second after she
    asked about it.
    """
    for attempt in range(attempts):
        try:
            return read()
        except Exception as error:
            if not _transient_page_error(error) or attempt:
                raise
            time.sleep(1.0)


def _read_page_text_once(page):
    """The visible words of whatever page she is on."""
    title = page.title()
    text = page.evaluate(
        "limit => (document.body && document.body.innerText || '').slice(0, limit)",
        MAX_PAGE_CHARS,
    )
    return f"Title: {title}\nURL: {page.url}\n\n{text[:MAX_PAGE_CHARS]}"


def _read_page_text(page):
    """The visible words of whatever page she is on, once it stops moving."""
    return _read_after_settling(lambda: _read_page_text_once(page))


@_browser_action
def _read_search_results(page, limit=8):
    """The search hits, not the whole page.

    DuckDuckGo's plain-HTML page opens with a list of every region it has, and
    that list is long enough to fill the whole text budget on its own. Reading
    the page as plain text handed back a column of country names and no results
    at all, so the hits are pulled out by name and given a budget of their own.
    """
    results = page.evaluate(
        """limit => Array.from(document.querySelectorAll('a.result__a')).slice(0, limit).map(link => {
            const box = link.closest('div.result') || link.parentElement;
            const snippet = box ? (box.querySelector('.result__snippet') || {}) : {};
            return {
                title: (link.innerText || '').trim(),
                url: link.href || '',
                snippet: (snippet.innerText || '').trim(),
            };
        })""",
        limit,
    )
    lines = []
    for index, item in enumerate(results, 1):
        lines.append(f"{index}. {item['title']}\n   {item['url']}\n   {item['snippet'][:240]}")
    if not lines:
        # No hits means the page changed shape or the search returned nothing,
        # and the ordinary page text is the more honest thing to hand back.
        return _read_page_text(page)
    return f"Search results for: {page.title()}\n\n" + "\n".join(lines)


def _browse_address(wanted):
    """The address to read, and whether it is a search or a page to open."""
    candidate = str(wanted or "").strip()
    if not candidate:
        raise ValueError("Nothing was named to look up.")
    if " " not in candidate and "://" in candidate:
        return _validate_public_url(candidate), False
    if " " not in candidate and "." in candidate and "/" not in candidate:
        try:
            return _validate_public_url("https://" + candidate), False
        except (ValueError, OSError):
            pass
    return f"{BROWSE_SEARCH_URL}{quote_plus(candidate)}", True


@_browser_action
def browse_website(target, stop_check=None, autonomous=False):
    """Look something up in the browser and bring back what the page says.

    This is the browsing she does on her own initiative, so it reads and does
    nothing else. Navigating and reading text is what "having a look" means;
    clicking, typing, submitting and logging in stay behind an actual request,
    because those are the actions that need somebody watching.
    """
    # A search is a request to go and find something, so the words of the search
    # are checked as well as the address it ends up on. Asking is the part where
    # she chooses, and that is where the block belongs. This runs before the
    # stand-down check because refusing costs nothing and needs no browser: a
    # stand-down is her owner calling her back in, not an answer about the site.
    unsafe = unsafe_query_reason(target)
    if unsafe:
        return (
            f"She does not look for that: {unsafe}. She looked somewhere else "
            "instead, or asked you what she should read."
        )
    if stop_check is not None:
        try:
            if stop_check():
                return "She stood down before the page opened."
        except Exception as error:
            print("stop_check failed:", error)
    try:
        address, is_search = _browse_address(target)
    except BlockedPage as blocked:
        return f"Not going there. {blocked.refused()}."
    except ValueError as error:
        return f"She cannot open that: {error}"
    page = _get_page()
    limit = BROWSE_AUTONOMOUS_TIMEOUT_SECONDS if autonomous else BROWSE_TIMEOUT_SECONDS
    try:
        # Her own browsing gets the same second attempt as everything else: a
        # search that timed out at twelve seconds while the model was busy is a
        # slow search, not a failed one.
        _load_page(page, address, seconds=limit)
    except Exception as error:
        # A search or a slow page timing out is an ordinary thing to hit, and
        # it must read as one rather than as a browser failure.
        return f"The page did not load: {error}"
    if stop_check is not None:
        try:
            if stop_check():
                return "She stood down before reading the page."
        except Exception as error:
            print("stop_check failed:", error)
    if is_search:
        return _read_search_results(page)
    return _read_page_text(page)


def _link_addressed_as(page, wanted):
    """Where a link on this page points, when the model named its address.

    The link listing shows both halves of a link - its words and where it goes -
    and the model reaches for the address at least as often as the words. Answering
    only one of the two things it was just shown is how "follow the link on the
    page" turned into "there is no link by that name".
    """
    wanted = str(wanted or "").strip()
    if not wanted or " " in wanted:
        return None
    hrefs = page.locator("a[href]").evaluate_all(
        "elements => elements"
        ".map(element => element.getAttribute('href') || '')"
        ".filter(Boolean)"
    )
    found = set()
    for href in hrefs:
        target = urljoin(page.url, href.strip())
        bare = wanted.rstrip("/")
        if wanted in (target, href) or (len(bare) > 1 and target.rstrip("/").endswith(bare)):
            found.add(target)
    if len(found) != 1:
        # Nothing on the page goes there, or more than one thing does, and
        # guessing which one she meant is how the wrong page gets opened.
        return None
    destination = found.pop()
    try:
        _validate_public_url(destination)
    except ValueError as blocked:
        return _refusal_detail(blocked)
    return destination


@_browser_action
def follow_link(link_text):
    page = _get_page()
    link = page.get_by_role("link", name=link_text, exact=True)
    count = link.count()
    if count == 0:
        addressed = _link_addressed_as(page, link_text)
        if isinstance(addressed, str) and addressed.startswith(("http://", "https://")):
            if HIGH_IMPACT_ACTION_PATTERN.search(link_text):
                approval = _confirm(
                    f"Follow {link_text!r}? Type CLICK to confirm, or anything else to cancel: "
                )
                if approval.strip() != "CLICK":
                    return "Link action cancelled."
            return open_page_url(addressed)
        if addressed:
            return f"Not following that. {addressed}"
        raise ValueError(
            f"No visible link named {link_text!r} was found, and nothing on this "
            "page points at that address either. Call list_page_links and use the "
            "link's visible text."
        )

    hrefs = link.evaluate_all("elements => elements.map(element => element.getAttribute('href') || '')")
    destinations = {
        urljoin(page.url, href or "")
        for href in hrefs
    }
    destinations.discard(page.url)
    if not destinations:
        # An href-less match resolves to the page itself and is dropped above,
        # so this is the "link with nowhere to go" case, not an ambiguity.
        raise ValueError(f"Link {link_text!r} points at the page you are already on.")
    if len(destinations) != 1:
        raise ValueError(f"Link name {link_text!r} is ambiguous; specify its URL.")
    destination = destinations.pop()
    if HIGH_IMPACT_ACTION_PATTERN.search(link_text):
        approval = _confirm(f"Follow {link_text!r}? Type CLICK to confirm, or anything else to cancel: ")
        if approval.strip() != "CLICK":
            return "Link action cancelled."
    return open_page_url(destination)


def _find_labeled_control(page, selector, label):
    fields = page.locator(selector)
    matches = fields.evaluate_all(
        "(elements, query) => elements.map((element, index) => ({"
        "index, element, labels: [element.getAttribute('aria-label'), element.placeholder, "
        "element.name, ...(Array.from(element.labels || []).map(label => label.innerText))]"
        ".filter(Boolean).join(' ').toLowerCase()}))"
        ".filter(item => item.labels.includes(query.toLowerCase()))",
        label,
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one matching control for {label!r}, found {len(matches)}.")
    return fields.nth(matches[0]["index"])


def _check_field_is_safe(field, label):
    state = field.evaluate(
        "element => ({disabled: element.disabled, readOnly: element.readOnly, "
        "type: element.type || '', autocomplete: element.autocomplete || '', "
        "name: element.name || '', id: element.id || ''})"
    )
    if state["disabled"] or state["readOnly"]:
        raise ValueError("That field cannot be edited.")
    metadata = " ".join((label, state["type"], state["autocomplete"], state["name"], state["id"]))
    if state["type"].casefold() in ("password", "file") or SENSITIVE_FIELD_PATTERN.search(metadata):
        raise ValueError("Password and payment fields are blocked.")
    return state


@_browser_action
def fill_text_field(label, value):
    page = _get_page()
    field = _find_labeled_control(
        page,
        "input[type='text'], input[type='search'], input[type='url'], input[type='email'], "
        "input[type='tel'], input[type='number'], textarea, [contenteditable='true']",
        label,
    )
    _check_field_is_safe(field, label)
    field.fill("")
    field.press_sequentially(value)
    return f"Draft text filled into {label!r}. Nothing was submitted."


@_browser_action
def type_into_focused_field(text):
    if len(text) > 5000:
        raise ValueError("Text is too long to type in one action.")
    page = _get_page()
    active_field = page.locator(":focus")
    if active_field.count() != 1:
        raise ValueError("Focus a text field first; no editable field is currently focused.")
    _check_field_is_safe(active_field, "focused field")
    is_editable = active_field.evaluate(
        "element => ['INPUT', 'TEXTAREA'].includes(element.tagName) || element.isContentEditable"
    )
    if not is_editable:
        raise ValueError("The focused element is not a text-editable field.")
    page.keyboard.insert_text(text)
    return "Typed text into the focused field. Nothing was submitted."


@_browser_action
def select_dropdown_option(label, option):
    page = _get_page()
    control = _find_labeled_control(page, "select", label)
    _check_field_is_safe(control, label)
    matches = control.locator("option").evaluate_all(
        "(elements, query) => elements.map(option => ({label: option.label.trim(), value: option.value}))"
        ".filter(option => option.label.toLowerCase() === query.toLowerCase() || option.value.toLowerCase() === query.toLowerCase())",
        option,
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one option {option!r} in {label!r}, found {len(matches)}.")
    control.select_option(value=matches[0]["value"])
    return f"Selected {matches[0]['label']!r} in {label!r}. Nothing was submitted."


@_browser_action
def set_checkbox(label, checked):
    checkbox = _get_page().get_by_role("checkbox", name=label, exact=True)
    count = checkbox.count()
    if count != 1:
        raise ValueError(f"Expected one checkbox named {label!r}, found {count}.")
    if checked:
        checkbox.check()
    else:
        checkbox.uncheck()
    return f"Set {label!r} to {'checked' if checked else 'unchecked'}. Nothing was submitted."


@_browser_action
def click_page_element(role, label):
    if role not in ("link", "button"):
        raise ValueError("Only links and buttons can be clicked by name.")

    page = _get_page()
    element = page.get_by_role(role, name=label, exact=True)
    count = element.count()
    if count != 1:
        raise ValueError(f"Expected one {role} named {label!r}, found {count}.")

    if role == "link":
        destination = urljoin(page.url, element.get_attribute("href") or "")
        try:
            _validate_public_url(destination)
        except ValueError as blocked:
            return f"Not clicking that. {_refusal_detail(blocked)}."
        if HIGH_IMPACT_ACTION_PATTERN.search(label):
            approval = _confirm(f"Click {label!r}? Type CLICK to confirm, or anything else to cancel: ")
            if approval.strip() != "CLICK":
                return "Link click cancelled."
        element.click(timeout=10000)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=10000)
        except PlaywrightTimeout:
            # The click or submit landed, which is the part that was asked
            # for. Whether the page it opened has finished loading is
            # something to read next, not a reason to call it a failure.
            pass
        return f"Clicked the {label!r} link. Current page: {_describe_current_url(page.url)}"

    button_type = element.get_attribute("type") or "submit"
    if button_type.casefold() == "submit":
        raise ValueError("Use submit_form to submit, which requires explicit confirmation.")
    if HIGH_IMPACT_ACTION_PATTERN.search(label):
        approval = _confirm(f"Click {label!r}? Type CLICK to confirm, or anything else to cancel: ")
        if approval.strip() != "CLICK":
            return "Button click cancelled."
    element.click(timeout=10000)
    return f"Clicked the {label!r} button. Current page: {_describe_current_url(page.url)}"


@_browser_action
def hover_page_element(role, label):
    if role not in ("link", "button"):
        raise ValueError("Only links and buttons can be hovered by name.")

    element = _get_page().get_by_role(role, name=label, exact=True)
    count = element.count()
    if count != 1:
        raise ValueError(f"Expected one {role} named {label!r}, found {count}.")
    element.hover(timeout=10000)
    return f"Moved the mouse over the {label!r} {role}."


@_browser_action
def scroll_page(direction):
    distances = {"down": 600, "up": -600, "left": -600, "right": 600}
    direction = direction.casefold()
    distance = distances.get(direction)
    if distance is None:
        raise ValueError("Scroll direction must be up, down, left, or right.")
    horizontal = direction in ("left", "right")
    _get_page().mouse.wheel(distance if horizontal else 0, 0 if horizontal else distance)
    return f"Scrolled {direction} on the page."


@_browser_action
def press_page_key(key):
    if key not in SAFE_PAGE_KEYS:
        raise ValueError("That key is not available to the browser agent.")
    if key in ("Enter", "Space"):
        confirmation = key.upper()
        approval = _confirm(f"Press {key} on the focused page element? Type {confirmation} to confirm: ")
        if approval.strip() != confirmation:
            return f"{key} keypress cancelled."
    _get_page().keyboard.press(key)
    return f"Pressed {key}."


@_browser_action
def navigate_page(action):
    page = _get_page()
    moves = {"back": page.go_back, "forward": page.go_forward, "reload": page.reload}
    if action not in moves:
        raise ValueError("Navigation action must be back, forward, or reload.")
    try:
        # "commit" for the same reason open_page_url uses it: the move is done the
        # moment the browser commits to it, and waiting for the whole page to
        # finish loading is what turned a one-second step back into a ten-second
        # timeout on a machine that was already busy.
        moves[action](
            wait_until="commit",
            timeout=BROWSER_NAVIGATION_TIMEOUT_SECONDS * 1000,
        )
    except PlaywrightTimeout:
        # Slow is not the same as impossible. Where the tab ended up is the answer
        # that was asked for, and that is readable whether or not it finished.
        pass
    # Going back can land on the empty page or a browser-internal one, which
    # is not a failure - it is just where the tab is.
    return f"Navigated {action}. Current page: {_describe_current_url(page.url)}"


@_browser_action
def submit_form(button_label):
    page = _get_page()
    buttons = page.locator("button[type='submit'], input[type='submit']")
    matches = buttons.evaluate_all(
        "(elements, query) => elements.map((element, index) => ({"
        "index, text: (element.innerText || element.value || '').trim()}))"
        ".filter(item => item.text.toLowerCase() === query.toLowerCase())",
        button_label,
    )
    if not matches and buttons.count() == 1:
        matches = [{"index": 0}]
    if len(matches) != 1:
        raise ValueError(f"Expected one submit button named {button_label!r}, found {len(matches)}.")

    page_title = page.title()
    page_url = _describe_current_url(page.url)
    approval = _confirm(
        f"Review the form in the browser. Submit {page_title!r} at {page_url}? "
        "Type SUBMIT to confirm, or anything else to cancel: "
    )
    if approval.strip() != "SUBMIT":
        return "Submission cancelled; the form was not submitted."

    buttons.nth(matches[0]["index"]).click(timeout=10000)
    try:
        page.wait_for_load_state("domcontentloaded", timeout=10000)
    except PlaywrightTimeout:
        # The submit landed, which is the part that was asked for. Whether
        # the page it opened has finished loading is something to read
        # next, not a reason to call it a failure.
        pass
    return f"Submitted the confirmed form. Current page: {_describe_current_url(page.url)}"


def _close_browser():
    """Shut the browser down on the Playwright thread. Safe to call twice."""
    global _playwright, _browser, _context, _page, _browser_name, _attached, _own_page
    global _ATTACHED_NOTE

    # Under the lock, because _get_page() reads the same globals while it
    # launches, and a half-torn-down browser there means a leaked process. The
    # lock is reentrant so this is safe from _get_page() as well, which holds it.
    with _BROWSER_LOCK:
        if _attached:
            # Somebody else's window: closing the context or the browser here
            # would close her profile and everything else she was told to use.
            # Stopping Playwright only drops the connection, and a tab she did
            # not open is none of hers to close.
            if _page and _own_page:
                try:
                    _page.close()
                except Exception:
                    pass
        else:
            if _context:
                try:
                    _context.close()
                except Exception:
                    pass
            if _browser:
                try:
                    _browser.close()
                except Exception:
                    pass
        if _playwright:
            try:
                _playwright.stop()
            except Exception:
                pass
        _playwright = None
        _browser = None
        _context = None
        _page = None
        _browser_name = None
        _attached = False
        _own_page = False
        _ATTACHED_NOTE = "she is using the window you already have open"


@_browser_action
def close_browser():
    _close_browser()


def shutdown_browser(timeout=5.0):
    """Let go of the browser without waiting behind a long action.

    Stopping Kookie must not hang: a confirmation prompt can hold the browser
    thread for minutes, and this runs while the process is on its way out. If the
    worker is still busy the references are simply dropped - it is a daemon
    thread and dies with the process - while a Chrome she attached to stays open
    exactly as it was found.
    """
    try:
        _submit(_close_browser, timeout=timeout)
    except ValueError:
        with _BROWSER_LOCK:
            globals().update(
                _playwright=None, _browser=None, _context=None, _page=None,
                _browser_name=None, _attached=False, _own_page=False,
                _ATTACHED_NOTE="she is using the window you already have open",
            )
    except Exception:
        pass