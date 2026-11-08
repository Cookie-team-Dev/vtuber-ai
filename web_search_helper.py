"""Web search helper for Kookie.

This used to be a placeholder that just echoed the query. It now does a real
search through DuckDuckGo's HTML endpoint (no API key, no JavaScript), reusing
the shared ``http_request`` machinery so it gets the same retry, timeout and
truncation behaviour as every other HTTP call.

The result is a short list of hits - title, URL and a one-line snippet - which
is what a person wants when they ask "what's the weather in Paris" or "who
wrote The Little Prince".
"""

import html
import re
import urllib.parse

import controller as _controller


_DDGH_URL = "https://html.duckduckgo.com/html/"
_UDDG_RE = re.compile(r"uddg=([^&]+)")
_SNIPPET_RE = re.compile(r"<a[^>]*class=\"result__snippet\"[^>]*>(.*?)</a>", re.S | re.I)
_TITLE_RE = re.compile(r"<a[^>]*class=\"result__a\"[^>]*>(.*?)</a>", re.S | re.I)
_LINK_RE = re.compile(r"<a[^>]*class=\"result__a\"[^>]*href=\"(.*?)\"", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text):
    return _TAG_RE.sub("", text).strip()


def _unescape(text):
    return html.unescape(_strip_tags(text))


def search(query, limit=5):
    """Search the web and return a list of ``{title, url, snippet}`` dicts.

    Falls back to the plain echo when the network is unavailable, so the tool
    never hard-fails just because a search engine was slow.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    try:
        body = _controller.http_request(
            _DDGH_URL + "?" + urllib.parse.urlencode({"q": query.strip()})
        )
    except Exception:  # noqa: BLE001 - search must not break the turn
        return []
    if not body or "result__snippet" not in body:
        return []
    snippets = _SNIPPET_RE.findall(body)
    hits = []
    # The snippet regex is the anchor: every result has exactly one. Walk
    # backwards from each snippet to the nearest preceding title link so the
    # two stay paired even when the page has several results.
    snippet_positions = [(m.start(), m.end()) for m in _SNIPPET_RE.finditer(body)]
    title_matches = list(_TITLE_RE.finditer(body))
    link_matches = list(_LINK_RE.finditer(body))
    for start, end in snippet_positions:
        snippet = body[start:end]
        title = ""
        for tm in title_matches:
            if tm.start() < start:
                title = tm.group(1)
            else:
                break
        raw = ""
        for lm in link_matches:
            if lm.start() < start:
                raw = lm.group(1)
            else:
                break
        # DuckDuckGo wraps real URLs in a redirect; unwrap it when present.
        match = _UDDG_RE.search(raw)
        url = urllib.parse.unquote(match.group(1)) if match else raw
        if not url or not url.startswith("http"):
            continue
        hits.append({
            "title": _unescape(title),
            "url": url,
            "snippet": _unescape(snippet),
        })
        if len(hits) >= max(1, limit):
            break
    return hits


def search_text(query, limit=5):
    """The same as ``search`` but formatted as a prompt-ready string."""
    hits = search(query, limit=limit)
    if not hits:
        return f"No results for {query!r}."
    lines = []
    for hit in hits:
        lines.append(f"- {hit['title'] or hit['url']}\n  {hit['snippet']}\n  {hit['url']}")
    return "\n".join(lines)
