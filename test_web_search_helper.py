"""Tests for the web search helper."""

import unittest
from unittest import mock

import web_search_helper


_SAMPLE_HTML = """
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=1">Example Title</a>
  </h2>
  <a class="result__snippet" href="https://example.com/page">This is a snippet of text.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a class="result__a" href="https://example.org/other">Another Title</a>
  </h2>
  <a class="result__snippet" href="https://example.org/other">Another snippet.</a>
</div>
"""


class WebSearchParseTests(unittest.TestCase):
    def test_strips_tags(self):
        self.assertEqual(web_search_helper._strip_tags("<b>hi</b>"), "hi")

    def test_unescapes_entities(self):
        # html.unescape turns &amp; into &, which is the correct behaviour for
        # search snippets: the ampersand is the real character, not markup.
        self.assertEqual(web_search_helper._unescape("a &amp; b &lt;tag&gt;"), "a & b <tag>")

    def test_search_extracts_hits(self):
        with mock.patch.object(web_search_helper._controller, "http_request", return_value=_SAMPLE_HTML):
            hits = web_search_helper.search("example", limit=5)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["title"], "Example Title")
        self.assertEqual(hits[0]["url"], "https://example.com/page")
        self.assertEqual(hits[0]["snippet"], "This is a snippet of text.")
        # The second hit has no redirect, so it comes through unchanged.
        self.assertEqual(hits[1]["url"], "https://example.org/other")

    def test_search_empty_query(self):
        self.assertEqual(web_search_helper.search(""), [])
        self.assertEqual(web_search_helper.search(None), [])

    def test_search_network_failure_returns_empty(self):
        with mock.patch.object(web_search_helper._controller, "http_request", side_effect=RuntimeError("down")):
            self.assertEqual(web_search_helper.search("anything"), [])

    def test_search_no_results_page(self):
        with mock.patch.object(web_search_helper._controller, "http_request", return_value="<html>nothing here</html>"):
            self.assertEqual(web_search_helper.search("zzz"), [])

    def test_search_respects_limit(self):
        with mock.patch.object(web_search_helper._controller, "http_request", return_value=_SAMPLE_HTML):
            hits = web_search_helper.search("example", limit=1)
        self.assertEqual(len(hits), 1)

    def test_search_text_formats_hits(self):
        with mock.patch.object(web_search_helper._controller, "http_request", return_value=_SAMPLE_HTML):
            text = web_search_helper.search_text("example", limit=5)
        self.assertIn("Example Title", text)
        self.assertIn("https://example.com/page", text)

    def test_search_text_no_results(self):
        with mock.patch.object(web_search_helper._controller, "http_request", return_value="<html>nothing</html>"):
            self.assertIn("No results", web_search_helper.search_text("zzz"))


if __name__ == "__main__":
    unittest.main()