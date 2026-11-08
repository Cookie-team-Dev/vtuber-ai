"""Tests for the core utility helpers.

These are pure functions with no I/O, so they are straightforward to cover.
The stub modules (misc, more_utils, tools_extra, helpers) re-export from
core_utils, which is checked here as well.
"""

import time
import unittest

import core_utils
import misc
import more_utils
import helpers
import tools_extra


class TruncateTests(unittest.TestCase):
    def test_short_text_is_returned_unchanged(self):
        self.assertEqual(core_utils.truncate("hello"), "hello")

    def test_none_becomes_empty_string(self):
        self.assertEqual(core_utils.truncate(None), "")

    def test_long_text_is_cut_with_suffix(self):
        result = core_utils.truncate("x" * 300, max_chars=10)
        self.assertEqual(len(result), 10)
        self.assertTrue(result.endswith("…"))

    def test_custom_suffix(self):
        result = core_utils.truncate("x" * 300, max_chars=8, suffix="..")
        self.assertTrue(result.endswith(".."))
        self.assertEqual(len(result), 8)

    def test_max_chars_equal_to_suffix_uses_suffix_only(self):
        result = core_utils.truncate("x" * 300, max_chars=1, suffix="…")
        self.assertEqual(result, "…")


class SlugifyTests(unittest.TestCase):
    def test_lowercases_and_joins_with_hyphens(self):
        self.assertEqual(core_utils.slugify("Hello World!"), "hello-world")

    def test_collapses_repeated_punctuation(self):
        self.assertEqual(core_utils.slugify("a--b___c"), "a-b-c")

    def test_strips_leading_and_trailing_hyphens(self):
        self.assertEqual(core_utils.slugify("!!!hello!!!"), "hello")

    def test_empty_returns_empty(self):
        self.assertEqual(core_utils.slugify(""), "")

    def test_max_chars(self):
        self.assertEqual(len(core_utils.slugify("a" * 100, max_chars=10)), 10)


class UniqueIdTests(unittest.TestCase):
    def test_prefix_is_preserved(self):
        self.assertTrue(core_utils.unique_id("req_").startswith("req_"))

    def test_two_calls_differ(self):
        self.assertNotEqual(core_utils.unique_id(), core_utils.unique_id())

    def test_length_without_prefix(self):
        self.assertEqual(len(core_utils.unique_id()), 12)


class ChunksTests(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(
            list(core_utils.chunks([1, 2, 3, 4], 2)),
            [[1, 2], [3, 4]],
        )

    def test_uneven_split(self):
        self.assertEqual(
            list(core_utils.chunks([1, 2, 3, 4, 5], 2)),
            [[1, 2], [3, 4], [5]],
        )

    def test_size_larger_than_list(self):
        self.assertEqual(list(core_utils.chunks([1, 2], 10)), [[1, 2]])

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            list(core_utils.chunks([1, 2], 0))


class ParseIntTests(unittest.TestCase):
    def test_string(self):
        self.assertEqual(core_utils.parse_int("42"), 42)

    def test_float(self):
        self.assertEqual(core_utils.parse_int(42.9), 42)

    def test_bool(self):
        self.assertEqual(core_utils.parse_int(True), 1)
        self.assertEqual(core_utils.parse_int(False), 0)

    def test_int_passthrough(self):
        self.assertEqual(core_utils.parse_int(7), 7)

    def test_bad_value_returns_default(self):
        self.assertEqual(core_utils.parse_int("nope"), 0)
        self.assertEqual(core_utils.parse_int("nope", default=-1), -1)

    def test_whitespace_is_stripped(self):
        self.assertEqual(core_utils.parse_int("  99  "), 99)


class HumanSecondsTests(unittest.TestCase):
    def test_sub_second(self):
        self.assertEqual(core_utils.human_seconds(0.3), "just now")

    def test_seconds(self):
        self.assertEqual(core_utils.human_seconds(5), "5s")

    def test_minutes_and_seconds(self):
        self.assertEqual(core_utils.human_seconds(65), "1m 5s")

    def test_hours_minutes_seconds(self):
        self.assertEqual(core_utils.human_seconds(3661), "1h 1m 1s")

    def test_days(self):
        self.assertEqual(core_utils.human_seconds(90_061), "1d 1h 1m 1s")

    def test_days_hide_seconds(self):
        self.assertEqual(core_utils.human_seconds(86_400), "1d")

    def test_bad_input(self):
        self.assertEqual(core_utils.human_seconds("not a number"), "unknown")


class NowIsoTests(unittest.TestCase):
    def test_format(self):
        value = core_utils.now_iso()
        self.assertEqual(len(value), 20)
        self.assertEqual(value[10], "T")
        self.assertEqual(value[19], "Z")


class CoalesceTests(unittest.TestCase):
    def test_returns_first_non_none(self):
        self.assertEqual(core_utils.coalesce(None, None, "x", "y"), "x")

    def test_all_none(self):
        self.assertIsNone(core_utils.coalesce(None, None))

    def test_falsy_values_are_kept(self):
        self.assertEqual(core_utils.coalesce(None, 0, "x"), 0)


class SafeGetTests(unittest.TestCase):
    def test_walks_nested_dict(self):
        self.assertEqual(core_utils.safe_get({"a": {"b": {"c": 1}}}, "a", "b", "c"), 1)

    def test_missing_key_returns_default(self):
        self.assertIsNone(core_utils.safe_get({"a": {}}, "a", "b"))

    def test_default_argument(self):
        self.assertEqual(core_utils.safe_get({"a": {}}, "a", "b", default="x"), "x")

    def test_non_dict_intermediate_returns_default(self):
        self.assertIsNone(core_utils.safe_get({"a": 1}, "a", "b"))


class StubReExportTests(unittest.TestCase):
    """The old stub modules must still work and re-export the real helpers."""

    def test_misc_re_exports(self):
        self.assertEqual(misc.truncate("hello"), "hello")
        self.assertEqual(misc.slugify("Hello World"), "hello-world")
        self.assertEqual(misc.x(), "x")

    def test_more_utils_re_exports(self):
        self.assertEqual(more_utils.parse_int("42"), 42)
        self.assertEqual(more_utils.foo(), "foo")
        self.assertEqual(more_utils.bar(), "bar")

    def test_helpers_re_exports(self):
        self.assertEqual(helpers.parse_int("7"), 7)
        self.assertEqual(helpers.a(), 1)
        self.assertEqual(helpers.b(), 2)

    def test_tools_extra_re_exports(self):
        self.assertEqual(tools_extra.truncate("hello", max_chars=3, suffix=""), "hel")
        self.assertEqual(tools_extra.slugify("Hello World!"), "hello-world")
        self.assertEqual(tools_extra.coalesce(None, None, "fallback"), "fallback")
        self.assertEqual(tools_extra.safe_get({"a": 1}, "a"), 1)
        self.assertEqual(tools_extra.safe_get({}, "missing", default="x"), "x")
        self.assertEqual(tools_extra.tool1(), "tool1")


if __name__ == "__main__":
    unittest.main()