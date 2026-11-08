"""Tests for the re-export stubs that keep backwards compatibility."""

import unittest

import extras
import extras_all
import mods_all
import misc
import more_utils
import tools_extra


class ReExportTests(unittest.TestCase):
    def test_extras_re_exports_core_utils(self):
        self.assertEqual(extras.truncate("hello", max_chars=3, suffix=""), "hel")
        self.assertEqual(extras.slugify("Hello World!"), "hello-world")
        self.assertEqual(extras.coalesce(None, None, "fallback"), "fallback")
        self.assertEqual(extras.safe_get({"a": 1}, "a"), 1)
        self.assertEqual(extras.safe_get({}, "missing", default="x"), "x")

    def test_extras_greet(self):
        self.assertIn("Mommy", extras.greet())

    def test_extras_all_re_exports(self):
        self.assertEqual(extras_all.truncate("hello", max_chars=3, suffix=""), "hel")
        self.assertEqual(extras_all.slugify("Hello World!"), "hello-world")
        self.assertEqual(extras_all.coalesce(None, "fallback"), "fallback")
        self.assertIsInstance(extras_all.unique_id(), str)

    def test_extras_all_extra_raises_on_unknown(self):
        with self.assertRaises(KeyError):
            extras_all.extra("does_not_exist")

    def test_mods_all_re_exports(self):
        self.assertEqual(mods_all.truncate("hello", max_chars=3, suffix=""), "hel")
        self.assertEqual(mods_all.slugify("Hello World!"), "hello-world")
        self.assertEqual(mods_all.coalesce(None, "fallback"), "fallback")

    def test_mods_all_mod_raises_on_unknown(self):
        with self.assertRaises(KeyError):
            mods_all.mod("z")

    def test_misc_re_exports(self):
        self.assertEqual(misc.truncate("hello", max_chars=3, suffix=""), "hel")
        self.assertEqual(misc.slugify("Hello World!"), "hello-world")
        self.assertEqual(misc.coalesce(None, "fallback"), "fallback")

    def test_more_utils_re_exports(self):
        self.assertEqual(more_utils.truncate("hello", max_chars=3, suffix=""), "hel")
        self.assertEqual(more_utils.slugify("Hello World!"), "hello-world")
        self.assertEqual(more_utils.coalesce(None, "fallback"), "fallback")

    def test_tools_extra_re_exports(self):
        self.assertEqual(tools_extra.truncate("hello", max_chars=3, suffix=""), "hel")
        self.assertEqual(tools_extra.slugify("Hello World!"), "hello-world")
        self.assertEqual(tools_extra.coalesce(None, "fallback"), "fallback")
        self.assertEqual(tools_extra.safe_get({"a": 1}, "a"), 1)


if __name__ == "__main__":
    unittest.main()