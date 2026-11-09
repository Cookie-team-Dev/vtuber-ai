"""Tests for controller.check_email — the IMAP inbox reader wired to Discord."""
import unittest
from unittest import mock

import controller


class CheckEmailTests(unittest.TestCase):
    def test_returns_hint_when_unconfigured(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            for key in ("KOOKIE_EMAIL_HOST", "KOOKIE_EMAIL_USER", "KOOKIE_EMAIL_PASSWORD"):
                import os
                os.environ.pop(key, None)
            result = controller.check_email()
        self.assertIn("not configured", result)

    def test_returns_hint_when_only_host_set(self):
        with mock.patch.dict("os.environ", {"KOOKIE_EMAIL_HOST": "imap.example.com"}, clear=False):
            result = controller.check_email()
        self.assertIn("not configured", result)

    def test_handles_imap_failure(self):
        env = {"KOOKIE_EMAIL_HOST": "imap.example.com",
               "KOOKIE_EMAIL_USER": "me@example.com",
               "KOOKIE_EMAIL_PASSWORD": "secret"}
        with mock.patch.dict("os.environ", env, clear=False):
            with mock.patch("imaplib.IMAP4_SSL", side_effect=ConnectionError("boom")):
                result = controller.check_email()
        self.assertIn("Could not check email", result)
        self.assertIn("boom", result)

    def test_no_unread_returns_clean_message(self):
        env = {"KOOKIE_EMAIL_HOST": "imap.example.com",
               "KOOKIE_EMAIL_USER": "me@example.com",
               "KOOKIE_EMAIL_PASSWORD": "secret"}

        class FakeMailbox:
            def __init__(self, *args, **kwargs):
                pass

            def login(self, user, password):
                pass

            def select(self, folder):
                return "OK", [b"0"]

            def search(self, *args):
                return "OK", [b""]

            def fetch(self, *args):
                return "OK", [b""]

            def logout(self):
                pass

        with mock.patch.dict("os.environ", env, clear=False):
            with mock.patch("imaplib.IMAP4_SSL", return_value=FakeMailbox()):
                result = controller.check_email()
        self.assertEqual(result, "No unread email.")

    def test_summarises_unread_messages(self):
        env = {"KOOKIE_EMAIL_HOST": "imap.example.com",
               "KOOKIE_EMAIL_USER": "me@example.com",
               "KOOKIE_EMAIL_PASSWORD": "secret"}
        raw_message = (
            b"From: sender@example.com\r\n"
            b"Subject: Hello there\r\n"
            b"Content-Type: text/plain\r\n\r\n"
            b"This is the body of the message."
        )

        class FakeMailbox:
            def __init__(self, *args, **kwargs):
                self.fetched = False

            def login(self, user, password):
                pass

            def select(self, folder):
                return "OK", [b"1"]

            def search(self, *args):
                return "OK", [b"1"]

            def fetch(self, uid, *args):
                self.fetched = True
                return "OK", [(b"1 (RFC822)", raw_message)]

            def logout(self):
                pass

        with mock.patch.dict("os.environ", env, clear=False):
            with mock.patch("imaplib.IMAP4_SSL", return_value=FakeMailbox()):
                result = controller.check_email()
        self.assertIn("1 unread", result)
        self.assertIn("sender@example.com", result)
        self.assertIn("Hello there", result)
        self.assertIn("This is the body of the message.", result)


if __name__ == "__main__":
    unittest.main()