#!/usr/bin/env python3
import sys

path = '/home/cookie/vtuber-ai/webui.py'
with open(path) as f:
    s = f.read()

# Fix _key_kind
old = '''    def _key_kind(self):
        """Which door this key opens: family, stranger, or none.

        Her family key is hers alone, and so is a
        stranger key the owner set. Without one, her
        website stands open and hands out keys of its
        own: one of those opens the stranger door, and
        anything else is not a key of hers.
        """
        supplied = self._supplied_key()
        if WEB_FAMILY_KEY and supplied == WEB_FAMILY_KEY:
            return "family"
        if supplied and _is_issued(supplied):
            return "stranger"
        return None'''
new = '''    def _key_kind(self):
        """Which door this key opens: family, stranger, or none."""
        supplied = self._supplied_key()
        if WEB_FAMILY_KEY and supplied:
            try:
                if hmac.compare_digest(supplied, WEB_FAMILY_KEY):
                    return "family"
            except Exception:
                pass
        if supplied and _is_issued(supplied):
            return "stranger"
        return None'''
if old not in s:
    print('KEY_KIND NOT FOUND')
    sys.exit(1)
s = s.replace(old, new)

# Fix writing keys file with secure permissions
old = '''        try:
            with open(KEYS_FILE, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=1)
        except OSError:
            pass'''
new = '''        try:
            fd = os.open(KEYS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, (json.dumps(store, indent=1) + "\\n").encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            pass'''
if old not in s:
    print('KEYS WRITE NOT FOUND')
    sys.exit(1)
s = s.replace(old, new)

with open(path, 'w') as f:
    f.write(s)
