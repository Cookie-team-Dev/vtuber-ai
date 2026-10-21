#!/usr/bin/env python3
import sys

path = '/home/cookie/vtuber-ai/webui.py'
with open(path) as f:
    s = f.read()

# Add imports
old = '''import hashlib
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer'''
new = '''import hashlib
import hmac
import html
import ipaddress
import json
import os
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer'''
if old not in s:
    print('IMPORTS NOT FOUND')
    sys.exit(1)
s2 = s.replace(old, new)

# Fix 5: Remove hardcoded family key default
old = '''WEB_FAMILY_KEY = os.environ.get(
    "KOOKIE_WEB_FAMILY_KEY", "__SET_IN_ENV__")'''
new = '''WEB_FAMILY_KEY = os.environ.get("KOOKIE_WEB_FAMILY_KEY", "")'''
if old not in s2:
    print('FAMILY KEY NOT FOUND')
    sys.exit(1)
s2 = s2.replace(old, new)

with open(path, 'w') as f:
    f.write(s2)
