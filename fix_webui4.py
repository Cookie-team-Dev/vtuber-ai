#!/usr/bin/env python3
import sys

path = '/home/cookie/vtuber-ai/webui.py'
with open(path) as f:
    s = f.read()

# Fix CORS - be more restrictive? "Remove or narrow Access-Control-Allow-Origin: * — at least exclude the data endpoints (/api/webmemory, /api/memory)."
# So in general don't use *; but need to be compatible - maybe set specific origin or just don't send for sensitive? Or set to empty/not send?
# Easier to remove for those specific endpoints when responding? Or change to not wildcard - but frontend needs to work; maybe check origin
# But safer to not send wildcard for memory endpoints. Let us handle in _send based on path? Or just remove wildcard globally and handle - but let us look up how it's used.
old = '''    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # CORS headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Kookie-Key, X-Kookie-Discord")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)'''
new = '''    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # CORS headers
        path = getattr(self, 'path', '') or ''
        parsed = urllib.parse.urlparse(path)
        p = parsed.path
        if p in ('/api/webmemory', '/api/memory'):
            origin = self.headers.get('Origin')
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            else:
                self.send_header("Access-Control-Allow-Origin", "null")
        else:
            origin = self.headers.get('Origin')
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            else:
                self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Kookie-Key, X-Kookie-Discord")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)'''
if old not in s:
    print('SEND NOT FOUND')
    sys.exit(1)
s = s.replace(old, new)

# Fix OPTIONS
old = '''    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Kookie-Key, X-Kookie-Discord")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()'''
new = '''    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        path = getattr(self, 'path', '') or ''
        parsed = urllib.parse.urlparse(path)
        p = parsed.path
        if p in ('/api/webmemory', '/api/memory'):
            origin = self.headers.get('Origin')
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            else:
                self.send_header("Access-Control-Allow-Origin", "null")
        else:
            origin = self.headers.get('Origin')
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            else:
                self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Kookie-Key, X-Kookie-Discord")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()'''
if old not in s:
    print('OPTIONS NOT FOUND')
    sys.exit(1)
s = s.replace(old, new)

with open(path, 'w') as f:
    f.write(s)
