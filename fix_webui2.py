#!/usr/bin/env python3
import sys

path = '/home/cookie/vtuber-ai/webui.py'
with open(path) as f:
    s = f.read()

# Add helpers
helpers = '''MAX_BODY_BYTES = 1024 * 1024


def _require_key(handler):
    """Require a valid issued key or family key."""
    kind = handler._key_kind()
    if kind is None:
        handler._unauthorized()
        return False
    return True


def _json_body(handler, max_bytes=MAX_BODY_BYTES):
    """Parse JSON body safely, return dict or None."""
    try:
        length = int(handler.headers.get("Content-Length", 0) or 0)
    except Exception:
        length = 0
    if length <= 0:
        return {}
    if length > max_bytes:
        handler._send(413, "body too large", "text/plain")
        return None
    try:
        raw = handler.rfile.read(length)
    except Exception:
        return None
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _is_private_or_special_host(hostname):
    try:
        # Resolve
        try:
            infos = socket.getaddrinfo(hostname, None)
        except Exception:
            return True  # fail closed
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except Exception:
                return True
            if (
                ip.is_loopback
                or ip.is_private
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return True
            # CGNAT 100.64.0.0/10
            if isinstance(ip, ipaddress.IPv4Address):
                if ip in ipaddress.ip_network("100.64.0.0/10"):
                    return True
        return False
    except Exception:
        return True


'''

# Insert after _remember_message
marker = '''    # Also save to Kookie\'s main memory for training
    if role == "user":
        memory.remember(f"User: {content}")
    elif role == "assistant":
        memory.remember(f"AI: {content}")


'''
if marker not in s:
    print('MARKER1 NOT FOUND')
    sys.exit(1)
s = s.replace(marker, marker + helpers)

with open(path, 'w') as f:
    f.write(s)
