"""Tiny event-stream web UI server for realtime.py — stdlib only, no deps.

Serves webui.html at / and a Server-Sent-Events firehose at /events.
realtime.py publishes JSON events (state changes, transcripts, per-block TTS
spectrum bands, mic levels) through Bus.publish(); every connected browser
gets them live. Controls return through same-origin JSON POSTs while audio
stays in the Python process.
"""
import json
import math
import os
import queue
import hmac
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.html")
PHONE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phone.html")


class Bus:
    def __init__(self):
        self._clients = []
        self._lock = threading.Lock()
        self._snapshot = {}  # last config/state event, replayed to new clients
        self.on_command = None  # set by realtime.py: fn(dict) for POST /control

    def publish(self, **ev):
        data = json.dumps(ev, separators=(",", ":"))
        with self._lock:
            if ev.get("type") in ("config", "state"):
                self._snapshot[ev["type"]] = data
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    dead.append(q)  # stalled client; drop it
            for q in dead:
                self._clients.remove(q)

    def subscribe(self):
        q = queue.Queue(maxsize=512)
        with self._lock:
            for data in self._snapshot.values():
                q.put(data)
            self._clients.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)


def _make_handler(bus, token=None, hosts=None):
    allowed_hosts = set(hosts or ()) | {"127.0.0.1", "localhost", "::1"}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request console spam
            pass

        def do_GET(self):
            if not self._authorized():
                # The reason phrase lands in the HTTP status line, which is
                # latin-1 only; anything fancier raises inside send_error and
                # the client gets a dropped connection instead of a 403.
                self.send_error(403, "Forbidden",
                                "token required: open the URL printed at "
                                "startup, which includes ?t=")
                return
            route = self.path.split("?")[0]
            if route == "/phone":
                try:
                    with open(PHONE_PATH, "rb") as f:
                        body = f.read()
                except OSError:
                    self.send_error(500, "phone.html missing")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                if token:
                    self.send_header("Set-Cookie",
                                     f"lithevoice_token={token}; Path=/; "
                                     "SameSite=Strict")
                self.end_headers()
                self.wfile.write(body)
            elif route in ("/", "/index.html"):
                try:
                    with open(HTML_PATH, "rb") as f:
                        body = f.read()
                except OSError:
                    self.send_error(500, "webui.html missing")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                if token:
                    # So the page's own fetch()/EventSource calls carry it.
                    self.send_header("Set-Cookie",
                                     f"lithevoice_token={token}; Path=/; "
                                     "SameSite=Strict")
                self.end_headers()
                self.wfile.write(body)
            elif self.path.split("?")[0] == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = bus.subscribe()
                try:
                    while True:
                        try:
                            data = q.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        self.wfile.write(b"data: " + data.encode() + b"\n\n")
                        self.wfile.flush()
                except (ConnectionAbortedError, BrokenPipeError, OSError):
                    pass
                finally:
                    bus.unsubscribe(q)
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)

        def _authorized(self):
            """Token in the query string, a cookie, or a header.

            Only enforced when the server is bound off loopback. The dashboard
            can interrupt, switch persona and shut the runtime down, and it
            streams the conversation, so putting it on a network without a
            secret would hand all of that to anyone who can reach the port.
            """
            if not token:
                return True
            # A request arriving from this machine already has full access to
            # it; requiring a token there only breaks http://localhost:7860
            # for the person running the thing.
            peer = (self.client_address or ("",))[0]
            if peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
                return True
            supplied = self.headers.get("X-LitheVoice-Token")
            if not supplied:
                query = parse_qs(urlsplit(self.path).query)
                supplied = (query.get("t") or [None])[0]
            if not supplied:
                cookie = self.headers.get("Cookie") or ""
                for part in cookie.split(";"):
                    name, _, value = part.strip().partition("=")
                    if name == "lithevoice_token":
                        supplied = value
                        break
            return bool(supplied) and hmac.compare_digest(supplied, token)

        def do_POST(self):
            if not self._authorized():
                self.send_error(403, "Forbidden", "token required")
                return
            if self.path != "/control":
                self.send_error(404)
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self.send_error(415, "application/json required")
                return
            origin = self.headers.get("Origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    origin_port = parsed.port
                except ValueError:
                    parsed, origin_port = None, None
                if (parsed is None or parsed.scheme != "http"
                        or parsed.hostname not in allowed_hosts
                        or origin_port != self.server.server_port):
                    self.send_error(403, "foreign origin rejected")
                    return
            fn = bus.on_command
            if fn is None:
                self.send_error(503, "pipeline still loading")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0 or n > 8192:
                    self.send_error(413, "invalid control payload size")
                    return
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self.send_error(400)
                return
            if not isinstance(cmd, dict) or cmd.get("action") not in {
                    "voice", "speed", "persona", "interrupt", "shutdown",
                    "enroll"}:
                self.send_error(400, "unknown control action")
                return
            action, value = cmd["action"], cmd.get("value")
            if action in {"voice", "persona"} and (
                    not isinstance(value, str) or not value or len(value) > 128):
                self.send_error(400, "invalid control value")
                return
            if action == "speed" and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                self.send_error(400, "invalid speed")
                return
            try:
                fn(cmd)
            except Exception as e:  # surface handler errors to the browser
                self.send_error(500, str(e)[:200])
                return
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def start(port=7860, host="127.0.0.1", token=None, hosts=None,
          ssl_context=None):
    """Start the UI server on <host>:<port> (tries a few ports up).

    host defaults to loopback. Binding anywhere else exposes a control surface
    that can interrupt, reconfigure and shut down the runtime, so callers are
    expected to pass a token as well.
    Returns (bus, actual_port)."""
    bus = Bus()
    last = None
    for p in range(port, port + 10):
        try:
            srv = ThreadingHTTPServer((host, p),
                                      _make_handler(bus, token, hosts))
            if ssl_context is not None:
                srv.socket = ssl_context.wrap_socket(srv.socket,
                                                     server_side=True)
        except OSError as e:
            last = e
            continue
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return bus, p
    raise RuntimeError(f"no free port near {port}: {last}")
