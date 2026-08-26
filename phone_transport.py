"""Use a browser — a phone, typically — as the microphone and speaker.

`run_live()` normally talks to a local sound card through `sounddevice`. This
module supplies the same two objects over a WebSocket instead, so the loop, the
barge-in gate, the AEC and everything else run completely unchanged. That seam
is not hypothetical: `tests/bargein_sim.py` already substitutes a virtual
full-duplex device the same way.

Why WebSocket rather than WebRTC. WebRTC is the better answer on a lossy or
remote network — Opus, a real jitter buffer, packet loss concealment — but
`aiortc` brings PyAV and roughly 50 MB with it. On a LAN, raw PCM over a
WebSocket is a couple of hundred kilobytes of dependency and measurably fine.
The one thing WebRTC would have given us for free that matters here is echo
cancellation, and the browser provides that anyway: `getUserMedia`'s
`echoCancellation` constraint cleans the capture regardless of how the audio
is subsequently shipped.

Two constraints shape the design:

* `getUserMedia` requires a **secure context**. Over `http://192.168.x.x`
  every mobile browser refuses microphone access; only `localhost` is exempt.
  So the page must be served over TLS, which is why this module also generates
  a certificate.
* The gate expects frames at a steady 32 ms cadence. Wi-Fi does not deliver
  them that evenly, so the input side buffers and paces rather than handing
  the loop whatever happens to arrive.
"""

from __future__ import annotations

import asyncio
import os
import queue
import subprocess
import threading
import time

import numpy as np

SR = 16000          # what the loop and the VAD expect
TTS_SR = 24000      # what Kokoro produces
FRAME = 512         # 32 ms at 16 kHz

CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")


# --- TLS -------------------------------------------------------------------

def ensure_cert(hosts):
    """Self-signed certificate covering localhost and the LAN address.

    Regenerated only when missing. The IP has to appear in subjectAltName or
    browsers reject it outright — a CN-only certificate has not been accepted
    for years.
    """
    os.makedirs(CERT_DIR, exist_ok=True)
    cert = os.path.join(CERT_DIR, "lithevoice.crt")
    key = os.path.join(CERT_DIR, "lithevoice.key")
    if os.path.isfile(cert) and os.path.isfile(key):
        return cert, key

    alt = ["DNS:localhost", "IP:127.0.0.1"]
    for host in hosts:
        if host and host not in ("127.0.0.1", "localhost", "0.0.0.0"):
            alt.append(f"IP:{host}")
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", key, "-out", cert, "-days", "825",
         "-subj", "/CN=LitheVoice",
         "-addext", "subjectAltName=" + ",".join(alt)],
        check=True, capture_output=True)
    return cert, key


# --- audio device replacements --------------------------------------------

class PhoneOutputStream:
    """Stands in for sounddevice.OutputStream.

    `write()` must block for roughly the duration of the audio, exactly as a
    real device does: the whole barge-in design depends on playback taking
    real time, and a non-blocking write would let a reply "play" instantly and
    make ducking meaningless.
    """

    def __init__(self, hub):
        self.hub = hub

    def write(self, block):
        block = np.asarray(block, np.float32)
        self.hub.send_audio(block)
        time.sleep(len(block) / TTS_SR)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class PhoneInputStream:
    """Stands in for sounddevice.InputStream, pacing frames at 32 ms."""

    def __init__(self, hub, callback):
        self.hub = hub
        self.callback = callback
        self._stop = threading.Event()
        self._thread = None

    def _pump(self):
        next_at = time.perf_counter()
        while not self._stop.is_set():
            frame = self.hub.next_frame()
            self.callback(frame.reshape(-1, 1), FRAME, None, None)
            next_at += FRAME / SR
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.perf_counter()   # fell behind; resynchronise

    def __enter__(self):
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False


class AudioHub:
    """Bridges the WebSocket (async) and the voice loop (threads).

    Input is buffered before the loop is allowed to see it. Handing the gate
    whatever arrived in the last instant would turn network jitter into
    apparent speech onsets and offsets, which is precisely what the gate is
    built to be suspicious of.
    """

    def __init__(self, jitter_ms=120):
        self.jitter = int(jitter_ms * SR / 1000)
        self._in = np.zeros(0, np.float32)
        self._lock = threading.Lock()
        self._primed = False
        self._out = queue.Queue()
        self._loop = None
        self.connected = threading.Event()
        self.frames_in = 0
        self.frames_out = 0
        self.underruns = 0

    # -- called from the websocket side ---------------------------------
    def push_mic(self, samples):
        with self._lock:
            self._in = np.concatenate([self._in, samples])
            if not self._primed and len(self._in) >= self.jitter:
                self._primed = True
            # Never let a slow consumer grow this without bound.
            cap = self.jitter * 8
            if len(self._in) > cap:
                self._in = self._in[-cap:]

    def take_playback(self, timeout=0.02):
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- called from the voice loop -------------------------------------
    def next_frame(self):
        with self._lock:
            if not self._primed or len(self._in) < FRAME:
                self.underruns += 1
                return np.zeros(FRAME, np.float32)
            frame = self._in[:FRAME]
            self._in = self._in[FRAME:]
            self.frames_in += 1
            return frame.astype(np.float32)

    def send_audio(self, block):
        self.frames_out += 1
        self._out.put(np.asarray(block, np.float32))


# --- websocket server ------------------------------------------------------

def serve(hub, host, port, ssl_context, token=None, log=print):
    """Run the audio WebSocket in its own thread with its own event loop."""
    import websockets

    async def handler(ws):
        # websockets >= 12 exposes the request line on ws.request; older
        # releases put it on ws.path. Reading only ws.path silently yields ""
        # on current versions, which makes every token check fail.
        request = getattr(ws, "request", None)
        path = getattr(request, "path", None) or getattr(ws, "path", "") or ""
        if token and f"t={token}" not in path:
            await ws.close(code=4003, reason="token required")
            return
        log("[phone] connected")
        hub.connected.set()

        async def to_phone():
            while True:
                block = await asyncio.to_thread(hub.take_playback)
                if block is None:
                    continue
                pcm = np.clip(block, -1.0, 1.0)
                await ws.send((pcm * 32767).astype("<i2").tobytes())

        sender = asyncio.create_task(to_phone())
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    samples = np.frombuffer(message, dtype="<i2")
                    hub.push_mic(samples.astype(np.float32) / 32768.0)
        finally:
            sender.cancel()
            hub.connected.clear()
            log("[phone] disconnected")

    async def main():
        async with websockets.serve(handler, host, port, ssl=ssl_context,
                                    max_size=2 ** 20):
            await asyncio.Future()

    threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()


def install(hub):
    """Replace the sounddevice module for this process."""
    import sys
    import types
    mod = types.ModuleType("sounddevice")
    mod.InputStream = lambda *a, callback=None, **k: PhoneInputStream(hub, callback)
    mod.OutputStream = lambda *a, **k: PhoneOutputStream(hub)
    mod.query_devices = lambda *a, **k: []
    sys.modules["sounddevice"] = mod
    return mod
