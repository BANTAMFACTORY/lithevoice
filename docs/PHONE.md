# Using a Phone as the Microphone and Speaker

`--phone` turns any browser on your network into LitheVoice's microphone and
speaker. The voice loop, the barge-in gate, the AEC and everything else run
completely unchanged; only the audio device is replaced.

Measured over Wi-Fi from a phone: **494–754 ms voice-to-voice**, first audio
117–180 ms, with ducking and barge-in intact.

---

## 1. Running it

```bash
./run.sh --phone --barge-in
```

Startup prints two URLs:

```
Phone:  https://192.168.1.42:7860/phone?t=<token>&ap=7861
Web UI: http://127.0.0.1:7860/?t=<token>
```

On the phone: open the `Phone:` URL, **accept the certificate warning**, tap
**Start talking**. Headphones or AirPods give the cleanest interruption.

The desktop dashboard keeps working at the same time — it is the same server,
so you can watch the transcript and latency on the computer while talking
through the phone.

| Flag | Meaning |
|---|---|
| `--phone` | browser as audio device; implies `--web-host 0.0.0.0`, HTTPS and a token |
| `--audio-port N` | audio WebSocket port (default: web port + 1) |
| `--web-token X` | fix the token instead of generating one |
| `--web-host` | bind address on its own, without the phone transport |

---

## 2. Why it is built this way

### WebSocket, not WebRTC

WebRTC is the better answer on a lossy or remote network — Opus, a real jitter
buffer, packet-loss concealment, NAT traversal. But `aiortc` brings PyAV and
roughly 50 MB with it, in a project whose lite profile is a 390 MB
environment. On a LAN, raw PCM over a WebSocket costs about 2 MB of dependency
(`websockets`) and measures fine.

The one thing WebRTC would have given us for free that actually matters here
is echo cancellation, and the browser supplies that anyway: `getUserMedia`'s
`echoCancellation` constraint cleans the capture regardless of how the audio
is subsequently shipped. That is what makes the phone's own speaker usable
rather than only headphones.

If this ever needs to work over the open internet, swap the transport for
WebRTC — the seam is the same one described below, and the policy layer does
not change.

### TLS is not optional

`getUserMedia` requires a **secure context**. Over `http://192.168.x.x` every
mobile browser refuses microphone access outright; only `localhost` is exempt.
So `--phone` generates a self-signed certificate covering localhost and the
detected LAN address:

```
X509v3 Subject Alternative Name:
    DNS:localhost, IP Address:127.0.0.1, IP Address:192.168.1.42
```

The IP has to be in `subjectAltName` — a CN-only certificate has not been
accepted by browsers for years. The certificate is written to `certs/` and
reused; delete it to regenerate (e.g. after changing networks).

### The audio device seam

`PhoneInputStream` and `PhoneOutputStream` stand in for `sounddevice` at
exactly the seam `tests/bargein_sim.py` already proved swappable. Two details
are load-bearing:

* **`write()` blocks for the audio's duration**, like a real device. A
  non-blocking write would let a reply "play" instantly, which would make
  ducking and holding meaningless.
* **Input is buffered ~120 ms before the loop sees it.** Wi-Fi does not
  deliver frames on an even 32 ms cadence, and handing the gate whatever
  arrived in the last instant would turn network jitter into apparent speech
  onsets — precisely what the gate is built to be suspicious of.

---

## 3. Security

Binding off loopback exposes a control surface that can interrupt, switch
persona and **shut the runtime down**, and it streams your conversation. So
whenever the host is not loopback:

* a token is generated and required — query string, cookie or
  `X-LitheVoice-Token` header, compared with `hmac.compare_digest`;
* requests arriving **from the machine itself** skip the token, so
  `http://localhost:7860` keeps working for the person running it;
* the same-origin check for control POSTs is widened to include the detected
  LAN address, and nothing else.

This is plain TLS with a shared secret and a self-signed certificate. It is
appropriate for your own network. It is not appropriate for the open internet:
no CA, no per-user identity, no rate limiting.

---

## 4. Known rough edges

- **Certificate friction.** Every new device gets a browser warning to tap
  through. Playwright refuses the certificate outright
  (`ERR_CERT_AUTHORITY_INVALID`), and on some iOS versions a cert-warning
  origin can still block `getUserMedia`. The clean fix is `mkcert`: a
  locally-trusted CA whose root you install once on the phone, after which it
  is an ordinary padlocked site. Not currently used.
- **No auto-reconnect.** If the socket drops, the page stops; tap Start again.
- **No "server not listening" indicator** distinct from "connected but idle".
- **One client at a time.** A second connection shares the same audio hub, and
  the result is not defined. Multi-user is a different project — see
  [SCALING.md](SCALING.md).
- **`ScriptProcessorNode`** is deprecated. It is used because it behaves
  identically on iOS Safari and Android Chrome without shipping a worklet
  file; an `AudioWorklet` would be the modern replacement.
