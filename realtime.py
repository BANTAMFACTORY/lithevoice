"""LitheVoice realtime voice loop.

Flow:  mic -> Silero VAD (utterance segmentation) -> Parakeet STT
       -> Gemma (llama.cpp, streamed) -> Kokoro TTS (GPU, sentence-streamed)
       -> speaker

Measures real voice-to-voice latency:
    end-of-speech -> transcript ready -> LLM first sentence -> first TTS audio.

Modes:
    python realtime.py                     # live mic loop (half-duplex)
    python realtime.py --direct-audio      # audio straight to the model, no STT gate
    python realtime.py --no-llm            # canned replies (no LLM server)
    python realtime.py --simulate x.wav    # feed a wav instead of the mic
    python realtime.py --simulate x.wav --no-play   # headless: write reply_*.wav

See README.md for the full feature guide.

Notes:
  - llama-server (Gemma E2B Q4_K_M on GPU) is auto-started if nothing is
    listening on --llm-url, always with the fixed gemma.jinja chat template
    (the GGUF's baked-in one forces chain-of-thought). Text mode builds the
    Gemma prompt by hand against /completion; --direct-audio sends utterance
    audio through /v1/chat/completions input_audio parts (mmproj).
  - Half-duplex by default: while speaking a reply it stops listening, then
    flushes the mic. --barge-in / --aec for full duplex.
  - Parakeet STT defaults to CPU int8. Kokoro and llama.cpp automatically use
    CUDA when the installer detects a working NVIDIA GPU, otherwise CPU.

Setup: run scripts/setup.ps1 (Windows) or scripts/setup.sh (Linux) once —
it creates the venv, installs deps, and fetches the LLM + llama.cpp.
See README.md.
"""
import argparse
import base64
import collections
import io
import json
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.request

# Everything is resolved relative to this file, so the package runs from
# wherever it's cloned/copied. Override any of it with environment variables
# if you keep models/llama.cpp elsewhere (e.g. a shared drive).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _lan_address():
    """Best guess at this machine's LAN address, for printing a reachable URL.
    Opening a UDP socket sends nothing; it just asks the routing table which
    interface would be used."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(("192.0.2.1", 9))   # TEST-NET-1, never routed
            return probe.getsockname()[0]
    except OSError:
        return None


def _env(name, legacy_name, default):
    """Read a LitheVoice setting while preserving old DadAI overrides."""
    return os.environ.get(name, os.environ.get(legacy_name, default))


MODELS_DIR = _env("LITHEVOICE_MODELS_DIR", "DADAI_MODELS_DIR",
                  os.path.join(BASE_DIR, "models"))
LLAMA_DIR = _env("LITHEVOICE_LLAMA_DIR", "DADAI_LLAMA_DIR",
                 os.path.join(BASE_DIR, "llama.cpp"))
os.environ.setdefault("HF_HOME", os.path.join(MODELS_DIR, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(MODELS_DIR, "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows' legacy console encoding cannot represent every status symbol used
# by the UI-oriented logs. Replacing an unsupported glyph is preferable to
# crashing during --help or a live turn.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

import numpy as np
import soundfile as sf

SR = 16000            # mic + STT + VAD sample rate
TTS_SR = 24000        # Kokoro output
FRAME = 512           # Silero VAD frame size at 16 kHz (32 ms)
SENTENCE_SPLIT = r"(?<=[.!?])\s+"
# Output device buffer. Big enough to ride out a GIL stall, small
# enough that a barge-in is not followed by a tail of stale audio.
OUTPUT_LATENCY_S = 0.08

CANNED = [
    "Hey bud, good to hear from you. Hope your day is going well.",
    "Got it. Don't forget to drink some water and take a break soon.",
    "Sounds good to me. Let me know if you need anything else.",
    "Alright. I'm proud of you, you know that? Talk soon.",
]

_SPOKEN_STYLE = (
    " Reply out loud in one or two short spoken sentences. Open with a short "
    "sentence. Plain conversational text only: no emojis, no markdown, no "
    "stage directions, no lists."
)

# Personas: system prompt + few-shot example exchanges that TEACH the voice.
# The shots are rendered as fake prior turns in the constant prompt prefix,
# so they are KV-cached at prewarm — character costs zero latency. Keep the
# example replies SHORT (they anchor reply length too).
PERSONAS = {
    "dad": {
        "system": (
            "You are Dad: warm, a little corny, endlessly proud of your kid. "
            "You grill on weekends, fix things in the garage, drink your "
            "coffee black, and have an opinion about everything. Call them "
            "bud or kiddo. Never mention being an AI, a model, or an "
            "assistant — you're just Dad." + _SPOKEN_STYLE),
        "shots": [
            ("Hey dad, how's it going?",
             "Hey bud! Just puttering around the garage, you caught me "
             "mid-coffee. What's going on with you?"),
            ("I'm kind of stressed about work.",
             "Deep breath, kiddo. No job's worth an ulcer — we'll sort it "
             "out one piece at a time."),
            ("What should I eat for dinner?",
             "Grilled cheese and tomato soup, that's a fact of life. Put "
             "garlic powder in the butter, trust your old man."),
            ("Tell me about yourself.",
             "Not much to tell, bud — I like my coffee black and my steaks "
             "medium rare. Ask your mother, she'll tell you I talk about "
             "the lawn too much."),
            ("Tell me something interesting.",
             "Get this: sharks have been around longer than trees. Heard "
             "that at the barbershop and haven't shut up about it since."),
        ],
    },
    "buddy": {
        "system": (
            "You are the user's easygoing best friend since high school. "
            "Casual, funny, a little sarcastic, always real with them. "
            "Never mention being an AI or an assistant." + _SPOKEN_STYLE),
        "shots": [
            ("Hey man.",
             "Yo, there he is! What's the move today?"),
            ("I'm bored.",
             "Same, honestly. Wanna brainstorm something dumb and fun to do?"),
            ("I bombed that interview.",
             "Their loss, dude, seriously. You want to vent or you want "
             "distraction memes?"),
        ],
    },
    "coach": {
        "system": (
            "You are an upbeat personal coach. Encouraging, focused, "
            "action-oriented — always end pointed at the next small win. "
            "Never mention being an AI or an assistant." + _SPOKEN_STYLE),
        "shots": [
            ("I don't feel like working out today.",
             "That's exactly when it counts. Ten minutes, that's all I'm "
             "asking — you'll thank me at minute eleven."),
            ("I finished everything on my list today.",
             "That's what I'm talking about! Stack a few more days like "
             "that and nothing can stop you."),
        ],
    },
    "assistant": {
        "system": ("You are a capable, friendly voice assistant. Direct and "
                   "helpful." + _SPOKEN_STYLE),
        "shots": [],
    },
}

SYSTEM_PROMPT = PERSONAS["dad"]["system"]

# Personas live as editable text files in personas\ (seeded from the built-ins
# above on first run). File format — first block is the system prompt, then
# each "---" block is one few-shot example ("> " = the user's line):
#
#   You are Dad: warm, a little corny...
#   ---
#   > Hey dad, how's it going?
#   Hey bud! Just puttering around the garage.
#   ---
#   > ...
PERSONA_DIR = _env("LITHEVOICE_PERSONA_DIR", "DADAI_PERSONA_DIR",
                   os.path.join(BASE_DIR, "personas"))
PERSONA_TEMPLATE = (
    "You are NAME: describe who they are, their quirks, strong opinions, and "
    "how they talk. Never mention being an AI, a model, or an assistant."
    + _SPOKEN_STYLE + "\n"
    "---\n"
    "> Hey, how's it going?\n"
    "A short example reply, written in the character's voice.\n"
    "---\n"
    "> Tell me about yourself.\n"
    "Another short reply that shows off the personality. Two or three "
    "examples is plenty; keep them brief, they anchor reply length.\n")


def _persona_to_text(p):
    parts = [p["system"].strip()]
    parts += [f"> {u}\n{a}" for u, a in p["shots"]]
    return "\n---\n".join(parts) + "\n"


def _parse_persona(text):
    blocks = [b.strip() for b in re.split(r"(?m)^---\s*$", text) if b.strip()]
    if not blocks:
        raise ValueError("empty persona file")
    shots = []
    for b in blocks[1:]:
        lines = b.splitlines()
        u = " ".join(l[1:].strip() for l in lines if l.startswith(">"))
        a = " ".join(l.strip() for l in lines if not l.startswith(">")).strip()
        if u and a:
            shots.append((u, a))
    return {"system": blocks[0], "shots": shots}


def load_personas():
    """Load personas\\*.txt; seed the folder from the built-ins on first run."""
    if not os.path.isdir(PERSONA_DIR):
        os.makedirs(PERSONA_DIR)
        for name, p in PERSONAS.items():
            with open(os.path.join(PERSONA_DIR, name + ".txt"), "w",
                      encoding="utf-8") as f:
                f.write(_persona_to_text(p))
    out = {}
    for fn in sorted(os.listdir(PERSONA_DIR)):
        if not fn.endswith(".txt"):
            continue
        try:
            with open(os.path.join(PERSONA_DIR, fn), encoding="utf-8") as f:
                out[fn[:-4]] = _parse_persona(f.read())
        except Exception as e:
            log(f"[personas] skipping {fn}: {e}")
    return out

# Curated Kokoro voices (American English). Any other Kokoro voice id also
# works via --voice.
VOICES = [
    ("af_heart",   "female, warm (default)"),
    ("af_bella",   "female, bright"),
    ("af_nicole",  "female, soft-spoken"),
    ("af_sarah",   "female, calm"),
    ("af_sky",     "female, clear"),
    ("am_michael", "male, warm — the dad pick"),
    ("am_adam",    "male, deep"),
    ("am_eric",    "male, mellow"),
    ("am_fenrir",  "male, energetic"),
    ("am_puck",    "male, playful"),
    ("am_onyx",    "male, deep narrator"),
    ("am_liam",    "male, casual"),
]

def _find_gguf(want_mmproj):
    """Locate the LLM weights / mmproj GGUF in MODELS_DIR. Env vars
    LITHEVOICE_MODEL / LITHEVOICE_MMPROJ always win (the legacy DADAI names
    are also accepted). Set them if you keep a different quant or fine-tune.
    Returns None if nothing matches — callers give a clear error at the point
    of use rather than failing here at import time."""
    if want_mmproj:
        env = _env("LITHEVOICE_MMPROJ", "DADAI_MMPROJ", None)
    else:
        env = _env("LITHEVOICE_MODEL", "DADAI_MODEL", None)
    if env:
        return env
    if not os.path.isdir(MODELS_DIR):
        return None
    hits = []
    for root, _, files in os.walk(MODELS_DIR):
        for filename in files:
            lower = filename.lower()
            if lower.endswith(".gguf") and ("mmproj" in lower) == want_mmproj:
                hits.append(os.path.join(root, filename))
    if not hits:
        return None

    def preference(path):
        name = os.path.basename(path).lower()
        preferred_quant = "q4_k_m" in name if not want_mmproj else "bf16" in name
        preferred_family = "gemma-4-e2b" in name
        return (not preferred_family, not preferred_quant, name, path.lower())

    return sorted(hits, key=preference)[0]


LLM_URL = "http://127.0.0.1:8080"
LLM_MODEL = _find_gguf(want_mmproj=False)
LLM_MMPROJ = _find_gguf(want_mmproj=True)  # optional — only direct-audio needs it
LLM_SERVER_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"
LLM_SERVER_EXE = os.path.join(LLAMA_DIR, "bin", LLM_SERVER_NAME)
SETUP_HINT = ("scripts\\setup.ps1" if os.name == "nt" else "scripts/setup.sh")
LLM_LOG = os.path.join(LLAMA_DIR, "server.log")
LLM_PID = os.path.join(LLAMA_DIR, "server.pid")
LLM_JINJA = os.path.join(BASE_DIR, "gemma.jinja")


def log(msg):
    print(msg, flush=True)


# --- web UI event bus (see webui.py; set by main() when the UI is on) -------
WEB = None


def web(**ev):
    """Publish an event to the browser UI. No-op when the UI is off."""
    if WEB is not None:
        WEB.publish(**ev)


_BAND_EDGES = np.geomspace(80, 8000, 25)  # 24 log-spaced bands for the orb


def _tts_bands(blk):
    """Coarse log-magnitude spectrum of one 100 ms playback block — drives
    the web UI's voice visualization (client applies smoothing + auto-gain)."""
    spec = np.abs(np.fft.rfft(blk * np.hanning(len(blk))))
    freqs = np.fft.rfftfreq(len(blk), 1.0 / TTS_SR)
    idx = np.searchsorted(freqs, _BAND_EDGES)
    return [round(float(np.log1p(spec[idx[i]:idx[i + 1]].mean()
                                 if idx[i] < idx[i + 1] else 0.0)), 3)
            for i in range(24)]


class Models:
    def __init__(self, tts_device="auto", stt_device="auto", voice="af_heart",
                 speed=1.0, tts_backend="torch", vad_backend="torch"):
        import torch
        from silero_vad import load_silero_vad, VADIterator
        import onnx_asr
        from kokoro import KPipeline

        if tts_device == "auto":
            tts_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif tts_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "--tts-device cuda was requested, but PyTorch cannot use CUDA. "
                f"Use --tts-device auto or rerun {SETUP_HINT}.")
        if stt_device == "auto":
            # CPU int8 remains faster than the much larger fp32 CUDA graph on
            # the reference system, even when a GPU is present.
            stt_device = "cpu"

        log(f"Loading models...  (TTS on {tts_device}, STT on {stt_device})")
        self.voice = voice
        self.speed = speed
        self.torch = torch
        if vad_backend == "onnx":
            from lite_backends import OnnxSilero
            self.vad_model = OnnxSilero()
        else:
            self.vad_model = load_silero_vad()
        self.VADIterator = VADIterator
        if stt_device == "cuda":
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "--stt-device cuda was requested, but ONNX Runtime has no "
                    f"CUDAExecutionProvider. Rerun {SETUP_HINT} on an NVIDIA system.")
            # GPU STT must use the fp32 model (int8's quantized ops fall back
            # off the CUDA EP and run SLOWER than plain CPU — measured).
            # onnxruntime-gpu==1.22 finds its CUDA 12 / cuDNN 9 DLLs in
            # torch's bundled lib dir.
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            _lib_var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            os.environ[_lib_var] = (
                torch_lib + os.pathsep + os.environ.get(_lib_var, ""))
            # The fp32 graph is an explicit optional setup download because it
            # adds roughly 2.5 GB and was slower than CPU int8 in measurements.
            local = os.path.join(MODELS_DIR, "parakeet-fp32")
            if not os.path.isfile(os.path.join(local, ".complete")):
                raise RuntimeError(
                    f"GPU Parakeet was not installed. Rerun {SETUP_HINT} "
                    "with --include-gpu-stt, or keep --stt-device auto.")
            if hasattr(os, "add_dll_directory"):
                self._torch_dll_dir = os.add_dll_directory(torch_lib)
            self.stt = onnx_asr.load_model(
                "nemo-parakeet-tdt-0.6b-v2",
                path=local,
                quantization=None,
                providers=[("CUDAExecutionProvider", {"device_id": 0}),
                           "CPUExecutionProvider"])
        else:
            # CPU int8 is the fastest CPU STT config measured. Explicit
            # providers also silence onnxruntime's TensorRT error spam.
            local = os.path.join(MODELS_DIR, "parakeet-int8")
            self.stt = onnx_asr.load_model(
                "nemo-parakeet-tdt-0.6b-v2",
                path=local if os.path.isfile(os.path.join(local, ".complete")) else None,
                quantization="int8",
                providers=["CPUExecutionProvider"])
        if tts_backend == "onnx":
            # Torch-free synthesis. Same yield contract as KPipeline, so
            # speak_stream() is unaffected. CPU only by construction.
            from lite_backends import OnnxKokoro
            self.tts = OnnxKokoro()
            tts_device = "cpu"
        else:
            self.tts = KPipeline(lang_code="a", device=tts_device)
        self.tts_device = tts_device
        self.stt_device = stt_device
        self._sync = torch.cuda.synchronize if tts_device == "cuda" else (lambda: None)
        # warm both up so first real turn isn't penalized
        self.stt.recognize(np.zeros(SR, np.float32), sample_rate=SR)
        for _ in self.tts("Warm up.", voice=self.voice):
            pass
        self._sync()
        log("Models ready.\n")

    def transcribe(self, audio_f32):
        return self.stt.recognize(audio_f32, sample_rate=SR).strip()

    def speak(self, text, play_stream, out_path=None, cancel=None, ref=None,
              duck=None, playing=None, hold=None):
        """Synthesize sentence-streamed; play chunks as produced. Returns
        (seconds_to_first_audio, total_audio_seconds, cancelled)."""
        first, dur, cut, _ = self.speak_stream(
            iter([text]), play_stream, out_path=out_path, cancel=cancel,
            ref=ref, duck=duck, playing=playing, hold=hold)
        return first, dur, cut

    def speak_stream(self, texts, play_stream, out_path=None, cancel=None,
                     ref=None, voice=None, duck=None, playing=None, hold=None):
        """Like speak(), but takes an ITERATOR of text segments (e.g. sentences
        arriving from a streaming LLM) and synthesizes each as it becomes
        available. Returns (seconds_to_first_audio, total_audio_seconds,
        cancelled, spoken_text).

        cancel: optional threading.Event — playback is written in ~100 ms
        blocks and aborts within a block of the event being set (barge-in).
        ref: optional RefBuffer — every played block is also pushed as the
        echo-cancellation reference signal.
        duck: optional threading.Event — while set, blocks are attenuated to
        DUCK_GAIN instead of stopping. This is the reversible half of
        barge-in: the reply steps back for a backchannel and comes back up if
        the user was not actually taking the floor.
        playing: optional threading.Event — set once the first block has been
        written, so the caller can tell "generating" from "audible".
        """
        first = None
        chunks = []
        spoken = []
        gain = 1.0        # current playback gain, ramped not stepped
        block = TTS_SR // 10  # 100 ms playback blocks -> fast cancellation
        t0 = time.perf_counter()
        for text in texts:
            if not text.strip() or (cancel is not None and cancel.is_set()):
                continue  # keep draining the iterator (closes the LLM stream)
            spoken.append(text.strip())
            for _, _, audio in self.tts(text, voice=voice or self.voice,
                                        speed=self.speed,
                                        split_pattern=SENTENCE_SPLIT):
                audio = np.asarray(audio, dtype=np.float32)
                if first is None:
                    self._sync()
                    first = time.perf_counter() - t0
                chunks.append(audio)
                if play_stream is not None:
                    for i in range(0, len(audio), block):
                        if cancel is not None and cancel.is_set():
                            break
                        # HOLD: go silent without losing our place, so the
                        # reply can resume verbatim if the interruption turns
                        # out to have been a backchannel.
                        while hold is not None and hold.is_set():
                            if cancel is not None and cancel.is_set():
                                break
                            time.sleep(0.02)
                        if cancel is not None and cancel.is_set():
                            break
                        blk = audio[i:i + block]
                        # Ramp between duck levels instead of stepping. A gain
                        # change applied as a step part-way through a waveform
                        # is a discontinuity, and a discontinuity is a click.
                        target = (DUCK_GAIN if (duck is not None
                                                and duck.is_set()) else 1.0)
                        if gain != target:
                            blk = (blk * np.linspace(gain, target, len(blk),
                                                     dtype=np.float32))
                            gain = target
                        elif gain != 1.0:
                            blk = blk * gain
                        blk = np.ascontiguousarray(blk, dtype=np.float32)
                        if ref is not None:
                            # Push BEFORE writing. write() blocks until the
                            # device has room, so pushing afterwards makes the
                            # reference lag the echo it is supposed to predict
                            # — which quietly disables both the AEC and the
                            # double-talk test. Push what was actually played:
                            # the ducked signal is the real echo reference.
                            ref.push(blk)
                        play_stream.write(blk)
                        if playing is not None:
                            playing.set()
                        if WEB is not None:  # voice viz, ~10 Hz
                            web(type="tts", bands=_tts_bands(blk),
                                rms=round(float(np.sqrt(np.mean(blk ** 2))), 4))
                if cancel is not None and cancel.is_set():
                    break
        if not chunks:
            return first, 0.0, bool(cancel and cancel.is_set()), ""
        full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if out_path:
            sf.write(out_path, full, TTS_SR)
        return (first, len(full) / TTS_SR, bool(cancel and cancel.is_set()),
                " ".join(spoken))


def _clean_spoken(text):
    # Strip template-token debris (this fine-tune can emit malformed variants
    # like <end_of_of_turn> that slip past the stop strings) and markdown
    # emphasis — none of it should reach the TTS.
    text = re.sub(r"<[^>]*>?", "", text)
    return text.replace("*", "").replace("_", " ")


def sentence_stream(pieces):
    """Group a stream of text fragments (LLM deltas) into sentences, yielding
    each sentence as soon as its terminator + following whitespace arrives."""
    buf = ""
    for piece in pieces:
        buf += piece
        while True:
            m = re.search(SENTENCE_SPLIT, buf)
            if not m:
                break
            yield _clean_spoken(buf[:m.start()])
            buf = buf[m.end():]
    if buf.strip():
        yield _clean_spoken(buf)


class LLM:
    """Streaming client for llama-server running Gemma E2B.

    Talks to the raw /completion endpoint with a hand-built Gemma prompt
    (system folded into the first user turn, stop on <end_of_turn>). This
    GGUF's baked-in chat template injects a chain-of-thought ("Thinking
    Process:") through /v1/chat/completions — fatal for voice latency — and
    llama.cpp's --chat-template gemma preset mis-tokenizes this custom vocab,
    so the manual template is the reliable path (measured: 0 thinking tokens,
    ~55 ms TTFT with the system-prompt KV prefix cached).

    Auto-starts llama-server with automatic CPU/GPU layer placement if nothing
    answers on `url`.
    """
    MAX_TURNS = 12       # history cap; oldest dropped beyond this
    N_PREDICT = 200      # hard cap per reply (system prompt asks for 1-2 sentences)

    def __init__(self, url=LLM_URL, system=SYSTEM_PROMPT, shots=(),
                 autostart=True, direct=False, mmproj_gpu=None, restart=False,
                 llm_device="auto"):
        """direct: audio-in mode — utterance audio goes straight to the model
        via /v1/chat/completions (needs the fixed jinja chat template), no
        transcription step. mmproj_gpu: where the audio/vision encoder runs
        (None -> GPU when direct, else CPU; only applies when THIS process
        starts the server — use restart=True to force new server args)."""
        self.url = url.rstrip("/")
        self.system = system
        self.shots = list(shots)  # few-shot (user, reply) pairs — voice anchor
        self.history = []         # text mode: [(user, assistant), ...]
        self.audio_history = []   # direct mode: [(wav_b64, assistant), ...]
        self.direct = direct
        self.mmproj_gpu = direct if mmproj_gpu is None else mmproj_gpu
        self.llm_device = llm_device
        self.gpu_layers = {"auto": "auto", "cpu": "0", "gpu": "all"}[llm_device]
        self._process = None
        self._log_handle = None
        if direct and not LLM_MMPROJ:
            raise RuntimeError(
                "--direct-audio needs an mmproj GGUF but none was found in "
                f"{MODELS_DIR} (a file with 'mmproj' in its name). Either "
                "place one there or set LITHEVOICE_MMPROJ to its path.")
        if restart and self._healthy():
            self._kill_server()
        if not self._healthy():
            if not autostart:
                raise RuntimeError(f"no llama-server at {self.url}")
            self._start_server()
        elif direct and not self._template_ok():
            # direct mode needs the fixed chat template; a stale server with
            # the GGUF's broken baked-in template would inject CoT thinking
            log("llama-server is up but with the broken baked-in chat "
                "template — restarting it with gemma.jinja...")
            self._kill_server()
            self._start_server()
        self._prewarm()
        mmproj_note = (f"; mmproj offload {'enabled' if self.mmproj_gpu else 'disabled'}"
                      if LLM_MMPROJ else "; no mmproj")
        log(f"LLM ready at {self.url} (Gemma E2B; device={self.llm_device}{mmproj_note}"
            + (", DIRECT AUDIO mode)" if direct else ")"))

    # -- server management ------------------------------------------------
    def _healthy(self, timeout=2):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=timeout) as r:
                return json.load(r).get("status") == "ok"
        except Exception:
            return False

    def _template_ok(self):
        """True if the server is using our fixed gemma.jinja chat template."""
        try:
            with urllib.request.urlopen(self.url + "/props", timeout=5) as r:
                return "first_user_prefix" in json.load(r).get("chat_template", "")
        except Exception:
            return False

    def _kill_server(self):
        pid = self._process.pid if self._process is not None else None
        if pid is None:
            try:
                pid = int(open(LLM_PID, encoding="ascii").read().strip())
            except (OSError, ValueError):
                raise RuntimeError(
                    "The llama-server at this URL was not started by LitheVoice. "
                    "Stop it manually or choose another --llm-url.")
        if os.name == "nt":
            try:
                listing = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    text=True, errors="replace")
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"could not inspect llama-server PID {pid}") from exc
            if '"llama-server.exe"' not in listing.lower() or f'"{pid}"' not in listing:
                raise RuntimeError(
                    f"Refusing to terminate PID {pid}: it is not llama-server.exe.")
            subprocess.check_call(["taskkill", "/F", "/PID", str(pid)],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        else:
            # Same guarantee as the Windows branch: a recorded PID may have
            # been recycled by an unrelated process since it was written.
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as cmdline_file:
                    argv0 = cmdline_file.read().split(b"\0")[0].decode(
                        "utf-8", "replace")
            except FileNotFoundError:
                argv0 = ""  # already gone; nothing to terminate
            else:
                if os.path.basename(argv0) != LLM_SERVER_NAME:
                    raise RuntimeError(
                        f"Refusing to terminate PID {pid}: it is not "
                        f"{LLM_SERVER_NAME}.")
                os.kill(pid, 15)
        while self._healthy(timeout=1):
            time.sleep(0.5)
        self._process = None
        try:
            os.remove(LLM_PID)
        except FileNotFoundError:
            pass

    def _start_server(self):
        if not os.path.isfile(LLM_SERVER_EXE):
            raise RuntimeError(
                f"{LLM_SERVER_NAME} not found at {LLM_SERVER_EXE}. Run "
                f"{SETUP_HINT} first (or point LITHEVOICE_LLAMA_DIR at an "
                "existing llama.cpp build).")
        if not LLM_MODEL:
            raise RuntimeError(
                f"No LLM .gguf found in {MODELS_DIR}. Run {SETUP_HINT} "
                "to fetch one, or set LITHEVOICE_MODEL to a GGUF path.")
        mmproj_note = (f", mmproj offload {'enabled' if self.mmproj_gpu else 'disabled'}"
                      if LLM_MMPROJ else "")
        log(f"Starting llama-server (Gemma E2B, device={self.llm_device}{mmproj_note})...")
        os.makedirs(LLAMA_DIR, exist_ok=True)
        logf = open(LLM_LOG, "w")
        port = self.url.rsplit(":", 1)[-1]
        # -np 1: single slot, so every turn reuses the same KV prefix cache
        # (4 default slots rotated -> TTFT crept 45->450 ms live).
        # gemma.jinja: proper Gemma template; the GGUF's baked-in one forces
        # CoT thinking on the chat endpoint (needed for direct audio mode).
        args = [LLM_SERVER_EXE, "-m", LLM_MODEL,
                "-ngl", self.gpu_layers, "-c", "4096", "-np", "1",
                "--jinja", "--chat-template-file", LLM_JINJA,
                "--host", "127.0.0.1", "--port", port]
        if LLM_MMPROJ:
            args += ["--mmproj", LLM_MMPROJ]
            if not self.mmproj_gpu:
                args.append("--no-mmproj-offload")
        # CUDA runtime DLLs (cudart64_12/cublas64_12/cudnn*) aren't bundled
        # with the plain llama.cpp release zip. scripts/download_models.py
        # extracts the matching cudart zip into bin\ (same folder as the
        # exe — Windows' DLL search checks there first), which covers a
        # fresh install with no CUDA toolkit around. As a second line of
        # defense (this is also what proved out during development), also
        # put torch's bundled CUDA libs on PATH — torch is already a
        # dependency (Kokoro/Parakeet need it), so this always resolves
        # regardless of whether the cudart zip extraction succeeded. If
        # llama-server silently falls back to CPU (~10 tok/s instead of
        # ~80+, visible in server.log's "eval time" line), a missing CUDA
        # runtime DLL is the first thing to check.
        # On Linux the release tarball keeps libggml*.so beside the binary and
        # the loader finds them via RPATH, so only the torch CUDA libraries are
        # worth adding — and there only for a GPU-enabled llama.cpp build.
        env = os.environ.copy()
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[var] = torch_lib + os.pathsep + env.get(var, "")
        except ImportError:
            pass
        self._log_handle = logf
        self._process = subprocess.Popen(
            args, stdout=logf, stderr=subprocess.STDOUT, env=env,
            cwd=os.path.dirname(LLM_SERVER_EXE),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with open(LLM_PID, "w", encoding="ascii") as pid_file:
            pid_file.write(str(self._process.pid) + "\n")
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._healthy():
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}; see {LLM_LOG}")
            time.sleep(1)
        self._process.terminate()
        raise RuntimeError(f"llama-server did not come up; see {LLM_LOG}")

    def _prewarm(self):
        # Process the constant prefix (system + shots) once so the first real
        # turn hits the KV prefix cache (~500 ms -> ~55 ms TTFT). Direct mode
        # prewarms through the chat endpoint so the rendered prefix matches.
        if self.direct:
            self._post_chat({"messages": self._messages(), "max_tokens": 1,
                             "cache_prompt": True})
        else:
            self._post({"prompt": self._prompt_prefix(), "n_predict": 1,
                        "cache_prompt": True})

    # -- prompting ---------------------------------------------------------
    # Gemma has no system role: the system prompt is folded into the first
    # user turn, and the few-shot examples are rendered as prior exchanges.
    # system + shots form a CONSTANT prefix (prewarmed into the KV cache, and
    # unaffected by history-cap trimming), followed by the rolling history.

    def _render(self, turns):
        p = ""
        for i, (u, a) in enumerate(turns):
            uu = f"{self.system}\n\n{u}" if i == 0 else u
            p += (f"<start_of_turn>user\n{uu}<end_of_turn>\n"
                  f"<start_of_turn>model\n{a}<end_of_turn>\n")
        return p

    def _prompt_prefix(self):
        if self.shots:
            return self._render(self.shots)
        return f"<start_of_turn>user\n{self.system}\n\n"

    def _prompt(self, user):
        turns = self.shots + self.history
        uu = user if turns else f"{self.system}\n\n{user}"
        return (self._render(turns)
                + f"<start_of_turn>user\n{uu}<end_of_turn>\n<start_of_turn>model\n")

    def _post(self, payload, stream=False, path="/completion"):
        req = urllib.request.Request(
            self.url + path,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=None if stream else 60)

    def _post_chat(self, payload, stream=False):
        return self._post(payload, stream, path="/v1/chat/completions")

    # -- direct audio mode (chat endpoint; audio parts via the mmproj) ------
    def _messages(self, wav_b64=None):
        """system + few-shot text turns + prior audio turns (+ current audio).
        Past audio is resent verbatim: with -np 1 the identical prefix tokens
        hit the KV cache, so history costs nothing to re-encode."""
        def _audio(b64):
            return [{"type": "input_audio",
                     "input_audio": {"data": b64, "format": "wav"}}]
        msgs = [{"role": "system", "content": self.system}]
        for u, a in self.shots:
            msgs += [{"role": "user", "content": u},
                     {"role": "assistant", "content": a}]
        for b64, a in self.audio_history:
            msgs += [{"role": "user", "content": _audio(b64)},
                     {"role": "assistant", "content": a}]
        if wav_b64 is not None:
            msgs.append({"role": "user", "content": _audio(wav_b64)})
        return msgs

    def stream_audio(self, wav_b64, cancel=None, stats=None):
        """Direct mode: yield reply fragments for a spoken utterance (base64
        wav) — the model hears the audio itself; no transcription step."""
        t0 = time.perf_counter()
        resp = self._post_chat(
            {"messages": self._messages(wav_b64), "max_tokens": self.N_PREDICT,
             "temperature": 0.7, "stop": ["<end_of", "<start_of"],
             "cache_prompt": True, "stream": True}, stream=True)
        try:
            for line in resp:
                if cancel is not None and cancel.is_set():
                    break
                s = line.decode("utf-8", "replace").strip()
                if not s.startswith("data:") or s[5:].strip() == "[DONE]":
                    continue
                j = json.loads(s[5:])
                piece = j["choices"][0]["delta"].get("content", "") \
                    if j.get("choices") else ""
                if piece:
                    if stats is not None and "ttft_ms" not in stats:
                        stats["ttft_ms"] = (time.perf_counter() - t0) * 1000
                    yield piece
        finally:
            resp.close()  # disconnect aborts any in-flight generation

    def commit_audio(self, wav_b64, assistant):
        self.audio_history.append((wav_b64, assistant))
        if len(self.audio_history) > self.MAX_TURNS:
            del self.audio_history[: len(self.audio_history) - self.MAX_TURNS]

    def stream(self, user, cancel=None, stats=None):
        """Yield reply text fragments for `user`. stats (optional dict) gets
        'ttft_ms'. Checks `cancel` between fragments and aborts generation
        server-side by closing the connection."""
        t0 = time.perf_counter()
        resp = self._post({"prompt": self._prompt(user), "n_predict": self.N_PREDICT,
                           "temperature": 0.7,
                           # prefix stops also catch this fine-tune's malformed
                           # template tokens (e.g. <end_of_of_turn>)
                           "stop": ["<end_of", "<start_of"],
                           "cache_prompt": True, "stream": True}, stream=True)
        try:
            for line in resp:
                if cancel is not None and cancel.is_set():
                    break
                s = line.decode("utf-8", "replace").strip()
                if not s.startswith("data:"):
                    continue
                j = json.loads(s[5:])
                piece = j.get("content", "")
                if piece:
                    if stats is not None and "ttft_ms" not in stats:
                        stats["ttft_ms"] = (time.perf_counter() - t0) * 1000
                    yield piece
                if j.get("stop"):
                    break
        finally:
            resp.close()  # disconnect aborts any in-flight generation

    def commit(self, user, assistant):
        """Record a completed exchange (what was actually spoken)."""
        self.history.append((user, assistant))
        if len(self.history) > self.MAX_TURNS:
            del self.history[: len(self.history) - self.MAX_TURNS]

    def set_system(self, system, shots=()):
        """Swap the persona: new system prompt + few-shot examples, fresh
        history, re-prewarm the new constant prefix."""
        self.system = system
        self.shots = list(shots)
        self.history = []
        self.audio_history = []
        self._prewarm()


class RefBuffer:
    """FIFO of the audio we are playing (the echo reference), resampled to the
    mic rate. speak() pushes each block as it's written to the device — which
    is BEFORE it comes out of the speakers — so pulled reference always leads
    the mic echo; the adaptive filter models the (device + acoustic) delay."""

    def __init__(self):
        self._chunks = collections.deque()
        self._offset = 0

    def push(self, audio, sr=TTS_SR):
        if sr != SR:
            n = int(round(len(audio) * SR / sr))
            audio = np.interp(np.linspace(0, 1, n, endpoint=False),
                              np.linspace(0, 1, len(audio), endpoint=False),
                              audio).astype(np.float32)
        self._chunks.append(np.asarray(audio, np.float32))

    def pull(self, n):
        out = np.zeros(n, np.float32)
        i = 0
        while i < n and self._chunks:
            c = self._chunks[0]
            take = min(n - i, len(c) - self._offset)
            out[i:i + take] = c[self._offset:self._offset + take]
            i += take
            self._offset += take
            if self._offset >= len(c):
                self._chunks.popleft()
                self._offset = 0
        return out


class EchoCanceller:
    """Acoustic echo canceller: partitioned-block frequency-domain NLMS
    (the classic Speex-style approach) + residual echo suppression, pure numpy.

    process(mic_frame, ref_frame) -> cleaned mic frame. When the reference is
    silent (bot not speaking) it returns the mic unchanged after one cheap
    energy check, so idle cost is ~zero. Filter: 8 partitions x 512 samples =
    covers 256 ms of device+room echo path; converges within ~1-2 s of the
    first reply and the learned path persists across turns."""

    K = 8           # partitions
    MU = 0.5        # NLMS step
    BETA = 4.0      # residual suppression aggressiveness
    BOOT_FRAMES = 31  # ~1 s of unconditional adaptation at startup

    def __init__(self):
        bins = FRAME + 1  # rfft bins for FFT size 2*FRAME
        self.W = np.zeros((self.K, bins), np.complex128)   # filter weights
        self.X = np.zeros((self.K, bins), np.complex128)   # ref FFT history
        self.Px = np.full(bins, 1e-8)                      # ref power estimate
        self.last_ref = np.zeros(FRAME, np.float32)
        self._zeros = np.zeros(FRAME)
        self.boot = 0  # frames adapted during the bootstrap window
        self._mic_e = self._out_e = 0.0  # live telemetry accumulators
        self._n = 0

    def erle_db(self):
        """Mean echo reduction (dB) over frames processed since last call.
        None if the bot hasn't spoken. ~0 dB live = the filter isn't modeling
        the real path (suspect mic DSP/AGC); >15 dB = working as designed."""
        if self._n == 0:
            return None
        v = 10 * np.log10((self._mic_e + 1e-12) / (self._out_e + 1e-12))
        self._mic_e = self._out_e = 0.0
        self._n = 0
        return v

    def process(self, mic, ref):
        if np.max(np.abs(ref)) < 1e-6 and np.max(np.abs(self.last_ref)) < 1e-6:
            self.last_ref = ref
            return mic  # bot silent -> passthrough

        # overlap-save: FFT of [previous ref block, current ref block]
        Xf = np.fft.rfft(np.concatenate([self.last_ref, ref]))
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = Xf
        self.last_ref = ref

        # echo estimate = sum over partitions, take valid half
        y = np.fft.irfft((self.W * self.X).sum(axis=0))[FRAME:]
        e = mic - y.astype(np.float32)

        # NLMS weight update with gradient constraint; skip during strong
        # near-end activity (crude double-talk protection so the user's voice
        # doesn't corrupt the learned echo path). BOOTSTRAP: the filter starts
        # at zero, so y==0 and the y-based gate can only open if the mic is
        # quieter than 1e-9 — true for a noise-free synthetic mic, NEVER true
        # for a real one (live, the AEC sat frozen at zero forever and the bot
        # barged itself). First ~1 s of far-end activity adapts uncondition-
        # ally (the bot's opening reply is echo-only in practice); if the user
        # happens to talk over that first second the path mis-learns slightly
        # and recovers through the normal gate afterwards.
        self.Px = 0.9 * self.Px + 0.1 * np.abs(Xf) ** 2
        # Only ever adapt against a genuinely active reference (> ~-40 dBFS):
        # near-silent ref frames make the normalized gradient divide by ~zero
        # power and poison the weights with noise (measured: ERLE went
        # NEGATIVE when bootstrap adapted on barely-audible ref frames).
        ref_active = np.mean(ref ** 2) > 1e-4
        if ref_active and (self.boot < self.BOOT_FRAMES or
                           np.mean(mic ** 2) < 8.0 * np.mean(y ** 2) + 1e-9):
            self.boot += 1 if self.boot < self.BOOT_FRAMES else 0
            Ef = np.fft.rfft(np.concatenate([self._zeros, e]))
            G = self.MU * np.conj(self.X) * Ef / (self.K * self.Px + 1e-8)
            g = np.fft.irfft(G, axis=1)
            g[:, FRAME:] = 0  # constrain to causal half
            self.W += np.fft.rfft(g, axis=1)

        # residual echo suppression: per-bin Wiener-style gain keyed to the
        # echo estimate (linear AEC alone leaves audible/VAD-visible residue)
        Es = np.fft.rfft(np.concatenate([self._zeros, e]))
        Ys = np.fft.rfft(np.concatenate([self._zeros, y]))
        gain = np.abs(Es) ** 2 / (np.abs(Es) ** 2 + self.BETA * np.abs(Ys) ** 2 + 1e-10)
        out = np.fft.irfft(gain * Es)[FRAME:].astype(np.float32)
        # full-band duck on echo-dominated frames: cancelling shapes the mic
        # noise floor with speech-like modulation that Silero can read as
        # faint speech (measured p up to 0.66 on pure residue). If most of
        # the frame's energy was removed (echo-dominated, no near-end), push
        # the leftovers below the VAD's radar. Double-talk keeps e ~ mic, so
        # user speech is never ducked.
        if np.mean(e ** 2) < 0.25 * np.mean(mic ** 2):
            out *= 0.3
        self._mic_e += np.mean(mic ** 2)
        self._out_e += np.mean(out ** 2)
        self._n += 1
        return out


BARGE_DEBUG = os.environ.get("LITHEVOICE_BARGE_DEBUG") == "1"

INTERRUPT = threading.Event()  # one-shot interrupt from the web UI button
ENROLL_REQUEST = threading.Event()  # web UI / CLI asked to (re)learn the voice
TURN_ACTIVE = threading.Event()  # generation or playback currently in flight


_X11 = None  # (libX11, display, keycode) once probed; False when unavailable


def _x11_keyboard(keysym=0x0060):  # XK_grave — the ` / ~ key
    """Bind libX11 and resolve the grave keycode once, or return None."""
    global _X11
    if _X11 is not None:
        return _X11 or None
    _X11 = False
    if not os.environ.get("DISPLAY"):
        log("  key interrupt needs an X11 display; use the web UI interrupt "
            "button instead (Wayland sessions can run Xwayland).")
        return None
    try:
        import ctypes
        import ctypes.util
        path = ctypes.util.find_library("X11")
        if not path:
            raise OSError("libX11 not found")
        xlib = ctypes.CDLL(path)
        xlib.XOpenDisplay.restype = ctypes.c_void_p
        xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        display = xlib.XOpenDisplay(None)
        if not display:
            raise OSError(f"cannot open display {os.environ['DISPLAY']}")
        xlib.XKeysymToKeycode.restype = ctypes.c_ubyte
        xlib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        xlib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char * 32]
        keycode = xlib.XKeysymToKeycode(ctypes.c_void_p(display), keysym)
        if not keycode:
            raise OSError("the ` / ~ key is not present in this layout")
        _X11 = (xlib, display, int(keycode))
    except Exception as exc:
        log(f"  key interrupt unavailable ({exc}); use the web UI interrupt "
            "button or --barge-in with a headset.")
        return None
    return _X11


def _key_down(vk=0xC0):
    """True while the ` / ~ key is physically held.

    Both backends are global — they report the physical key state no matter
    which window has focus. Windows uses GetAsyncKeyState (VK_OEM_3 on US
    layouts); X11 uses XQueryKeymap, which needs neither root nor the `input`
    group. Returns False wherever no such backend exists, leaving the web UI
    interrupt button as the way to take the turn back.
    """
    if os.name == "nt":
        try:
            import ctypes
            return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
        except Exception:
            return False
    keyboard = _x11_keyboard()
    if keyboard is None:
        return False
    xlib, display, keycode = keyboard
    try:
        import ctypes
        keys = (ctypes.c_char * 32)()
        xlib.XQueryKeymap(ctypes.c_void_p(display), keys)
        return bool(keys[keycode // 8][0] & (1 << (keycode % 8)))
    except Exception:
        return False


DUCK_GAIN = 0.15  # -16 dB: the reply steps back without vanishing

# Short acknowledgements a listener makes while the other person is still
# talking. They are how people signal "I'm following, keep going" — treating
# them as interruptions is what makes naive barge-in exhausting to talk to.
BACKCHANNEL_WORDS = {
    "mm", "mmm", "mhm", "mmhmm", "mm-hmm", "hmm", "huh", "uh", "uhhuh",
    "uh-huh", "ah", "aha", "oh", "ok", "okay", "yeah", "yep", "yup", "yes",
    "right", "sure", "true", "nice", "cool", "wow", "gotcha", "totally",
    "exactly", "definitely", "absolutely", "i", "see", "got", "it",
}
# Short words that unambiguously DO mean "stop talking", and must take the
# floor even though they are as brief as a backchannel.
FLOOR_TAKING_WORDS = {
    "stop", "wait", "no", "nope", "hold", "pause", "quiet", "enough",
    "actually", "sorry", "but", "shut", "cancel", "nevermind", "never",
}
_WORD_RE = re.compile(r"[a-z0-9'-]+")


def classify_utterance(text, max_words=4):
    """Label a captured utterance: "empty", "backchannel" or "speech".

    Used at the turn boundary to decide whether an utterance that arrived
    while the assistant was speaking actually meant to take the floor. This is
    the one place where cheap semantics beat acoustics: a backchannel and a
    real interruption are indistinguishable by energy, spectrum or VAD score
    (measured in tests/bargein_probe.py — both sit at p=1.00 and the same
    level above the noise floor), but they are trivially different words.
    """
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return "empty"
    if any(w in FLOOR_TAKING_WORDS for w in words):
        return "speech"
    if len(words) <= max_words and all(w in BACKCHANNEL_WORDS for w in words):
        return "backchannel"
    return "speech"


# A reference quieter than this is not meaningfully playing: comparing
# against it produces meaningless ratios (see SpeechAdmit._verdict).
REF_ACTIVE = 1e-3

SPEAKER_REPO = "onnx-community/wespeaker-voxceleb-resnet34-LM"
SPEAKER_FILE = "onnx/model.onnx"
SPEAKER_REVISION = "6a61a1833ff2583aabeba044f5c8221f00b67ceb"
PROFILE_PATH = _env("LITHEVOICE_VOICE_PROFILE", "DADAI_VOICE_PROFILE",
                    os.path.join(MODELS_DIR, "voice_profile.npz"))


class SpeakerVerifier:
    """Decide whether an utterance came from the enrolled talker.

    This is real speaker verification, unlike the spectral match it replaces.
    That earlier version compared L2-normalised log-band energies and could
    not do the job: measured live, it rejected 4 frames out of 429, because
    every human voice scores 0.94-1.00 against every other one on that
    representation — the threshold sat inside the enrolled speaker's own
    spread, so tightening it would have started rejecting the user before it
    rejected anybody else.

    WeSpeaker ResNet34-LM (VoxCeleb) produces a 256-d embedding from ~1 s of
    audio in about 12 ms on four CPU threads. Cosine against the enrolled
    centroid separates cleanly — measured on held-out clips, 0.70-0.81 for the
    same speaker against 0.13 for a different one.

    An embedding needs roughly a second of speech, which is far too slow to
    gate the 96 ms duck. So identity is applied where a second is already
    being spent: at the turn boundary, deciding whether an utterance may take
    the floor. A stranger can still make the reply dip briefly; they cannot
    take a turn.
    """

    def __init__(self, threshold=0.40, min_speech_s=0.45,
                 min_verify_s=0.8):
        self.threshold = threshold
        self.min_speech_s = min_speech_s
        # Rejecting somebody needs more evidence than embedding
        # them does. A ~0.5 s clip scores 0.29 against its own
        # speaker, so vetoing on that little voice would throw
        # away the user's own short 'stop'. Below this much
        # speech the gate abstains and the backchannel/floor-word
        # logic decides instead.
        self.min_verify_s = min_verify_s
        self.centroid = None
        self.enrolled_from = 0
        self._sess = None
        self._kaldi = None
        self._torch = None

    # -- model ------------------------------------------------------------
    def _session(self):
        if self._sess is None:
            import onnxruntime as ort
            import torch
            import torchaudio.compliance.kaldi as kaldi
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(SPEAKER_REPO, SPEAKER_FILE,
                                   revision=SPEAKER_REVISION)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
            opts.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"])
            self._torch, self._kaldi = torch, kaldi
        return self._sess

    @staticmethod
    def voiced(audio, frame=FRAME):
        """Keep the loudest frames. Silence dilutes an embedding badly: a 1.5 s
        clip that is one second of room tone scored 0.43 against its own
        speaker, where the trimmed version scores far higher."""
        n = len(audio) // frame
        if n < 2:
            return audio
        frames = audio[:n * frame].reshape(n, frame)
        energy = (frames ** 2).mean(axis=1)
        keep = energy >= max(float(np.median(energy)), 1e-8)
        if not keep.any():
            return audio
        return frames[keep].reshape(-1)

    def embed(self, audio):
        """L2-normalised 256-d embedding, or None if there is too little voice."""
        sess = self._session()
        voiced = self.voiced(np.asarray(audio, np.float32))
        if len(voiced) < self.min_speech_s * SR:
            return None
        wav = self._torch.from_numpy(voiced).unsqueeze(0) * (1 << 15)
        feats = self._kaldi.fbank(
            wav, num_mel_bins=80, frame_length=25, frame_shift=10,
            dither=0.0, sample_frequency=SR, window_type="hamming",
            use_energy=False)
        feats = feats - feats.mean(dim=0, keepdim=True)   # WeSpeaker uses CMN
        vec = sess.run(None, {"input_features": feats.numpy()[None]})[0][0]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-9 else None

    # -- enrolment --------------------------------------------------------
    def enroll(self, utterances):
        """Build the centroid from one or more enrolment utterances."""
        vecs = [v for v in (self.embed(u) for u in utterances) if v is not None]
        if not vecs:
            return False
        mean = np.mean(vecs, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-9:
            return False
        self.centroid = mean / norm
        self.enrolled_from = len(vecs)
        return True

    @property
    def ready(self):
        return self.centroid is not None

    def score(self, audio):
        """Cosine against the enrolled voice, or None if it cannot be judged."""
        if self.centroid is None:
            return None
        audio = np.asarray(audio, np.float32)
        if len(self.voiced(audio)) < self.min_verify_s * SR:
            return None          # too little voice to accuse anyone
        vec = self.embed(audio)
        return None if vec is None else float(np.dot(vec, self.centroid))

    # -- persistence ------------------------------------------------------
    def save(self, path=None):
        path = path or PROFILE_PATH
        if self.centroid is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, centroid=self.centroid,
                 threshold=np.float32(self.threshold),
                 enrolled_from=np.int32(self.enrolled_from))

    def load(self, path=None):
        path = path or PROFILE_PATH
        try:
            data = np.load(path)
        except (OSError, ValueError):
            return False
        centroid = data.get("centroid") if hasattr(data, "get") else None
        if centroid is None or centroid.shape[-1] != 256:
            return False
        self.centroid = np.asarray(centroid, np.float32)
        self.enrolled_from = int(data["enrolled_from"]) if \
            "enrolled_from" in data else 0
        return True


class SpeechAdmit:
    """Is this 32 ms frame the user talking, as opposed to the room or us?

    One object answers that question for the whole loop, and both consumers —
    turn-taking and barge-in — use the same answer. That matters: measurement
    showed the barge gate was not the main source of pain. A television or a
    desk fan would open a *turn*, the assistant would answer it, and starting
    that answer cancelled whatever it was already saying. Fixing only the
    barge path left the reply dying by a different route.

    Three signals, all a few microseconds per frame:

      1. Silero's speech probability — necessary, nowhere near sufficient.
      2. Frame energy above an adaptive noise floor, estimated as a low
         percentile of the recent past (minimum statistics) rather than by
         averaging quiet frames. That distinction is load-bearing: a floor
         that only learns from non-speech never initialises at all when a
         television is talking continuously, which silently disables the whole
         test exactly when it is needed. A percentile always produces an
         answer, and steady babble simply becomes the floor.
      3. Frame energy above the echo predicted from the playback reference.
         Coupling — how much of what we play comes back — is learned during
         echo-only frames, making this a double-talk detector that works with
         or without the AEC running in front of it.

    speaker_ok: optional callable(frame) -> bool for speaker identification.
    Recorded speech from a television is confident speech on every acoustic
    signal above (measured: 77% of frames at p=1.00, sitting *higher* above
    the noise floor than the real user), so separating "my user" from "a voice
    in the room" is a question of identity, not of level. Left None, the class
    behaves as documented and the noise floor absorbs steady babble.
    """

    def __init__(self, thresh=0.5, snr_db=6.0, echo_margin_db=4.0,
                 speaker_ok=None):
        self.thresh = thresh
        self.snr = 10.0 ** (snr_db / 20.0)
        self.echo_margin = 10.0 ** (echo_margin_db / 20.0)
        self.speaker_ok = speaker_ok
        self.floor = None
        self.coupling = 0.0
        self.playing = False
        self.last_ok = False
        self.reason = "vad"
        self.level = self.ref_level = 0.0
        self._ref_env = collections.deque(maxlen=8)    # ~256 ms of reference
        self._floor_win = collections.deque(maxlen=125)  # ~4 s of room
        self._ratios = collections.deque(maxlen=100)  # ~3 s of coupling
        self._play_hold = 0

    def _verdict(self, p, frame, ref_frame=None):
        e = float(np.sqrt(np.mean(frame ** 2)) + 1e-12)

        ref_e = 0.0
        if ref_frame is not None:
            self._ref_env.append(float(np.sqrt(np.mean(ref_frame ** 2))))
            # Max over the window: the reference leads the echo by the output
            # latency, so an envelope comparison has to tolerate a few frames
            # of skew.
            ref_e = max(self._ref_env) if self._ref_env else 0.0
        # Sticky: on CPU the synthesiser routinely fails to keep ahead of the
        # speaker, so the reference goes briefly empty while the room is still
        # ringing with the last block. Treating those frames as "not playing"
        # would file our own echo away as room noise and skip the echo test on
        # exactly the frames that need it.
        if ref_e > REF_ACTIVE:
            self._play_hold = 12          # ~380 ms
        elif self._play_hold:
            self._play_hold -= 1
        self.playing = ref_e > REF_ACTIVE or self._play_hold > 0

        if not self.playing:
            # Every frame goes in, speech included; the percentile does the
            # rejecting. Echo frames are excluded because the floor is meant
            # to describe the room, not us.
            self._floor_win.append(e)
            if len(self._floor_win) >= 16:
                self.floor = max(1e-5, float(np.percentile(
                    np.fromiter(self._floor_win, np.float32), 20)))
        elif p < 0.35 and ref_e > REF_ACTIVE:
            # Learn coupling only against a genuinely loud reference, and as a
            # quantile of recent observations rather than a running maximum.
            # A max() ratchet is a trap: at the very start and end of a reply
            # the reference is near zero while the room is not, so one frame
            # of ref=1e-4 against ambient 1e-3 yields a ratio of 10 and pins
            # the estimate at its cap forever. On a headset — where the mic
            # genuinely hears none of the reply — that pinned value demanded
            # the user shout 20 dB over the reference before being believed,
            # so barge-in only worked in the gaps between sentences.
            self._ratios.append(e / ref_e)
        if self._ratios:
            # A high quantile of a bounded window, NOT a maximum.
            #
            # Three versions of this line, and the history matters:
            #   1. An unbounded max() ratchet. Pinned at its cap on a headset
            #      and made the gate demand the user shout over the reference.
            #   2. A 90th percentile plus the REF_ACTIVE guard. Fixed that, and
            #      was confirmed good in live use on a headset.
            #   3. max() over a bounded window, to recover an open-speaker
            #      regression seen in simulation.
            # Version 3 was never tried live, and it reintroduced version 1's
            # symptom on a headset: speech rejected as echo, so the start of an
            # interruption went missing. The simulated open-speaker gain was
            # not worth a real regression on real hardware, so this is back to
            # version 2. The remaining open-speaker cost is a small number of
            # echo-triggered turns in scenarios that are already the hardest;
            # `--barge-snr-db` and `--aec` are the levers there.
            self.coupling = float(np.percentile(
                np.fromiter(self._ratios, np.float32), 90))
        else:
            self.coupling = 0.0

        self.level, self.ref_level = e, ref_e
        if p < self.thresh:
            self.reason = "vad"
            return False
        if self.floor is not None and e < self.floor * self.snr:
            self.reason = "floor"
            return False
        if (self.playing and ref_e > REF_ACTIVE
                and e < self.coupling * ref_e * self.echo_margin):
            self.reason = "echo"
            return False
        if self.speaker_ok is not None and not self.speaker_ok(frame):
            self.reason = "speaker"
            return False
        self.reason = "ok"
        return True

    def update(self, p, frame, ref_frame=None):
        self.last_ok = self._verdict(p, frame, ref_frame)
        return self.last_ok


class BargeGate:
    """Decide whether mic activity during playback is the user taking the floor.

    Whether a frame counts as the user speaking is decided by SpeechAdmit;
    this class only decides what to do about a run of such frames. The old
    gate fired on three consecutive Silero frames above 0.5, which measurement
    (tests/bargein_probe.py) shows cannot tell a person from the room: with
    the reply coming back through open speakers, 28% of echo-only frames score
    >= 0.5, so the assistant reliably interrupted itself.

    The response is staged, which is what makes it feel comfortable:

      DUCK   after ~duck_ms — the reply drops to DUCK_GAIN but keeps going.
             Fast enough to feel instant, and because the echo drops by the
             same 16 dB it also sharpens signals 2 and 3.
      HOLD   after ~hold_ms — the reply stops making sound but keeps its
             place. Nothing has been lost yet.
      CANCEL after ~cancel_ms — the turn is genuinely abandoned.

    Only the last stage is destructive, and the perceived responsiveness is
    set by the *first* one. That is the whole trick: the assistant is quiet
    about 140 ms after you open your mouth (measured end to end through
    tests/bargein_sim.py, including the VAD's own onset delay), so it already
    feels like it yielded, which buys the gate more than a second to decide
    whether you actually meant to take the floor. Probe data shows a backchannel is
    acoustically identical to a real interruption, so nothing cheaper than
    time (and, at the turn boundary, the transcript) can separate them.
    """

    FRAME_MS = FRAME * 1000 // SR  # 32 ms

    def __init__(self, duck_ms=96, hold_ms=352, cancel_ms=1400, decay=0.5,
                 grace_ms=250):
        self.duck_ms = duck_ms
        self.hold_ms = hold_ms
        # Cancel is deliberately late. By hold_ms the assistant is already
        # silent, so the user cannot hear the difference, and the reply cannot
        # start being *useful* again until they stop talking anyway. Spending
        # that free time on certainty is why a long "right, sure" no longer
        # destroys a turn.
        self.cancel_ms = cancel_ms
        self.decay = decay
        self.grace_ms = grace_ms
        self.reset()

    def reset(self):
        """New reply: forget the run, but keep what we learned about the room."""
        self.acc_ms = 0.0        # leaky accumulator of qualifying speech
        self.stage = 0           # 0 listening, 1 ducked, 2 held, 3 cancelled
        self.played_ms = 0       # audible reply time, for the grace window

    def update(self, ok, playing=False):
        """Feed one frame's SpeechAdmit verdict.

        Returns "duck", "hold", "cancel", "release" or None.

        playing: True once the reply is actually audible, which is what the
        grace window is measured against — on CPU the first audio can be two
        seconds behind the decision to speak.

        The accumulator leaks rather than resetting. Requiring N *consecutive*
        speech frames looks reasonable and is wrong: "Wait, actually I meant
        next week" has a real pause after the comma, so a consecutive counter
        gives up in the middle of a genuine interruption (measured — the reply
        survived a deliberate barge and only died a second later at the turn
        boundary). Leaking at `decay` tolerates prosodic gaps while still
        demanding that speech dominate the window.
        """
        if playing:
            self.played_ms += self.FRAME_MS

        if ok:
            self.acc_ms += self.FRAME_MS
        else:
            self.acc_ms = max(0.0, self.acc_ms - self.FRAME_MS * self.decay)
            if self.acc_ms == 0.0 and self.stage in (1, 2):
                self.stage = 0
                return "release"
            return None

        # Do not let the opening moments of a reply cancel it: that window is
        # where the echo transient lands and where the coupling estimate is
        # still settling.
        if self.played_ms < self.grace_ms:
            return None
        # Escalate one stage at a time so every step is observable (and, up to
        # the cancel, reversible).
        if self.stage < 3 and self.acc_ms >= self.cancel_ms:
            self.stage = 3
            return "cancel"
        if self.stage < 2 and self.acc_ms >= self.hold_ms:
            self.stage = 2
            return "hold"
        if self.stage < 1 and self.acc_ms >= self.duck_ms:
            self.stage = 1
            return "duck"
        return None

    @property
    def committed(self):
        """True once the gate has actually abandoned the reply."""
        return self.stage >= 3

    def relax(self):
        """The interruption was resolved as not taking the floor: stand down
        without forgetting the room, so the reply can pick up where it left
        off."""
        self.acc_ms = 0.0
        self.stage = 0


def handle_turn(models, audio_f32, idx, play_stream, no_play, parallel=True,
                spec=None, cancel=None, background=False, ref=None, llm=None,
                turn_delay_ms=0.0, duck=None, playing=None, hold=None):
    """Given a captured utterance, transcribe, produce a reply, speak it.

    llm: LLM client — the reply is generated from the transcript and streamed
    sentence-by-sentence into TTS (necessarily serial w.r.t. STT; speculative
    STT still hides most of the transcription inside the silence window).
    llm=None falls back to canned replies.

    parallel=True (canned only): replies don't depend on the transcript, so
    TTS starts immediately at end-of-speech while STT resolves alongside.

    spec: in-flight speculative STT launched when silence began (~200 ms in) —
    by turn-confirm time it is usually already finished, making STT ~free.
    cancel: threading.Event that aborts reply playback (barge-in).
    background: run the reply in a daemon thread and return it immediately, so
    the caller's mic loop keeps running (required for barge-in).
    """
    t_end = time.perf_counter()
    out = None if not no_play else f"reply_{idx}.wav"

    def get_transcript():
        if spec is not None:
            spec["thread"].join()
            return spec["out"]["text"]
        return models.transcribe(audio_f32)

    who = (getattr(llm, "persona_name", None) or "dad") if llm else "dad"

    def _echo_sentences(sents):
        """Pass-through that prints each reply sentence as it is handed to
        TTS — i.e. as it starts being spoken — instead of after playback."""
        label = f"  {who} : "
        for s in sents:
            t = s.strip()
            if t:
                log(label + t)
                if label.strip():
                    web(type="state", state="speak")
                label = " " * len(label)
                web(type="reply", text=t)
            yield s

    def _run_inner():
        llm_stats = {}
        web(type="state", state="think")
        if llm is not None and llm.direct:
            # audio straight to the model — no transcription gate at all
            buf = io.BytesIO()
            sf.write(buf, audio_f32, SR, format="WAV", subtype="PCM_16")
            wav = base64.b64encode(buf.getvalue()).decode()
            t_stt = t_end
            log(f"  you : [{len(audio_f32)/SR:.1f}s audio -> model]")
            web(type="you", text=f"{len(audio_f32)/SR:.1f}s of audio")
            try:
                pieces = llm.stream_audio(wav, cancel=cancel, stats=llm_stats)
                first, dur, cut, reply = models.speak_stream(
                    _echo_sentences(sentence_stream(pieces)), play_stream,
                    out_path=out, cancel=cancel, ref=ref, duck=duck,
                    playing=playing, hold=hold)
                if reply:
                    llm.commit_audio(wav, reply)
            except Exception as e:
                log(f"  [llm error: {e} — canned fallback]")
                reply = CANNED[idx % len(CANNED)]
                log(f"  {who} : {reply}")
                web(type="state", state="speak")
                web(type="reply", text=reply)
                first, dur, cut = models.speak(reply, play_stream, out_path=out,
                                               cancel=cancel, ref=ref,
                                               duck=duck, playing=playing,
                                           hold=hold)
            pipeline = first or 0
            mode = "direct-audio"
            # Parakeet reference for A/B judging (speculative STT already ran
            # during the silence window — this join is free; logged after the
            # reply so it never gates it)
            transcript = get_transcript()
            log(f'  (parakeet heard: "{transcript}")')
            web(type="ref", text=transcript)
        elif llm is not None:
            transcript = get_transcript()
            t_stt = time.perf_counter()
            log(f'  you : "{transcript}"')
            web(type="you", text=transcript)
            try:
                pieces = llm.stream(transcript, cancel=cancel, stats=llm_stats)
                first, dur, cut, reply = models.speak_stream(
                    _echo_sentences(sentence_stream(pieces)), play_stream,
                    out_path=out, cancel=cancel, ref=ref, duck=duck,
                    playing=playing, hold=hold)
                if reply:
                    llm.commit(transcript, reply)
            except Exception as e:
                log(f"  [llm error: {e} — canned fallback]")
                reply = CANNED[idx % len(CANNED)]
                log(f"  {who} : {reply}")
                web(type="state", state="speak")
                web(type="reply", text=reply)
                first, dur, cut = models.speak(reply, play_stream, out_path=out,
                                               cancel=cancel, ref=ref,
                                               duck=duck, playing=playing,
                                           hold=hold)
            pipeline = (t_stt - t_end) + (first or 0)
            mode = "llm"
        elif parallel:
            reply = CANNED[idx % len(CANNED)]
            log(f"  {who} : {reply}")
            web(type="state", state="speak")
            web(type="reply", text=reply)
            stt_result = {}

            def _stt():
                stt_result["text"] = get_transcript()
                stt_result["t"] = time.perf_counter()

            th = threading.Thread(target=_stt)
            th.start()
            first, dur, cut = models.speak(reply, play_stream, out_path=out,
                                           cancel=cancel, ref=ref,
                                           duck=duck, playing=playing,
                                           hold=hold)
            th.join()
            transcript, t_stt = stt_result["text"], stt_result["t"]
            log(f'  you : "{transcript}"   (transcribed while speaking)')
            web(type="you", text=transcript)
            pipeline = first or 0
            mode = "parallel"
        else:
            reply = CANNED[idx % len(CANNED)]
            transcript = get_transcript()
            t_stt = time.perf_counter()
            log(f'  you : "{transcript}"')
            web(type="you", text=transcript)
            log(f"  {who} : {reply}")
            web(type="state", state="speak")
            web(type="reply", text=reply)
            first, dur, cut = models.speak(reply, play_stream, out_path=out,
                                           cancel=cancel, ref=ref,
                                           duck=duck, playing=playing,
                                           hold=hold)
            pipeline = (t_stt - t_end) + (first or 0)
            mode = "serial"

        if cut:
            log("  [reply cut off by barge-in]")
        log(f"  --- latency ({mode}) ---")
        if mode == "direct-audio":
            log("  STT wait         : skipped (audio straight to model)")
        else:
            log(f"  STT wait         : {(t_stt - t_end)*1000:.0f} ms"
                + ("   (speculative — ran during silence window)" if spec else ""))
        if llm_stats.get("ttft_ms") is not None:
            log(f"  LLM first-token  : {llm_stats['ttft_ms']:.0f} ms")
        if first is not None:
            log(f"  first-audio      : {first*1000:.0f} ms"
                + ("   (LLM stream -> first TTS sound)" if llm is not None else ""))
        v2v = pipeline + turn_delay_ms / 1000.0
        if turn_delay_ms:
            log(f"  turn decision    : {turn_delay_ms:.0f} ms")
            log(f"  VOICE-TO-VOICE   : {v2v*1000:.0f} ms   (end-of-speech -> first sound)")
        else:
            log(f"  PIPELINE-TO-AUDIO: {pipeline*1000:.0f} ms   (turn dispatch -> first sound)")
        log(f"  (reply is {dur:.1f}s of audio)\n")
        web(type="stats", mode=mode,
            stt_ms=None if mode == "direct-audio" else (t_stt - t_end) * 1000,
            ttft_ms=llm_stats.get("ttft_ms"),
            first_ms=first * 1000 if first is not None else None,
            turn_ms=turn_delay_ms or None,
            v2v_ms=v2v * 1000, dur_s=dur)
        web(type="state", state="idle")

    def _run():
        TURN_ACTIVE.set()
        try:
            _run_inner()
        finally:
            TURN_ACTIVE.clear()

    if background:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t
    _run()
    return None


def open_output(no_play):
    if no_play:
        return None
    import sounddevice as sd
    # Enough buffer to survive a scheduling hiccup, and no more. "low" asks the
    # backend for the smallest buffer it will give, which on PipeWire is small
    # enough that the GIL — shared with Silero every 32 ms and with Kokoro
    # synthesis — starves the device: measured 36 dropouts of 2-60 ms in 13 s
    # of speech, heard as a crackle. The cost of buffering is that already-
    # queued audio still plays after a barge, so this is deliberately modest
    # rather than generous; it is additive with the ~76 ms duck.
    s = sd.OutputStream(samplerate=TTS_SR, channels=1, dtype="float32",
                        latency=OUTPUT_LATENCY_S)
    s.start()
    return s


def startup_menu(models, llm, personas=None, ask_voice=True, ask_persona=True):
    """Interactive pickers shown at launch (skipped by --voice/--persona/
    --no-menu, in simulate mode, or when stdin isn't a terminal)."""
    if ask_voice:
        log("\n--- voice ---")
        for i, (v, d) in enumerate(VOICES, 1):
            log(f"  {i:2}. {v:12} {d}")
        log("number = select | p<number> = preview out loud | Enter = keep "
            f"{models.voice}")
        play = None
        while True:
            c = input("voice> ").strip().lower().replace(" ", "")
            if not c:
                break
            try:
                if c.startswith("p"):
                    v = VOICES[int(c[1:]) - 1][0]
                    if play is None:
                        play = open_output(False)
                    for _, _, a in models.tts(
                            "Hey bud, this is what I sound like.", voice=v,
                            speed=models.speed):
                        play.write(np.asarray(a, np.float32))
                    continue
                models.voice = VOICES[int(c) - 1][0]
                break
            except (ValueError, IndexError):
                log("  ? enter a number, p<number> to preview, or Enter")
        if play is not None:
            play.stop(); play.close()
        # warm the chosen voice so turn 1 isn't penalized by its first load
        for _ in models.tts("Ready.", voice=models.voice):
            pass

    if ask_persona and llm is not None and personas:
        edited = False
        while True:
            names = list(personas)
            default = "dad" if "dad" in personas else names[0]
            log("\n--- persona ---")
            for i, n in enumerate(names, 1):
                log(f"  {i:2}. {n:12} ({len(personas[n]['shots'])} examples)"
                    + ("  (default)" if n == default else ""))
            log("number = select | e<number> = edit in notepad | n = new | "
                f"Enter = {default}")
            c = input("persona> ").strip().lower().replace(" ", "")
            try:
                if not c:
                    if edited:  # files changed; re-apply so edits take effect
                        p = personas[default]
                        llm.set_system(p["system"], p["shots"])
                        llm.persona_name = default
                    break
                if c == "n" or c.startswith("e"):
                    if c == "n":
                        name = re.sub(r"[^a-z0-9_-]", "",
                                      input("new persona name> ").strip().lower())
                        if not name:
                            continue
                        path = os.path.join(PERSONA_DIR, name + ".txt")
                        if not os.path.exists(path):
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(PERSONA_TEMPLATE.replace("NAME", name))
                    else:
                        path = os.path.join(PERSONA_DIR,
                                            names[int(c[1:]) - 1] + ".txt")
                    log("  notepad open — edit, save, CLOSE it to continue...")
                    subprocess.call(["notepad", path])
                    personas.clear()
                    personas.update(load_personas())
                    edited = True
                    continue
                name = names[int(c) - 1]
                p = personas[name]
                llm.set_system(p["system"], p["shots"])
                llm.persona_name = name
                log(f"persona: {name}")
                break
            except (ValueError, IndexError, KeyError):
                log("  ? enter a number, e<number>, n, or Enter")
    log(f"\nspeaking as {models.voice} @ {models.speed:.2g}x"
        + (", persona ready" if llm is not None else ", canned replies"))


def run_simulate(models, wav_path, no_play, parallel=True, llm=None):
    log(f"[simulate] feeding {wav_path}")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        n = int(round(len(audio) * SR / sr))
        audio = np.interp(np.linspace(0, 1, n, endpoint=False),
                          np.linspace(0, 1, len(audio), endpoint=False), audio).astype(np.float32)
    play = open_output(no_play)
    handle_turn(models, audio, 0, play, no_play, parallel, llm=llm)
    if play:
        play.stop(); play.close()


class SmartTurn:
    """Semantic end-of-turn detection (Pipecat smart-turn v3.2, ONNX on CPU).

    Given utterance audio (16 kHz float32, trailing silence included), returns
    P(turn is complete). Lets us confirm a turn at ~200 ms of silence instead
    of waiting the full min_silence — without cutting people off mid-thought,
    because a mid-thought pause scores "incomplete" and we fall back to the
    plain silence timeout. Feature extraction is vendored in
    whisper_features.py (BSD-2, from pipecat)."""

    REPO, FILE = "pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"
    THRESHOLD = 0.5
    MAX_SECS = 8

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from whisper_features import compute_whisper_log_mel_features

        self._features = compute_whisper_log_mel_features
        path = hf_hub_download(self.REPO, self.FILE)
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"])
        self.predict(np.zeros(SR, np.float32))  # warmup

    def predict(self, audio_16k):
        n = self.MAX_SECS * SR
        a = audio_16k[-n:] if len(audio_16k) > n else np.pad(
            audio_16k, (n - len(audio_16k), 0))  # keep last 8 s, left-pad
        feats = self._features(a, do_normalize=True)[None, ...]
        return float(self._sess.run(None, {"input_features": feats})[0][0].item())


class TurnDetector:
    """Frame-by-frame turn detection on raw Silero probabilities.

    Unlike VADIterator, exposes the moment silence STARTS ("spec"), so STT can
    run speculatively inside the end-of-turn wait window instead of after it.
    Events returned by process(frame):
      "start" — speech onset (utterance begins, preroll included)
      "spec"  — silence began; audio-so-far is final unless speech resumes.
                Fire speculative STT on the payload now.
      "resume"— speech resumed before confirm; discard the speculation.
      "end"   — min_silence reached; turn confirmed. Payload = full utterance.
    """
    START_P, CONT_P = 0.5, 0.35  # hysteresis thresholds
    FRAME_MS = FRAME * 1000 // SR  # 32 ms

    def __init__(self, model, min_silence_ms=500, spec_silence_ms=200,
                 preroll_frames=8):
        import torch
        self.torch = torch
        self.model = model
        self.min_silence_ms = min_silence_ms
        self.spec_silence_ms = spec_silence_ms
        self.preroll = collections.deque(maxlen=preroll_frames)
        self._reset_turn()

    def _reset_turn(self):
        self.in_speech = False
        self.silence_ms = 0
        self.spec_fired = False
        self.speech = []
        self.last_p = 0.0

    def reset(self):
        self.model.reset_states()
        self.preroll.clear()
        self._reset_turn()

    def process(self, frame, admit=True):
        """admit: a bool, or a callable(p) -> bool, saying whether this frame
        is the user talking to us (see SpeechAdmit). Frames that fail are
        treated as silence for turn purposes, so a fan, a keyboard or the
        assistant's own echo cannot open a turn. The raw probability is still
        exposed for the UI. Passing a callable keeps Silero — a stateful RNN —
        stepped exactly once per frame."""
        # OnnxSilero exposes .prob(); the torch model is called directly.
        p = (self.model.prob(frame) if hasattr(self.model, "prob")
             else self.model(self.torch.from_numpy(frame), SR).item())
        self.last_p = p  # exposed for the mic visualisation
        if not (admit(p) if callable(admit) else admit):
            p = 0.0
        if not self.in_speech:
            self.preroll.append(frame)
            if p >= self.START_P:
                self.in_speech = True
                self.speech = list(self.preroll)
                self.silence_ms = 0
                return "start", None
            return None, None

        self.speech.append(frame)
        if p >= self.CONT_P:
            resumed = self.silence_ms > 0 and self.spec_fired
            self.silence_ms = 0
            if resumed:
                self.spec_fired = False
                return "resume", None
            return None, None

        self.silence_ms += self.FRAME_MS
        if self.silence_ms >= self.min_silence_ms:
            utt = np.concatenate(self.speech).astype(np.float32)
            self._reset_turn()
            return "end", utt
        if self.silence_ms >= self.spec_silence_ms and not self.spec_fired:
            self.spec_fired = True
            return "spec", np.concatenate(self.speech).astype(np.float32)
        return None, None


class PTTDetector:
    """Walky-talky turn detection: the held ` / ~ key IS the voice gate.

    Same process(frame) -> (event, payload) contract as TurnDetector, driven
    by the physical key instead of the VAD: "start" when the key goes down
    (preroll included so the first syllable is not clipped), "end" with the
    utterance when it is released. No "spec"/"resume" -- release is
    definitive, so there is no wait window to speculate inside. Frames
    without the key held are never recorded: a voice nearby cannot open a
    turn, which is the point. The admission callback is deliberately ignored
    -- the key is the admission -- but Silero-free operation also means
    last_p is synthesized (1.0 held / 0.0 not) for the orb and gate debug.
    """
    FRAME_MS = FRAME * 1000 // SR

    def __init__(self, preroll_frames=8, min_hold_ms=150):
        self.preroll = collections.deque(maxlen=preroll_frames)
        self.min_hold_frames = max(1, min_hold_ms // self.FRAME_MS)
        self.speech = []
        self.held = False
        self.held_frames = 0    # min-hold counts the KEY, not the preroll
        self.last_p = 0.0
        self.silence_ms = 0

    def reset(self):
        self.preroll.clear()
        self.speech = []
        self.held = False
        self.held_frames = 0
        self.last_p = 0.0

    def process(self, frame, admit=True):
        down = _key_down()
        self.last_p = 1.0 if down else 0.0
        if down and not self.held:
            self.held = True
            self.held_frames = 1
            self.speech = list(self.preroll) + [frame]
            return "start", None
        if down:
            self.held_frames += 1
            self.speech.append(frame)
            return None, None
        if self.held:
            self.held = False
            frames, self.speech = self.speech, []
            held, self.held_frames = self.held_frames, 0
            self.preroll.append(frame)
            if held < self.min_hold_frames:
                return None, None          # an accidental tap is not a turn
            return "end", np.concatenate(frames).astype(np.float32)
        self.preroll.append(frame)
        return None, None


def run_live(models, no_play, parallel=True, min_silence_ms=600,
             spec_silence_ms=200,
             smart_turn=True, barge_in=False, aec=False, llm=None,
             key_barge=False, speaker_lock=False, speaker_ok=None,
             duck_ms=None, cancel_ms=None, snr_db=None,
             speaker_threshold=None, enroll=False, ptt=False):
    import sounddevice as sd

    # min_silence_ms: silence needed to confirm end-of-turn by timeout alone.
    # With smart-turn on, most turns confirm semantically at ~200 ms silence
    # and this is just the fallback for "incomplete"-scored pauses.
    # key_barge: barge-in triggered by HOLDING ` / ~ (or the web ✋ button)
    # instead of the VAD gate. Speaker-safe without AEC: while a reply plays
    # the mic is ignored entirely (echo can't fake turns), and the key hands
    # the floor back to you.
    if ptt:
        det = PTTDetector()
        st = None            # release IS the end of turn; nothing to score
        if os.name != "nt" and _x11_keyboard() is None:
            log("[ptt] no key backend here -- hold-to-talk cannot hear you")
        else:
            log("[ptt] walky-talky mode: HOLD ` / ~ to speak, release to send")
    else:
        det = TurnDetector(models.vad_model, min_silence_ms=min_silence_ms,
                           spec_silence_ms=spec_silence_ms)
        st = SmartTurn() if smart_turn else None
    tune = {k: v for k, v in (("duck_ms", duck_ms),
                              ("cancel_ms", cancel_ms)) if v is not None}
    gate = BargeGate(**tune) if (barge_in and not key_barge) else None
    # One admission verdict per frame, shared by turn-taking and barge-in.
    # Identity is deliberately NOT in the per-frame path: an embedding needs
    # about a second of speech, and the duck happens in 96 ms. speaker_ok
    # remains the seam for a test oracle or a future fast identifier.
    admit = SpeechAdmit(
        speaker_ok=speaker_ok,
        **({"snr_db": snr_db} if snr_db is not None else {})
    ) if gate is not None else None

    verifier = None
    if speaker_lock and gate is not None:
        verifier = SpeakerVerifier(
            **({"threshold": speaker_threshold}
               if speaker_threshold is not None else {}))
        # Warm the ONNX session now: it costs ~200 ms once, and paying it
        # inside the first turn would show up as voice-to-voice latency.
        t_warm = time.perf_counter()
        verifier.embed(np.zeros(SR, np.float32) + 1e-3)
        warm_ms = (time.perf_counter() - t_warm) * 1000
        if enroll or not verifier.load():
            ENROLL_REQUEST.set()
        else:
            log(f"Voice lock ON (profile loaded, {verifier.enrolled_from} "
                f"samples, warm {warm_ms:.0f} ms). Only your voice can "
                f"take a turn.")
    enrolling = []   # utterances collected for the current enrolment
    # The reference is what the gate uses to predict echo, so voice barge-in
    # needs it whether or not the AEC is running in front.
    ref = RefBuffer() if (aec or gate is not None) else None
    canceller = EchoCanceller() if aec else None
    # in-flight reply (barge mode)
    speaking = {"thread": None, "cancel": None, "duck": None, "playing": None,
                "hold": None}
    q = queue.Queue()

    def cb(indata, frames, tinfo, status):
        q.put(indata[:, 0].copy())

    idx = 0
    play = open_output(no_play)
    spec = None  # in-flight speculative STT: {"thread": t, "out": dict}

    def launch_spec(audio):
        out = {}

        def _run():
            t0 = time.perf_counter()
            out["text"] = models.transcribe(audio)
            out["ms"] = (time.perf_counter() - t0) * 1000
        t = threading.Thread(target=_run)
        t.start()
        return {"thread": t, "out": out}

    def reply_active():
        return speaking["thread"] is not None and speaking["thread"].is_alive()

    def peek_transcript(utt):
        """The transcript we already paid for, if the speculative run has it."""
        if spec is not None:
            spec["thread"].join()
            return spec["out"].get("text", "")
        return models.transcribe(utt)

    def resume_reply():
        """Undo a duck/hold and let the in-flight reply carry on."""
        if speaking["hold"] is not None:
            speaking["hold"].clear()
        if speaking["duck"] is not None:
            speaking["duck"].clear()
        if gate is not None:
            gate.relax()
        web(type="duck", on=False)

    def _announce_watcher():
        """Speak BANTAM's narration lines when the floor is free.

        BANTAM appends {"text": ...} JSONL to $BANTAM_VOICE_ANNOUNCE while its
        agent works. The watcher tails the file (from EOF: history is not
        news), waits for a quiet moment -- no reply playing, no capture in
        flight -- and installs itself as speaking["thread"], so every existing
        jig (duck, hold, barge-cancel) applies to narration exactly as to a
        reply. A backlog deeper than 2 is dropped: stale narration is noise.
        """
        apath = os.environ.get("BANTAM_VOICE_ANNOUNCE", "").strip()
        if not apath:
            return
        try:
            aoff = os.path.getsize(apath) if os.path.exists(apath) else 0
        except OSError:
            aoff = 0
        pending = []
        time.sleep(1.5)   # let run_live finish assembling its state
        while True:
            time.sleep(0.4)
            try:
                size = os.path.getsize(apath)
            except OSError:
                continue
            if size < aoff:
                aoff = 0
            if size > aoff:
                try:
                    with open(apath, encoding="utf-8") as fh:
                        fh.seek(aoff)
                        chunk = fh.read()
                    aoff = size
                    for ln in chunk.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            text = str(json.loads(ln).get("text", "")).strip()
                        except ValueError:
                            continue
                        if text:
                            pending.append(text)
                except OSError:
                    continue
            if not pending:
                continue
            if reply_active() or getattr(det, "in_speech", False) \
                    or getattr(det, "held", False):
                continue
            if len(pending) > 2:
                del pending[:len(pending) - 2]
            text = pending.pop(0)
            log(f"  [narrate] {text}")
            if play is None:
                continue
            speaking["cancel"] = threading.Event()
            speaking["duck"] = threading.Event()
            speaking["playing"] = threading.Event()
            speaking["hold"] = threading.Event()

            def _speak(text=text, cancel=speaking["cancel"],
                       duck=speaking["duck"], playing=speaking["playing"],
                       hold=speaking["hold"]):
                try:
                    playing.set()
                    for _, _, blk in models.tts(text, voice=models.voice):
                        if cancel.is_set():
                            return
                        while hold.is_set():
                            time.sleep(0.05)
                            if cancel.is_set():
                                return
                        buf = np.asarray(blk, dtype=np.float32).ravel()
                        if duck.is_set():
                            buf = buf * DUCK_GAIN
                        play.write(buf)
                finally:
                    playing.clear()
            speaking["thread"] = threading.Thread(target=_speak, daemon=True)
            speaking["thread"].start()
            speaking["thread"].join()

    threading.Thread(target=_announce_watcher, daemon=True).start()

    def finish_turn(utt, why, turn_delay_ms):
        nonlocal spec, idx
        if canceller is not None:
            e = canceller.erle_db()
            if e is not None:  # covers the previous reply's playback
                log(f"  [aec] echo reduction during last reply: {e:.0f} dB")
                web(type="sys", text=f"aec {e:.0f} dB")

        # Does this utterance actually earn the floor? Direct-audio mode has
        # no transcript to consult here and pays for one on the critical path,
        # so it keeps the old unconditional behaviour.
        # --- who is speaking -------------------------------------------
        # Identity resolves here rather than per frame: this is the moment a
        # decision becomes irreversible, and a second of audio is already in
        # hand. The duck may already have fired for whoever spoke; that is by
        # design, and it un-ducks below if they turn out to be someone else.
        if verifier is not None:
            if ENROLL_REQUEST.is_set():
                if len(utt) >= verifier.min_speech_s * SR * 2:
                    enrolling.append(utt)
                    if len(enrolling) < 2:
                        log(f"[enrol] got sample {len(enrolling)}/2 — "
                            "say one more sentence")
                        web(type="sys", text=f"enrol {len(enrolling)}/2")
                    elif verifier.enroll(enrolling):
                        verifier.save()
                        ENROLL_REQUEST.clear()
                        enrolling.clear()
                        log("[enrol] voice learned and saved. Only this voice "
                            "can take a turn now.")
                        web(type="sys", text="voice enrolled")
                    else:
                        enrolling.clear()
                        log("[enrol] could not use those samples — try again, "
                            "a little longer")
                else:
                    log("[enrol] too short — say a full sentence")
                if reply_active():
                    resume_reply()
                spec = None
                det.reset()
                return
            if not verifier.ready:
                log("[locked] no voice enrolled yet — click 'Learn my voice' "
                    "in the dashboard, or restart with --enroll")
                web(type="sys", text="not enrolled")
                if reply_active():
                    resume_reply()
                spec = None
                det.reset()
                return
            t_spk = time.perf_counter()
            similarity = verifier.score(utt)
            spk_ms = (time.perf_counter() - t_spk) * 1000
            if BARGE_DEBUG:
                log(f"  [speaker] {('%.2f' % similarity) if similarity is not None else 'abstain'}"
                    f"  ({spk_ms:.0f} ms)")
            if similarity is not None and similarity < verifier.threshold:
                log(f"[other voice] ignored ({similarity:+.2f} vs "
                    f"{verifier.threshold:+.2f} threshold)")
                web(type="sys", text="different voice ignored")
                if reply_active():
                    resume_reply()
                spec = None
                det.reset()
                return

        heard = ""
        if gate is not None and not (llm is not None and llm.direct):
            heard = peek_transcript(utt)
            kind = classify_utterance(heard)
            if kind == "empty":
                # Fans, keyboards and door clicks reach the VAD but carry no
                # words. Answering them is worse than ignoring them.
                log(f"[ignored] {len(utt)/SR:.1f}s of sound, no words ({why})")
                web(type="sys", text="ignored non-speech")
                if reply_active():
                    resume_reply()
                spec = None
                det.reset()
                return
            if kind == "backchannel" and reply_active() and not gate.committed:
                log(f'[backchannel] "{heard}" — keeping the floor')
                web(type="sys", text=f"backchannel: {heard}")
                resume_reply()
                spec = None
                det.reset()
                return

        log(f"[turn {idx}] captured {len(utt)/SR:.1f}s  ({why})")
        if barge_in:
            if reply_active():        # user talked through the previous reply
                speaking["cancel"].set()
                speaking["thread"].join()
            speaking["cancel"] = threading.Event()
            speaking["duck"] = threading.Event()
            speaking["playing"] = threading.Event()
            speaking["hold"] = threading.Event()
            speaking["thread"] = handle_turn(
                models, utt, idx, play, no_play, parallel, spec=spec,
                cancel=speaking["cancel"], background=True, ref=ref, llm=llm,
                turn_delay_ms=turn_delay_ms, duck=speaking["duck"],
                playing=speaking["playing"], hold=speaking["hold"])
            if gate is not None:
                gate.reset()
        else:
            INTERRUPT.clear()
            handle_turn(models, utt, idx, play, no_play, parallel, spec=spec,
                        cancel=INTERRUPT, llm=llm, turn_delay_ms=turn_delay_ms)
            INTERRUPT.clear()
        spec = None
        idx += 1
        det.reset()
        if not barge_in:
            while not q.empty():                  # half-duplex mic flush
                q.get_nowait()

    if key_barge:
        log("Key barge-in ON: HOLD ` / ~ (or click the web UI's ✋) to cut a "
            "reply off. Mic is ignored while a reply plays — speaker-safe.")
    elif barge_in and aec:
        log("Barge-in ON with echo cancellation: start talking and the reply "
            "ducks; keep going and it stops. A short 'mm-hmm' only dips it.")
    elif barge_in:
        log("Barge-in ON: start talking and the reply ducks; keep going and "
            "it stops. The gate predicts speaker echo from the playback "
            "reference, but a HEADSET (or --aec) is still the safest with "
            "loud open speakers.")
    if verifier is not None and ENROLL_REQUEST.is_set():
        log("Voice lock ON but no profile yet: say a sentence, twice, and "
            "that voice becomes the only one that can take a turn.")
    log("Listening... (speak into the mic; Ctrl+C to quit)\n")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=FRAME, callback=cb):
        was_replying = False
        while True:
            frame = q.get()
            ref_frame = ref.pull(FRAME) if ref is not None else None
            if canceller is not None:
                frame = canceller.process(frame, ref_frame)
            if key_barge:
                replying = reply_active()
                cancelled = (speaking["cancel"] is not None
                             and speaking["cancel"].is_set())
                if replying and not cancelled:
                    if _key_down() or INTERRUPT.is_set():
                        INTERRUPT.clear()
                        speaking["cancel"].set()
                        det.reset()
                        while not q.empty():      # drop buffered echo
                            q.get_nowait()
                        log("[key-barge] interrupted — listening")
                        web(type="barge")
                    was_replying = True
                    continue          # mic ignored while the reply plays
                if was_replying and not cancelled:
                    det.reset()       # reply just ended: drop the echo tail
                    while not q.empty():
                        q.get_nowait()
                    was_replying = False
                    continue
                was_replying = replying
                INTERRUPT.clear()     # stale ✋ clicks don't fire later
            # Silero is a stateful RNN, so it must be stepped exactly once per
            # frame: the admission test is handed in as a callback and runs on
            # the probability the detector just computed.
            event, payload = det.process(
                frame,
                (lambda p: admit.update(p, frame, ref_frame))
                if admit is not None else True)
            if event == "start":
                # Headless front-ends (BANTAM's terminal) need the same onset
                # signal the web orb gets: say it on stdout.
                log("[mic] hearing you...")
            if WEB is not None:  # mic level for the orb, ~31 Hz
                web(type="mic", p=round(det.last_p, 3),
                    rms=round(float(np.sqrt(np.mean(frame ** 2))), 4))
                if event == "start":
                    web(type="state", state="user")
            if gate is not None and reply_active():
                audible = speaking["playing"] is not None and \
                    speaking["playing"].is_set()
                decision = gate.update(admit.last_ok, playing=audible)
                if INTERRUPT.is_set():   # the web ✋ is always an immediate cut
                    INTERRUPT.clear()
                    decision = "cancel"
                if BARGE_DEBUG and audible:
                    gate._dbg = getattr(gate, "_dbg", 0) + 1
                    if gate._dbg % 8 == 0:
                        # Identity is resolved once per turn, not per frame,
                        # so it deliberately does not appear here.
                        log(f"  [gate] {admit.reason:7} p={det.last_p:.2f} "
                            f"lvl={admit.level:.4f} floor={admit.floor or 0:.4f} "
                            f"ref={admit.ref_level:.4f} coup={admit.coupling:.2f} "
                            f"acc={gate.acc_ms:.0f}ms")
                if decision == "duck":
                    speaking["duck"].set()
                    log("  [barge] ducking — you started talking")
                    web(type="duck", on=True, stage="duck")
                elif decision == "hold":
                    speaking["hold"].set()
                    log("  [barge] holding — reply paused")
                    web(type="duck", on=True, stage="hold")
                elif decision == "release":
                    speaking["hold"].clear()
                    speaking["duck"].clear()
                    log("  [barge] released — resuming the reply")
                    web(type="duck", on=False, stage="release")
                elif decision == "cancel":
                    speaking["hold"].clear()
                    speaking["duck"].clear()
                    speaking["cancel"].set()
                    log("[barge-in] you spoke — reply cancelled")
                    web(type="barge")
            if event == "spec":
                # Direct-audio mode: DON'T transcribe speculatively — Parakeet
                # saturating the CPU right when llama-server needs it for
                # audio decode/sampling cost ~250-400 ms of TTFT live. The
                # reference transcript is produced after the reply instead.
                if llm is None or not llm.direct:
                    spec = launch_spec(payload)  # STT inside the wait window
                if st is not None:
                    t0 = time.perf_counter()
                    p = st.predict(payload)
                    st_ms = (time.perf_counter() - t0) * 1000
                    if p > st.THRESHOLD:          # semantically complete —
                        finish_turn(payload,      # don't wait out min_silence
                                    f"smart-turn p={p:.2f} in {st_ms:.0f}ms",
                                    det.silence_ms + st_ms)
                    else:
                        log(f"  (smart-turn p={p:.2f}: incomplete, waiting)")
            elif event == "resume":
                spec = None                       # speech resumed; discard
            elif event == "end":
                timeout_delay = ((min_silence_ms + det.FRAME_MS - 1)
                                 // det.FRAME_MS) * det.FRAME_MS
                finish_turn(payload, "silence timeout", timeout_delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", metavar="WAV", help="feed a wav instead of the mic")
    ap.add_argument("--no-play", action="store_true", help="don't play audio; write reply_*.wav")
    ap.add_argument("--tts-device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Kokoro TTS device (default auto: CUDA when available)")
    ap.add_argument("--stt-device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Parakeet STT device (default auto = CPU int8, fastest on "
                         "CPU; cuda = fp32 model on the GPU, ~2.4 GB download)")
    ap.add_argument("--no-llm", action="store_true",
                    help="canned replies instead of the Gemma LLM (no server needed)")
    ap.add_argument("--llm-bridge", default=os.environ.get("LITHEVOICE_LLM_BRIDGE"),
                    help="MODULE[:CLASS] providing the LLM surface (stream/commit/set_system); "
                         "replaces the built-in Gemma client. Used by BANTAM's voice plugin.")
    ap.add_argument("--llm-url", default=LLM_URL,
                    help="llama-server base URL (auto-started if not reachable)")
    ap.add_argument("--llm-device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="llama.cpp layer placement (default auto fits available hardware)")
    ap.add_argument("--direct-audio", action="store_true",
                    help="send utterance audio straight to the model (E2B "
                         "mmproj) instead of transcribing with Parakeet; "
                         "Parakeet still logs a reference transcript")
    ap.add_argument("--mmproj-device", choices=["cpu", "gpu"], default=None,
                    help="audio/vision encoder device (default: gpu in "
                         "--direct-audio mode, else cpu). Applies when this "
                         "script starts the server — add --restart-llm to "
                         "force it onto a running one")
    ap.add_argument("--restart-llm", action="store_true",
                    help="restart the llama-server previously started by this "
                         "LitheVoice installation with the current flags")
    ap.add_argument("--voice", default=None, metavar="ID",
                    help="Kokoro voice id (e.g. am_michael); skips the "
                         "startup voice menu")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="TTS speaking speed multiplier (default 1.0)")
    ap.add_argument("--persona", default=None, metavar="NAME",
                    help="persona name (a .txt in personas\\); skips the "
                         "startup persona menu")
    ap.add_argument("--system", default=None, metavar="TEXT",
                    help="fully custom system prompt (overrides --persona)")
    ap.add_argument("--no-menu", action="store_true",
                    help="skip the interactive startup menu; use defaults/flags")
    ap.add_argument("--web", action="store_true",
                    help="force the web UI on (it's automatic in live mode)")
    ap.add_argument("--no-web", action="store_true",
                    help="disable the web UI")
    ap.add_argument("--phone", action="store_true",
                    help="use a browser as the microphone and speaker. Serves "
                         "the UI over HTTPS (getUserMedia needs it) and opens "
                         "an audio WebSocket; implies --web-host 0.0.0.0")
    ap.add_argument("--audio-port", type=int, default=None,
                    help="port for the phone audio WebSocket (default: web "
                         "port + 1)")
    ap.add_argument("--web-host", default="127.0.0.1",
                    help="address the dashboard binds to. Default is loopback "
                         "only. Use 0.0.0.0 to reach it from a phone on your "
                         "LAN; a token is then required (see --web-token)")
    ap.add_argument("--web-token", default=None,
                    help="shared secret for a non-loopback dashboard. "
                         "Generated and printed if omitted")
    ap.add_argument("--web-port", type=int, default=7860,
                    help="web UI port (default 7860)")
    ap.add_argument("--serial", action="store_true",
                    help="canned mode only: wait for STT before speaking "
                         "(default: parallel; LLM mode is inherently serial)")
    ap.add_argument("--spec-silence", type=int, default=200,
                    help="ms of silence before smart-turn is consulted. Below "
                         "this, an ordinary mid-sentence pause (a comma runs "
                         "250-300 ms) gets scored as a finished turn and cuts "
                         "the speaker off mid-thought; above it, end-of-turn "
                         "detection costs that much more latency.")
    ap.add_argument("--min-silence", type=int, default=600,
                    help="ms of silence to end a turn by timeout (fallback when "
                         "smart-turn scores the pause incomplete)")
    ap.add_argument("--no-smart-turn", action="store_true",
                    help="disable semantic end-of-turn; use silence timeout only")
    ap.add_argument("--barge-in", action="store_true",
                    help="keep listening while the bot speaks; talking over a "
                         "reply cancels it (use a headset, or add --aec)")
    ap.add_argument("--aec", action="store_true",
                    help="software echo cancellation so open speakers don't "
                         "self-trigger barge-in (implies --barge-in; measured "
                         "only 1-5 dB live on this hardware — prefer "
                         "--barge-key with speakers)")
    ap.add_argument("--barge-duck-ms", type=float, default=None,
                    help="speech needed before the reply ducks (default 96); "
                         "lower feels snappier, higher ignores more noise")
    ap.add_argument("--barge-cancel-ms", type=float, default=None,
                    help="speech needed before the turn is abandoned "
                         "(default 1400); raise it if short acknowledgements "
                         "still cost you a turn")
    ap.add_argument("--barge-snr-db", type=float, default=None,
                    help="how far above the room noise floor speech must sit "
                         "to count (default 6); raise it in a loud room")
    ap.add_argument("--tts-backend", choices=["torch", "onnx"], default="torch",
                    help="synthesis runtime. onnx drops the PyTorch dependency "
                         "(much smaller install, CPU only); see docs/LITE.md")
    ap.add_argument("--lite", action="store_true",
                    help="torch-free profile: ONNX synthesis and ONNX VAD. "
                         "Smaller deployable, CPU only, no speaker lock")
    ap.add_argument("--speaker-lock", action="store_true",
                    help="EXPERIMENTAL: only the enrolled voice may take a "
                         "turn. Off by default — the threshold needs "
                         "calibrating against your own microphone first with "
                         "tests/calibrate_speaker.py, or it will reject you")
    ap.add_argument("--enroll", action="store_true",
                    help="re-learn your voice at startup, replacing any saved "
                         "profile")
    ap.add_argument("--speaker-threshold", type=float, default=None,
                    help="cosine similarity required to accept a turn "
                         "(default 0.40; raise it in a room with similar "
                         "voices, lower it if you are being rejected)")
    ap.add_argument("--ptt", action="store_true",
                    help="walky-talky mode: hold ` / ~ to speak, release to "
                         "send; the VAD never opens a turn on its own")
    ap.add_argument("--barge-key", action="store_true",
                    help="speaker-safe barge-in: HOLD ` / ~ (or click the web "
                         "interrupt button) to cut a reply off; the mic is ignored "
                         "while a reply plays, so no echo issues")
    args = ap.parse_args()

    if args.direct_audio and args.no_llm:
        ap.error("--direct-audio needs the LLM (drop --no-llm)")

    # web UI first, so the page is reachable while models load
    hub = None
    if args.phone:
        # Must happen before anything imports sounddevice, and before the web
        # server starts, because both the TLS context and the audio socket are
        # set up alongside it.
        import phone_transport
        if args.web_host == "127.0.0.1":
            args.web_host = "0.0.0.0"
        hub = phone_transport.AudioHub()
        phone_transport.install(hub)     # sounddevice is now the browser

    if (args.web or not args.simulate) and not args.no_web:
        global WEB
        import webui
        loopback = args.web_host in ("127.0.0.1", "localhost", "::1")
        ssl_context = None
        if args.phone:
            import ssl as _ssl
            import phone_transport
            cert, key = phone_transport.ensure_cert([_lan_address()])
            ssl_context = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(cert, key)
        # A dashboard that can interrupt, reconfigure and shut down the
        # runtime — and that streams the conversation — should not sit
        # unauthenticated on a network.
        token = args.web_token or (None if loopback else secrets.token_urlsafe(16))
        lan_ip = _lan_address() if not loopback else None
        WEB, web_port = webui.start(args.web_port, host=args.web_host,
                                    token=token,
                                    hosts=[lan_ip] if lan_ip else None,
                                    ssl_context=ssl_context)
        if args.phone:
            audio_port = args.audio_port or (web_port + 1)
            actx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
            actx.load_cert_chain(cert, key)
            phone_transport.serve(hub, args.web_host, audio_port, actx,
                                  token=token, log=log)
            scheme = "https"
            suffix = f"?t={token}&ap={audio_port}" if token else f"?ap={audio_port}"
            log(f"Phone: {scheme}://{lan_ip or '127.0.0.1'}:{web_port}/phone{suffix}")
            log("  Open that on the phone, accept the self-signed certificate, "
                "then tap Start talking. Headphones recommended.")
        web(type="state", state="loading")
        if loopback:
            log(f"Web UI: http://127.0.0.1:{web_port}")
        else:
            suffix = f"/?t={token}" if token else ""
            log(f"Web UI: http://127.0.0.1:{web_port}{suffix}")
            if lan_ip:
                log(f"        http://{lan_ip}:{web_port}{suffix}   "
                    f"<- open this on your phone")
            log("  NOTE: this dashboard can interrupt, switch persona and "
                "stop the runtime, and it streams your conversation. The "
                "token above is the only thing protecting it, and the link "
                "is plain HTTP on your LAN — treat it as trusted-network only.")
    personas = load_personas() if not args.no_llm else {}
    if args.system:
        system, shots = args.system, ()
    else:
        if args.persona and args.persona not in personas:
            ap.error(f"unknown persona {args.persona!r}; available: "
                     + ", ".join(personas) + f"  (add a .txt in {PERSONA_DIR})")
        name = args.persona or ("dad" if "dad" in personas
                                else next(iter(personas), None))
        p = personas.get(name, {"system": SYSTEM_PROMPT, "shots": []})
        system, shots = p["system"], p["shots"]
    mmproj_gpu = None if args.mmproj_device is None else args.mmproj_device == "gpu"
    # --llm-bridge: another program's brain, behind the same duck-typed surface
    # handle_turn already relies on (stream/commit/set_system, persona_name,
    # direct). Default path below is byte-identical to before.
    llm_cls = LLM
    if not args.no_llm and args.llm_bridge:
        import importlib
        mod_name, _, cls_name = str(args.llm_bridge).partition(":")
        mod = importlib.import_module(mod_name)
        llm_cls = getattr(mod, cls_name or "LLM")
        log(f"[llm] bridge: {mod_name}:{cls_name or 'LLM'}")
    llm = None if args.no_llm else llm_cls(
        url=args.llm_url, system=system, shots=shots,
        direct=args.direct_audio, mmproj_gpu=mmproj_gpu,
        restart=args.restart_llm, llm_device=args.llm_device)
    if llm is not None:
        llm.persona_name = "custom" if args.system else name
    if args.lite:
        args.tts_backend = "onnx"
        args.vad_backend = "onnx"
        args.tts_device = "cpu"
        if args.speaker_lock:
            log("[lite] speaker lock needs torchaudio's Kaldi filterbank; "
                "disabling it for this run (see docs/LITE.md)")
            args.speaker_lock = False
    models = Models(tts_device=args.tts_device, stt_device=args.stt_device,
                    tts_backend=args.tts_backend,
                    vad_backend=getattr(args, "vad_backend", "torch"),
                    voice=args.voice or "af_heart", speed=args.speed)
    parallel = not args.serial

    interactive = (not args.no_menu and not args.simulate
                   and sys.stdin is not None and sys.stdin.isatty())
    if interactive:
        startup_menu(models, llm, personas,
                     ask_voice=args.voice is None,
                     ask_persona=args.persona is None and args.system is None)

    if WEB is not None:
        def push_config():
            web(type="config",
                mode=("direct-audio" if args.direct_audio
                      else "canned" if args.no_llm else "text"),
                persona=(getattr(llm, "persona_name", None) or "canned"),
                voice=models.voice, speed=models.speed,
                stt_device=models.stt_device, tts_device=models.tts_device,
                llm_device=args.llm_device, interruptible=True, ready=True,
                voices=[[v, d] for v, d in VOICES],
                personas=sorted(load_personas()) if llm is not None else [])

        def control(cmd):
            """Applied from the web UI (POST /control on an HTTP thread).
            Voice/speed take effect at the next spoken sentence; persona swap
            resets the conversation and re-prewarms the prompt prefix."""
            act, val = cmd.get("action"), cmd.get("value")
            if act == "voice" and isinstance(val, str) and val:
                models.voice = val
                log(f"[web] voice -> {val}")
                web(type="sys", text=f"voice → {val}")
            elif act == "speed":
                models.speed = max(0.5, min(2.0, float(val)))
                log(f"[web] speed -> {models.speed:g}x")
            elif act == "persona" and llm is not None:
                if TURN_ACTIVE.is_set():
                    raise RuntimeError("persona cannot change during an active reply")
                ps = load_personas()  # fresh read: picks up file edits live
                if val in ps:
                    llm.set_system(ps[val]["system"], ps[val]["shots"])
                    llm.persona_name = val
                    log(f"[web] persona -> {val}")
                    web(type="sys", text=f"persona → {val} · memory cleared")
            elif act == "interrupt":
                if not TURN_ACTIVE.is_set():
                    INTERRUPT.clear()
                    raise RuntimeError("there is no active reply to interrupt")
                INTERRUPT.set()
            elif act == "enroll":
                ENROLL_REQUEST.set()
                log("[web] learning your voice — say a sentence, twice")
                web(type="sys", text="say a sentence, twice")
            elif act == "shutdown":
                log("[web] shutdown requested — bye")
                web(type="sys", text="pipeline stopped")
                web(type="state", state="stopped")
                threading.Timer(0.4, lambda: os._exit(0)).start()
            push_config()

        WEB.on_command = control
        push_config()
        web(type="state", state="idle")
        if not args.simulate:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{web_port}")

    if args.simulate:
        run_simulate(models, args.simulate, args.no_play, parallel, llm=llm)
    else:
        try:
            run_live(models, args.no_play, parallel, args.min_silence,
                     spec_silence_ms=args.spec_silence,
                     smart_turn=not args.no_smart_turn,
                     barge_in=args.barge_in or args.aec or args.barge_key,
                     aec=args.aec, llm=llm,
                     key_barge=args.barge_key or args.ptt,
                     ptt=args.ptt,
                     speaker_lock=args.speaker_lock,
                     duck_ms=args.barge_duck_ms,
                     cancel_ms=args.barge_cancel_ms,
                     snr_db=args.barge_snr_db,
                     speaker_threshold=args.speaker_threshold,
                     enroll=args.enroll)
        except KeyboardInterrupt:
            log("\nbye")


if __name__ == "__main__":
    main()
