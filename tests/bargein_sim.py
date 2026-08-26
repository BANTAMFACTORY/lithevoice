"""Silent, repeatable barge-in simulation against the real run_live() loop.

No audio hardware is touched. A virtual full-duplex device stands in for
sounddevice: the assistant's Kokoro audio is "played" into a modelled room and
comes back into the microphone as echo, mixed with a background bed and with
user speech scheduled relative to the reply. Everything else — Silero VAD,
TurnDetector, SmartTurn, BargeGate, the AEC, Parakeet — is the real code.

    ./.venv/bin/python tests/bargein_sim.py --list
    ./.venv/bin/python tests/bargein_sim.py --scenario speaker_moderate
    ./.venv/bin/python tests/bargein_sim.py            # whole suite

Metrics per scenario:
  false barges   — reply cancelled while the user was NOT speaking (echo,
                   noise or the television did it). The self-interrupt bug.
  cut backchannel— reply cancelled by a short "mm-hmm" that was not meant to
                   take the floor. The main comfort complaint.
  true barges    — reply cancelled while the user really was interrupting.
  barge latency  — ms from the user's speech onset to the cancel.
  missed barges  — the user talked over the reply and it never yielded.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))

ASSETS = ROOT / "outputs" / "bargein"

SR = 16000
TTS_SR = 24000
FRAME = 512
FRAME_S = FRAME / SR

# Speaker -> microphone impulse response, as (delay_ms, gain) taps. The direct
# path plus a handful of early reflections; `echo_gain` scales the whole thing.
ROOM_TAPS = ((0.0, 1.0), (11.0, 0.5), (23.0, 0.30), (37.0, 0.18),
             (53.0, 0.10), (71.0, 0.06))
# Time from write() to sound leaving the speaker. Real devices buffer; this is
# also what keeps the reference ahead of the echo for the AEC.
OUTPUT_LATENCY_S = 0.060


def db(x: float) -> str:
    return "-inf" if x <= 0 else f"{20 * np.log10(x):+.0f} dB"


def speech_onset(audio: np.ndarray, thresh: float = 0.02) -> float:
    """Seconds of leading near-silence before the clip actually speaks."""
    hits = np.flatnonzero(np.abs(audio) > thresh)
    return float(hits[0]) / SR if len(hits) else 0.0


class Room:
    """Virtual acoustic environment shared by the fake input/output streams."""

    def __init__(self, duration_s: float, echo_gain: float, bed: np.ndarray,
                 bed_gain: float, user_gain: float = 1.0,
                 nonlinear: bool = True, seed: int = 7):
        self.n = int(duration_s * SR) + 4 * SR
        self.echo_gain = echo_gain
        self.user_gain = user_gain
        self.nonlinear = nonlinear
        self.rng = np.random.default_rng(seed)

        self.speaker = np.zeros(self.n, np.float32)   # what the speaker emits
        self.user = np.zeros(self.n, np.float32)      # near-end (real) speech
        self.bed = np.zeros(self.n, np.float32)
        if bed is not None and bed_gain > 0:
            tiled = np.tile(bed, int(np.ceil(self.n / len(bed))))[:self.n]
            self.bed = (tiled * bed_gain).astype(np.float32)

        self.taps = [(int(d / 1000.0 * SR), g * echo_gain) for d, g in ROOM_TAPS]
        self.lock = threading.Lock()
        self.t0 = None
        self.frames_emitted = 0
        self.mic_log = np.zeros(self.n, np.float32)

        # ground truth
        self.user_spans: list[tuple[float, float, str]] = []
        self.reply_onsets: list[float] = []
        self._last_write_at = -10.0
        self._pending: list[tuple[int, float, np.ndarray, str]] = []

    # -- clock -------------------------------------------------------------
    def start(self):
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    # -- scheduling --------------------------------------------------------
    def schedule_abs(self, t: float, audio: np.ndarray, label: str):
        """Place a user utterance at an absolute simulation time."""
        with self.lock:
            self._place(t, audio, label)

    def _place(self, t: float, audio: np.ndarray, label: str):
        a = int(t * SR)
        b = min(self.n, a + len(audio))
        if a >= self.n:
            return
        self.user[a:b] += (audio[:b - a] * self.user_gain).astype(np.float32)
        # Barge latency is judged from the first audible sample, not from the
        # scheduling instant: Kokoro clips start with a little leading silence
        # and charging that to the gate would flatter/penalise it unfairly.
        self.user_spans.append((t + speech_onset(audio), t + (b - a) / SR,
                                label))

    def schedule_after_reply(self, reply_index: int, delay_s: float,
                             audio: np.ndarray, label: str):
        """Place an utterance delay_s after reply `reply_index` starts."""
        with self.lock:
            self._pending.append((reply_index, delay_s, audio, label))

    # -- device sides ------------------------------------------------------
    def speaker_write(self, block_tts: np.ndarray):
        """Called by the fake OutputStream; blocks like a real device."""
        dur = len(block_tts) / TTS_SR
        n16 = int(round(len(block_tts) * SR / TTS_SR))
        blk = np.interp(np.linspace(0, 1, n16, endpoint=False),
                        np.linspace(0, 1, len(block_tts), endpoint=False),
                        block_tts).astype(np.float32)
        now = self.now()
        with self.lock:
            if now - self._last_write_at > 0.4:      # a new reply began
                self.reply_onsets.append(now)
                idx = len(self.reply_onsets) - 1
                still = []
                for want, delay, audio, label in self._pending:
                    if want == idx:
                        self._place(now + delay, audio, label)
                    else:
                        still.append((want, delay, audio, label))
                self._pending = still
            self._last_write_at = now + dur
            a = int((now + OUTPUT_LATENCY_S) * SR)
            b = min(self.n, a + len(blk))
            if a < self.n:
                self.speaker[a:b] += blk[:b - a]
        time.sleep(dur)   # a real device paces playback

    def mic_frame(self) -> np.ndarray:
        """Build the next 32 ms microphone frame."""
        with self.lock:
            a = self.frames_emitted * FRAME
            self.frames_emitted += 1
            b = a + FRAME
            if b > self.n:
                return np.zeros(FRAME, np.float32)

            out = self.bed[a:b] + self.user[a:b]
            if self.echo_gain > 0:
                echo = np.zeros(FRAME, np.float32)
                for d, g in self.taps:
                    if g == 0:
                        continue
                    s, e = a - d, b - d
                    if e <= 0:
                        continue
                    s2 = max(0, s)
                    echo[s2 - s:] += self.speaker[s2:e] * g
                if self.nonlinear:
                    # Small speakers distort; the harmonics they add are what a
                    # linear AEC cannot subtract, so this is load-bearing.
                    echo = np.tanh(3.0 * echo).astype(np.float32) / 3.0
                out = out + echo
            out = out + self.rng.standard_normal(FRAME).astype(np.float32) * 2e-4
            out = np.clip(out, -1.0, 1.0).astype(np.float32)
            self.mic_log[a:b] = out
            return out

    # -- ground truth ------------------------------------------------------
    def user_active_at(self, t: float, pad: float = 0.25) -> str | None:
        for s, e, label in self.user_spans:
            if s - 0.05 <= t <= e + pad:
                return label
        return None


class FakeOutputStream:
    def __init__(self, room: Room):
        self.room = room

    def write(self, block):
        self.room.speaker_write(np.asarray(block, np.float32))

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class SimulationOver(Exception):
    pass


class FakeInputStream:
    """Feeds room frames to run_live's callback on a wall clock."""

    def __init__(self, room: Room, callback, stop: threading.Event):
        self.room = room
        self.callback = callback
        self.stop = stop
        self.thread = None

    def _pump(self):
        self.room.start()
        k = 0
        while not self.stop.is_set():
            target = self.room.t0 + k * FRAME_S
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            frame = self.room.mic_frame()
            self.callback(frame.reshape(-1, 1), FRAME, None, None)
            k += 1

    def __enter__(self):
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        return False


def install_fake_sounddevice(room: Room, stop: threading.Event):
    """Replace the sounddevice module for the duration of a scenario."""
    mod = types.ModuleType("sounddevice")

    def _input_stream(*a, callback=None, **kw):
        return FakeInputStream(room, callback, stop)

    def _output_stream(*a, **kw):
        return FakeOutputStream(room)

    mod.InputStream = _input_stream
    mod.OutputStream = _output_stream
    mod.query_devices = lambda *a, **k: []
    sys.modules["sounddevice"] = mod
    return mod


@dataclass
class Scenario:
    name: str
    what: str
    echo_gain: float
    bed: str
    bed_gain: float
    aec: bool = False
    duration: float = 26.0
    opening: str = "turn_weather"
    # (reply_index, delay_after_reply_start, asset, kind)
    events: list = field(default_factory=list)
    # (absolute_time, asset, kind) — used for enrolment and for a second person
    absolute: list = field(default_factory=list)
    speaker_lock: bool = False
    expect: str = ""


def load(name: str) -> np.ndarray:
    audio, sr = sf.read(ASSETS / f"{name}.wav", dtype="float32")
    assert sr == SR, f"{name} is {sr} Hz, expected {SR}"
    return audio


def build_suite() -> list[Scenario]:
    B = "barge"        # a real interruption: SHOULD cut the reply
    K = "backchannel"  # an acknowledgement: should NOT cut the reply
    return [
        Scenario("headset_clean", "headset, no echo, quiet room",
                 echo_gain=0.0, bed="bed_room", bed_gain=0.004,
                 events=[(0, 1.4, "barge_wait", B)],
                 expect="fast barge, no false trigger"),
        Scenario("speaker_moderate", "open speakers at -15 dB, no AEC",
                 echo_gain=0.18, bed="bed_room", bed_gain=0.004,
                 events=[(0, 1.4, "barge_wait", B)],
                 expect="echo must not self-interrupt"),
        Scenario("speaker_moderate_aec", "open speakers at -15 dB, AEC on",
                 echo_gain=0.18, bed="bed_room", bed_gain=0.004, aec=True,
                 events=[(0, 1.4, "barge_wait", B)],
                 expect="AEC suppresses echo, barge still works"),
        Scenario("speaker_loud_aec", "open speakers at -9 dB, AEC on",
                 echo_gain=0.35, bed="bed_room", bed_gain=0.004, aec=True,
                 events=[(0, 1.4, "barge_stop", B)],
                 expect="louder echo, AEC still holds"),
        Scenario("backchannel", "user says mm-hmm mid-reply (headset)",
                 echo_gain=0.0, bed="bed_room", bed_gain=0.004,
                 events=[(0, 1.3, "back_mhm", K), (0, 2.6, "back_yeah", K)],
                 expect="reply should NOT be cut"),
        Scenario("backchannel_speaker", "mm-hmm mid-reply, open speakers+AEC",
                 echo_gain=0.18, bed="bed_room", bed_gain=0.004, aec=True,
                 events=[(0, 1.3, "back_okay", K), (0, 2.7, "back_right", K)],
                 expect="reply should NOT be cut"),
        Scenario("tv_babble", "television in the next room, nobody talking",
                 echo_gain=0.18, bed="bed_tv", bed_gain=0.05, aec=True,
                 events=[], expect="zero barges"),
        Scenario("keyboard", "typing during the reply",
                 echo_gain=0.18, bed="bed_keyboard", bed_gain=0.10, aec=True,
                 events=[], expect="zero barges"),
        Scenario("fan_noise", "loud desk fan, nobody talking",
                 echo_gain=0.18, bed="bed_fan", bed_gain=0.06, aec=True,
                 events=[], expect="zero barges"),
        Scenario("speaker_lock", "a second person tries to take the floor",
                 echo_gain=0.0, bed="bed_room", bed_gain=0.004,
                 speaker_lock=True, duration=46.0,
                 opening="turn_weather",
                 absolute=[(7.0, "turn_reminder", "enrol"),
                           (14.0, "turn_store", "user"),
                           (23.0, "other_question", "intruder"),
                           (31.0, "other_command", "intruder"),
                           (39.0, "barge_wait", "user")],
                 expect="enrols on the first two, then only that voice is "
                        "answered"),
        Scenario("double_talk", "real interruption over loud speakers",
                 echo_gain=0.30, bed="bed_room", bed_gain=0.006, aec=True,
                 events=[(0, 1.2, "barge_rephrase", B)],
                 expect="cut fast despite echo"),
    ]


class Recorder:
    """Captures run_live's own events by wrapping realtime.log / realtime.web."""

    def __init__(self, room: Room, verbose: bool):
        self.room = room
        self.verbose = verbose
        self.barges: list[float] = []
        self.turns: list[float] = []
        self.ducks: list[float] = []
        self.holds: list[float] = []
        self.releases: list[float] = []
        self.cuts: list[float] = []            # replies actually destroyed
        self.ignored_back: list[float] = []    # backchannels kept off the floor
        self.ignored_noise: list[float] = []   # wordless sound rejected
        self.other_voice: list[float] = []     # rejected by identity
        self.enrolled = False
        self.lines: list[tuple[float, str]] = []

    def log(self, msg):
        t = self.room.now()
        self.lines.append((t, str(msg)))
        if self.verbose:
            print(f"    [{t:6.2f}s] {msg}", flush=True)
        text = str(msg).strip()
        if text.startswith("[barge-in]"):
            self.barges.append(t)
        elif text.startswith("[reply cut off"):
            self.cuts.append(t)
        elif text.startswith("[backchannel]"):
            self.ignored_back.append(t)
        elif text.startswith("[ignored]"):
            self.ignored_noise.append(t)
        elif text.startswith("[other voice]"):
            self.other_voice.append(t)
        elif text.startswith("[enrol] voice learned"):
            self.enrolled = True
        elif text.startswith("[turn "):
            self.turns.append(t)

    def web(self, **ev):
        if ev.get("type") == "duck":
            t = self.room.now()
            stage = ev.get("stage", "duck" if ev.get("on") else "release")
            if stage == "duck":
                self.ducks.append(t)
            elif stage == "hold":
                self.holds.append(t)
            else:
                self.releases.append(t)
            if self.verbose:
                print(f"    [{t:6.2f}s] {stage}", flush=True)


def evaluate(sc: Scenario, room: Room, rec: Recorder) -> dict:
    false_b, true_b, cut_back, lat = 0, 0, 0, []
    kinds = {a: k for (_, _, a, k) in sc.events}
    # absolutely-scheduled utterances are ground truth too
    kinds.update({a: k for (_, a, k) in sc.absolute})
    for t in rec.barges:
        label = room.user_active_at(t)
        kind = kinds.get(label) if label else None
        if kind is None:
            # Nobody was interrupting: echo, noise, the television — or the
            # user's own opening turn, which is not an interruption either.
            false_b += 1
            continue
        if kind == "backchannel":
            cut_back += 1
        else:
            true_b += 1
        # the most recent onset of that label at or before the cancel
        onsets = [s for s, e, l in room.user_spans if l == label and s <= t]
        onset = max(onsets) if onsets else min(
            s for s, e, l in room.user_spans if l == label)
        lat.append((t - onset) * 1000)

    # A real interruption that never produced a cancel is a miss.
    wanted = {a for (_, _, a, k) in sc.events if k == "barge"}
    got = {room.user_active_at(t) for t in rec.barges}
    missed = len(wanted - got)

    # A duck while nobody was speaking is a (much milder) false positive: the
    # reply dips for a moment instead of being destroyed.
    false_duck = sum(1 for t in rec.ducks if room.user_active_at(t) is None)

    # The metric that actually matters is how a reply DIED, by whatever path.
    # A gate that never fires but lets finish_turn kill the reply a second
    # later has not solved anything.
    lost_noise = lost_back = lost_real = 0
    for t in rec.cuts:
        label = room.user_active_at(t)
        kind = kinds.get(label) if label else None
        if kind in ("barge", "user"):
            lost_real += 1          # correct: the user took the floor
        elif kind == "backchannel":
            lost_back += 1          # the comfort failure
        else:
            lost_noise += 1         # echo/noise/TV destroyed a reply

    # What the user actually perceives as "it yielded" is the duck, not the
    # cancel — so this is the headline responsiveness number.
    duck_lat = []
    for t in rec.ducks:
        label = room.user_active_at(t)
        if kinds.get(label) != "barge":
            continue
        onsets = [s for s, e, l in room.user_spans if l == label and s <= t]
        if onsets:
            duck_lat.append((t - max(onsets)) * 1000)

    return {
        "false": false_b,
        "cut_backchannel": cut_back,
        "true": true_b,
        "missed": missed,
        "latency_ms": lat,
        "duck_latency_ms": duck_lat,
        "barges": len(rec.barges),
        "ducks": len(rec.ducks),
        "false_ducks": false_duck,
        "lost_noise": lost_noise,
        "lost_back": lost_back,
        "lost_real": lost_real,
        "kept_back": len(rec.ignored_back),
        "kept_noise": len(rec.ignored_noise),
        "other_voice_rejected": len(rec.other_voice),
        "enrolled": rec.enrolled,
        "turns": len(rec.turns),
        "replies": len(room.reply_onsets),
    }


class LegacyAdmit:
    """Emulates the original gate's view: Silero probability and nothing else.

    Returning True from update() leaves TurnDetector ungated (as it was), while
    last_ok reproduces the old barge test of p >= 0.5.
    """

    def __init__(self, *a, **k):
        self.last_ok = False
        self.floor = None
        self.playing = False

    def update(self, p, frame, ref_frame=None):
        self.last_ok = p >= 0.5
        return True


class LegacyGate:
    """The original rule: three consecutive speech frames, then kill the reply.
    No duck, no hold, no grace window, no echo or identity test."""

    def __init__(self, *a, **k):
        self.reset()

    def reset(self):
        self.count = 0

    def relax(self):
        self.reset()

    @property
    def committed(self):
        return False

    def update(self, ok, playing=False):
        self.count = self.count + 1 if ok else 0
        if self.count == 3:
            return "cancel"
        return None


def run_scenario(models, sc: Scenario, verbose: bool = False,
                 dump: bool = False, voice_profile: bool = True,
                 oracle: bool = False, legacy: bool = False,
                 policy: str | None = None) -> dict:
    import realtime

    bed = load(sc.bed) if sc.bed else None
    room = Room(sc.duration, sc.echo_gain, bed, sc.bed_gain)
    stop = threading.Event()
    install_fake_sounddevice(room, stop)

    rec = Recorder(room, verbose)
    orig_log, orig_web = realtime.log, realtime.web
    realtime.log, realtime.web = rec.log, rec.web

    saved = (realtime.BargeGate, realtime.SpeechAdmit,
             realtime.classify_utterance)
    if legacy:
        realtime.BargeGate = LegacyGate
        realtime.SpeechAdmit = LegacyAdmit
        realtime.classify_utterance = lambda text, max_words=4: "speech"
    elif policy and policy != "ours":
        # Same acoustic front end (SpeechAdmit) and same turn-boundary
        # transcript logic; only the policy differs, which is the thing under
        # comparison.
        from policies import POLICIES
        realtime.BargeGate = POLICIES[policy]

    # Break run_live out of its infinite q.get() when the scenario ends.
    orig_queue = realtime.queue
    shim = types.ModuleType("queue")
    shim.Empty = orig_queue.Empty

    class StoppableQueue(orig_queue.Queue):
        def get(self, block=True, timeout=None):
            while True:
                try:
                    return super().get(timeout=0.1)
                except orig_queue.Empty:
                    if stop.is_set():
                        raise SimulationOver
    shim.Queue = StoppableQueue
    realtime.queue = shim

    # The identity ceiling: a perfect speaker identifier, to measure what the
    # acoustic gates cannot reach on their own. Not a shippable component —
    # it reads the ground truth — but it prices the upgrade honestly.
    speaker_ok = None
    if oracle:
        def speaker_ok(_frame, _room=room):
            return _room.user_active_at(_room.now(), pad=0.15) is not None

    if sc.speaker_lock:
        # Never touch the operator's real enrolment.
        realtime.PROFILE_PATH = str(ASSETS / "sim_voice_profile.npz")

    room.schedule_abs(1.0, load(sc.opening), sc.opening)
    for when, asset, _kind in sc.absolute:
        room.schedule_abs(when, load(asset), asset)
    for reply_idx, delay, asset, _kind in sc.events:
        room.schedule_after_reply(reply_idx, delay, load(asset), asset)

    def _loop():
        try:
            realtime.run_live(models, no_play=False, parallel=True,
                              min_silence_ms=600, smart_turn=True,
                              barge_in=True, aec=sc.aec, llm=None,
                              key_barge=False, speaker_ok=speaker_ok,
                              speaker_lock=sc.speaker_lock,
                              enroll=sc.speaker_lock)
        except SimulationOver:
            pass
        except Exception as exc:              # surfaced in the summary
            rec.lines.append((room.now(), f"!! {type(exc).__name__}: {exc}"))

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()

    deadline = time.perf_counter() + sc.duration
    while time.perf_counter() < deadline and thread.is_alive():
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=15)

    realtime.log, realtime.web = orig_log, orig_web
    realtime.queue = orig_queue
    (realtime.BargeGate, realtime.SpeechAdmit,
     realtime.classify_utterance) = saved

    if dump:
        out = ASSETS / f"mic_{sc.name}.wav"
        sf.write(out, room.mic_log[:room.frames_emitted * FRAME], SR)

    result = evaluate(sc, room, rec)
    result["errors"] = [l for _, l in rec.lines if l.startswith("!!")]
    return result


def print_row(sc: Scenario, r: dict):
    lat = r["latency_ms"]
    lat_s = f"{np.mean(lat):5.0f}" if lat else "    -"
    wanted = sum(1 for (_, _, _, k) in sc.events if k == "barge")
    flag = "ok " if (r["lost_noise"] == 0 and r["lost_back"] == 0
                     and r["lost_real"] >= wanted) else "BAD"
    print(f"  {flag} {sc.name:22} "
          f"lost:noise={r['lost_noise']:2d} back={r['lost_back']:2d} "
          f"real={r['lost_real']:2d}/{wanted}  "
          f"kept:back={r['kept_back']:2d} noise={r['kept_noise']:2d}  "
          f"duck={r['ducks']:2d}(bad {r['false_ducks']:2d})  "
          f"lat={lat_s} ms")
    for e in r["errors"]:
        print(f"      {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", action="append",
                    help="run only these scenarios (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-voice-profile", action="store_true",
                    help="A/B: run without the enrolled-voice match")
    ap.add_argument("--policy", default="ours",
                    help="barge-in policy: ours, ours-v2, livekit, pipecat")
    ap.add_argument("--legacy", action="store_true",
                    help="A/B: emulate the original three-frame gate")
    ap.add_argument("--oracle-speaker", action="store_true",
                    help="A/B: substitute a perfect speaker identifier to "
                         "measure the ceiling identity would buy")
    ap.add_argument("--dump", action="store_true",
                    help="write the simulated mic signal per scenario")
    args = ap.parse_args()

    suite = build_suite()
    if args.list:
        for sc in suite:
            print(f"  {sc.name:22} {sc.what}")
            print(f"  {'':22} expect: {sc.expect}")
        return 0
    if args.scenario:
        suite = [s for s in suite if s.name in set(args.scenario)]
        if not suite:
            print("no scenario matched", file=sys.stderr)
            return 2

    if not (ASSETS / "turn_weather.wav").is_file():
        print("assets missing — run tests/bargein_assets.py first",
              file=sys.stderr)
        return 2

    import realtime
    print("Loading models on CPU (no GPU, no LLM)...")
    models = realtime.Models(tts_device="cpu", stt_device="cpu")

    print(f"\nRunning {len(suite)} scenario(s), silently. "
          f"policy={args.policy}\n")
    results = []
    for sc in suite:
        print(f"* {sc.name}: {sc.what}")
        print(f"    echo={db(sc.echo_gain)} bed={sc.bed}@{sc.bed_gain} "
              f"aec={'on' if sc.aec else 'off'} | expect: {sc.expect}")
        r = run_scenario(models, sc, verbose=args.verbose, dump=args.dump,
                         voice_profile=not args.no_voice_profile,
                         oracle=args.oracle_speaker, legacy=args.legacy,
                         policy=args.policy)
        results.append((sc, r))
        print_row(sc, r)
        print()

    print("=" * 78)
    print("SUMMARY")
    for sc, r in results:
        print_row(sc, r)
    all_lat = [x for _, r in results for x in r["latency_ms"]]
    wanted = sum(1 for sc, _ in results
                 for (_, _, _, k) in sc.events if k == "barge")
    print("-" * 78)
    print(f"  replies destroyed by noise/echo/TV : "
          f"{sum(r['lost_noise'] for _, r in results)}   (want 0)")
    print(f"  replies destroyed by a backchannel : "
          f"{sum(r['lost_back'] for _, r in results)}   (want 0)")
    print(f"  real interruptions honoured        : "
          f"{sum(r['lost_real'] for _, r in results)}/{wanted}")
    print(f"  backchannels kept off the floor    : "
          f"{sum(r['kept_back'] for _, r in results)}")
    print(f"  wordless sound rejected            : "
          f"{sum(r['kept_noise'] for _, r in results)}")
    print(f"  ducks while nobody spoke           : "
          f"{sum(r['false_ducks'] for _, r in results)}")
    all_duck = [x for _, r in results for x in r["duck_latency_ms"]]
    if all_duck:
        print(f"  time to go quiet (perceived)       : "
              f"mean {np.mean(all_duck):.0f} ms, max {np.max(all_duck):.0f} ms")
    if all_lat:
        print(f"  time to abandon the turn           : "
              f"mean {np.mean(all_lat):.0f} ms, max {np.max(all_lat):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
