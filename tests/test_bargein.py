"""Headless validation of barge-in: gate staging + playback cancellation.

No audio hardware needed:
  1. BargeGate staging — duck, then hold, then cancel; blips are ignored, a
     backchannel-length burst releases instead of cancelling, and a prosodic
     pause mid-interruption does not reset the decision.
  1b. The backchannel classifier that resolves the turn boundary.
  2. Live-loop simulation — TurnDetector + SpeechAdmit + gate over a synthetic
     mic stream (silence, then speech) while a reply is "playing"; measures
     both the perceived yield (duck) and the abandon (cancel).
  3. Real Models.speak cancellation — synthesizes a long reply into a fake
     play stream that blocks like a real speaker (write sleeps for the block's
     duration); cancel is set mid-playback; asserts speak() aborts quickly
     and reports cancelled=True.

For the full acoustic picture — echo, background beds and a television — see
tests/bargein_sim.py, which drives the real run_live() loop.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime import (BargeGate, SpeechAdmit, TurnDetector, FRAME, SR,
                      TTS_SR)


def log(m):
    print(m, flush=True)


def run_gate(gate, pattern, playing=True):
    """Feed a list of qualifying/not-qualifying frames; collect the decisions."""
    return [d for d in (gate.update(bool(ok), playing=playing)
                        for ok in pattern) if d]


def test_gate_logic():
    log("=== 1. BargeGate staging ===")
    g = BargeGate()
    ms = BargeGate.FRAME_MS

    # Sustained speech escalates duck -> hold -> cancel, in that order.
    decisions = run_gate(g, [1] * int(g.cancel_ms / ms + 4))
    assert decisions == ["duck", "hold", "cancel"], decisions
    log(f"  sustained speech -> {' -> '.join(decisions)}  OK")

    # A blip must not even duck.
    g = BargeGate()
    assert run_gate(g, [1, 1, 0, 0, 0, 0, 0, 0]) == [], "gate reacted to a blip"
    log("  click/blip -> nothing  OK")

    # A backchannel-length burst ducks, then releases, and never cancels.
    g = BargeGate()
    burst = int(400 / ms)
    decisions = run_gate(g, [1] * burst + [0] * int(1200 / ms))
    assert "cancel" not in decisions, decisions
    assert decisions[0] == "duck" and decisions[-1] == "release", decisions
    log(f"  short burst -> {' -> '.join(decisions)}  OK")

    # The pause after a comma must not abandon a real interruption: this is
    # what a consecutive-frame counter got wrong.
    g = BargeGate()
    gap = [1] * int(300 / ms) + [0] * int(150 / ms) + [1] * int(1400 / ms)
    assert "cancel" in run_gate(g, gap), "prosodic gap defeated the gate"
    log("  speech with a pause in it -> still cancels  OK")

    # Nothing may fire before the reply is actually audible.
    g = BargeGate()
    assert run_gate(g, [1] * 40, playing=False) == [], "fired during grace"
    log("  grace window respected  OK\n")


def test_utterance_classifier():
    log("=== 1b. Backchannel classifier ===")
    from realtime import classify_utterance

    for text in ("mm-hmm", "Yeah.", "okay", "Right, sure.", "uh huh"):
        assert classify_utterance(text) == "backchannel", text
    for text in ("wait, that's wrong", "no stop", "what time does it close",
                 "actually I meant next week"):
        assert classify_utterance(text) == "speech", text
    assert classify_utterance("") == "empty"
    assert classify_utterance("   ") == "empty"
    log("  acknowledgements, stop-words and silence all classified  OK\n")


def test_live_sim():
    log("=== 2. Live-loop barge simulation (Silero VAD, synthetic stream) ===")
    import pytest

    load_silero_vad = pytest.importorskip(
        "silero_vad", reason="needs silero-vad (scripts/setup.sh installs it)"
    ).load_silero_vad
    import soundfile as sf

    vad_model = load_silero_vad()
    det = TurnDetector(vad_model)
    gate = BargeGate()

    # synthetic "user interrupts": 1 s silence then real speech (reuse a wav)
    speech, sr = sf.read(str(Path(__file__).with_name("roundtrip_0.wav")),
                         dtype="float32")
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    n = int(round(len(speech) * SR / sr))
    speech = np.interp(np.linspace(0, 1, n, endpoint=False),
                       np.linspace(0, 1, len(speech), endpoint=False),
                       speech).astype(np.float32)
    stream = np.concatenate([np.zeros(SR, np.float32), speech])
    onset = SR  # speech starts at exactly 1.0 s

    admit = SpeechAdmit()
    ducked_at = cancelled_at = None
    for off in range(0, len(stream) - FRAME + 1, FRAME):
        frame = stream[off:off + FRAME]
        # No reference: nothing is playing, so only the noise floor applies.
        event, _ = det.process(frame, lambda p: admit.update(p, frame))
        decision = gate.update(admit.last_ok, playing=True)
        if decision == "duck" and ducked_at is None:
            ducked_at = off + FRAME
        elif decision == "cancel":
            cancelled_at = off + FRAME
            break
    assert ducked_at is not None, "reply never ducked for real speech"
    assert cancelled_at is not None, "reply never cancelled for real speech"
    log(f"  went quiet {(ducked_at - onset) / SR * 1000:.0f} ms after onset, "
        f"abandoned the turn at {(cancelled_at - onset) / SR * 1000:.0f} ms  OK")
    log("  (the first number is what a speaker perceives as yielding)\n")
    det.reset()


class FakePlayStream:
    """Blocks like a real speaker: write(a) sleeps len(a)/TTS_SR seconds."""

    def __init__(self):
        self.samples_played = 0

    def write(self, a):
        self.samples_played += len(a)
        time.sleep(len(a) / TTS_SR)


def test_speak_cancel():
    log("=== 3. Models.speak cancellation (real Kokoro on CPU) ===")
    import pytest
    from realtime import Models

    # Models() legitimately demands the whole stack — torch, silero-vad,
    # onnx-asr, kokoro. That is production behaviour worth keeping loud; it is
    # this TEST that should stand aside when the machine has not been set up.
    try:
        models = Models(tts_device="cpu")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"voice stack unavailable ({exc}); run scripts/setup.sh")
    long_reply_stream = FakePlayStream()
    cancel = threading.Event()

    # cancel 1.0 s into playback of a ~13 s reply
    t = threading.Timer(1.0, cancel.set)
    t.start()
    t0 = time.perf_counter()
    # temporarily swap the canned reply for a long one via direct speak call
    text = ("This is a very long reply that should get cut off. "
            "It keeps going with several sentences. "
            "Each one takes a while to say out loud. "
            "You should never hear the end of it.")
    first, dur, cut = models.speak(text, long_reply_stream, cancel=cancel)
    elapsed = time.perf_counter() - t0

    assert cut, "speak() did not report cancellation"
    played_s = long_reply_stream.samples_played / TTS_SR
    log(f"  reply audio {dur:.1f}s total; playback stopped after ~{played_s:.1f}s")
    # What matters is when the room goes quiet. Playback is written in 100 ms
    # blocks and each one re-checks the flag, so the sound stops within a block
    # of the cancel.
    assert played_s < 1.4, f"kept playing after cancel: {played_s:.2f}s"
    log(f"  speak() returned {elapsed:.2f}s after start (cancel at 1.0 s)")
    if elapsed > 1.5:
        # Synthesis is not interruptible mid-sentence. On CPU one sentence can
        # take seconds, so the call returns late even though nothing was heard
        # — it costs CPU, not user-perceived latency.
        log(f"  note: {elapsed - 1.0:.1f}s of that is Kokoro finishing the "
            f"sentence it had already started (CPU); no audio was played")
    log("  cancellation OK\n")


if __name__ == "__main__":
    test_gate_logic()
    test_utterance_classifier()
    test_live_sim()
    test_speak_cancel()
    log("ALL BARGE-IN TESTS PASSED")
