"""Headless validation of the echo canceller (realtime.py EchoCanceller).

Simulates the acoustic loop with no audio hardware:
  bot voice (existing Kokoro wavs) -> synthetic room/device echo path
  (25 ms bulk delay + decaying reflections, loud: 50% gain) -> "mic".

Case A — echo only: after convergence the cleaned signal must not look like
speech to Silero (p < 0.5), i.e. the bot can't barge itself in. Reports ERLE.

Case B — double-talk: user speech (another wav) overlaps the echo; it must
survive cleaning — VAD fires AND Parakeet still transcribes it correctly.

Also reports per-frame processing cost.
"""
import time

import numpy as np
import soundfile as sf
from numpy.fft import irfft, rfft  # noqa: F401  (test imports nothing heavy)

from realtime import EchoCanceller, RefBuffer, FRAME, SR


def log(m):
    print(m, flush=True)


def load16(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, 1, n, endpoint=False),
                      np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)
    return a


def make_echo(far):
    """Room+device impulse response: 25 ms delay, decaying taps, 50% level."""
    ir = np.zeros(SR // 4, np.float32)          # up to 250 ms of path
    ir[int(0.025 * SR)] = 0.50
    ir[int(0.031 * SR)] = 0.22
    ir[int(0.043 * SR)] = 0.10
    ir[int(0.070 * SR)] = 0.05
    echo = np.convolve(far, ir)[: len(far)].astype(np.float32)
    return echo


def run_case(name, mic, far, vad_model, torch, check_stt=None, user_at=None):
    aec = EchoCanceller()
    ref = RefBuffer()
    ref.push(far, sr=SR)  # pre-push whole far-end (FIFO order == live order)

    cleaned = np.zeros_like(mic)
    probs_raw, probs_clean = [], []
    t_proc = 0.0
    nfr = 0
    vad_state2 = None
    for off in range(0, len(mic) - FRAME + 1, FRAME):
        m = mic[off:off + FRAME]
        t0 = time.perf_counter()
        c = aec.process(m, ref.pull(FRAME))
        t_proc += time.perf_counter() - t0
        nfr += 1
        cleaned[off:off + FRAME] = c
        probs_clean.append(vad_model(torch.from_numpy(c), SR).item())
    vad_model.reset_states()
    for off in range(0, len(mic) - FRAME + 1, FRAME):
        probs_raw.append(vad_model(torch.from_numpy(mic[off:off + FRAME]), SR).item())
    vad_model.reset_states()

    probs_raw, probs_clean = np.array(probs_raw), np.array(probs_clean)
    sec = SR // FRAME  # frames per second
    conv = 2 * sec     # ignore first 2 s (convergence)

    if user_at is None:
        # echo-only: report ERLE + whether VAD would fire after convergence
        erle = 10 * np.log10(
            (np.mean(mic[conv * FRAME:] ** 2) + 1e-12)
            / (np.mean(cleaned[conv * FRAME:] ** 2) + 1e-12))
        p_raw = probs_raw[conv:].max()
        p_clean = probs_clean[conv:].max()
        ok = p_clean < 0.5
        log(f"[{name}] ERLE {erle:.1f} dB | VAD max prob raw {p_raw:.2f} -> "
            f"clean {p_clean:.2f} (post-convergence) "
            f"{'OK — echo will not barge' if ok else '** FAIL — would self-barge **'}")
        assert ok
    else:
        # double-talk: user speech must survive
        u0 = int(user_at * SR) // FRAME
        p_user = probs_clean[u0:u0 + 2 * sec].max()
        ok = p_user >= 0.5
        log(f"[{name}] VAD max prob during user speech (cleaned): {p_user:.2f} "
            f"{'OK — barge-in still works' if ok else '** FAIL — user speech killed **'}")
        assert ok
        if check_stt is not None:
            seg = cleaned[int(user_at * SR):int(user_at * SR) + 3 * SR]
            txt = check_stt(seg).lower()
            hit = sum(w in txt for w in ("quick", "brown", "fox", "lazy", "dog"))
            log(f'  STT on cleaned double-talk: "{txt.strip()}"  '
                f"({hit}/5 keywords) {'OK' if hit >= 3 else '** FAIL **'}")
            assert hit >= 3
    log(f"  processing: {t_proc / nfr * 1000:.2f} ms/frame (budget 32 ms)\n")
    return cleaned


def main():
    log("Loading Silero + Parakeet...")
    import torch
    from silero_vad import load_silero_vad
    import onnx_asr

    vad_model = load_silero_vad()
    stt = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8",
                              providers=["CPUExecutionProvider"])
    log("Ready.\n")

    # bot voice: ~8.5 s (two replies back to back), user: quick brown fox
    far = np.concatenate([load16("roundtrip_1.wav"), load16("roundtrip_2.wav")])
    user = load16("roundtrip_0.wav")
    echo = make_echo(far)

    # Ambient mic noise floor (~-50 dBFS). CRITICAL: a noise-free mic let the
    # old adaptation gate bootstrap from digitally-silent frames — something a
    # real mic never provides (live, the AEC never adapted at all and the bot
    # self-barged). Any regression here reproduces that failure.
    noise = np.random.default_rng(0).normal(
        0, 0.003, len(echo)).astype(np.float32)

    # Case A: echo only (+ noise floor)
    run_case("A: echo-only", echo + noise, far, vad_model, torch)

    # Case B: double-talk — user starts at 4 s, well after convergence
    mic = echo + noise
    at = 4.0
    end = min(len(mic), int(at * SR) + len(user))
    mic[int(at * SR):end] += user[: end - int(at * SR)]
    run_case("B: double-talk", mic, far, vad_model, torch,
             check_stt=lambda a: stt.recognize(a, sample_rate=SR), user_at=at)

    log("ALL AEC TESTS PASSED")


if __name__ == "__main__":
    main()
