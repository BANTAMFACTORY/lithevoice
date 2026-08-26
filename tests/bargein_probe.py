"""Offline measurement of what the barge gate actually sees.

Builds controlled microphone signals with known ground truth and reports, per
class, the distribution of Silero's speech probability and of frame energy
relative to the running noise floor. This is the evidence used to pick the
barge gate's thresholds instead of guessing them.

    ./.venv/bin/python tests/bargein_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))

from bargein_sim import ASSETS, ROOM_TAPS, SR, FRAME, load  # noqa: E402


def echo_of(speaker: np.ndarray, gain: float, n: int) -> np.ndarray:
    out = np.zeros(n, np.float32)
    for d_ms, g in ROOM_TAPS:
        d = int(d_ms / 1000.0 * SR)
        if d >= n:
            continue
        take = min(len(speaker), n - d)
        out[d:d + take] += speaker[:take] * (g * gain)
    return np.tanh(3.0 * out).astype(np.float32) / 3.0


def frames(x: np.ndarray):
    for i in range(0, len(x) - FRAME + 1, FRAME):
        yield i, x[i:i + FRAME]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def main() -> int:
    import realtime
    from silero_vad import load_silero_vad
    import torch

    vad = load_silero_vad()

    print("Synthesizing the assistant's reply on CPU...")
    models = realtime.Models(tts_device="cpu", stt_device="cpu")
    parts = [np.asarray(a, np.float32) for _, _, a in
             models.tts(realtime.CANNED[0], voice="af_heart", speed=1.0)]
    reply24 = np.concatenate(parts)
    n16 = int(round(len(reply24) * SR / 24000))
    reply = np.interp(np.linspace(0, 1, n16, endpoint=False),
                      np.linspace(0, 1, len(reply24), endpoint=False),
                      reply24).astype(np.float32)

    beds = {name: load(name) for name in
            ("bed_room", "bed_fan", "bed_keyboard", "bed_tv")}
    barge = load("barge_wait")
    back = load("back_mhm")

    n = len(reply) + 6 * SR
    cases = []

    # 1. assistant speaking into open speakers, nobody else in the room
    for gain, tag in ((0.18, "echo -15dB"), (0.35, "echo -9dB")):
        sig = np.zeros(n, np.float32)
        sig += beds["bed_room"][:n] * 0.004
        sig += echo_of(reply, gain, n)
        cases.append((f"echo_only ({tag})", sig, None))

    # 2. background beds with nobody talking
    for name, g in (("bed_fan", 0.06), ("bed_keyboard", 0.10),
                    ("bed_tv", 0.05)):
        sig = np.zeros(n, np.float32)
        sig += beds[name][:n] * g
        sig += echo_of(reply, 0.18, n)
        cases.append((f"noise_only ({name})", sig, None))

    # 3. a real interruption over the echo (double-talk)
    sig = np.zeros(n, np.float32)
    sig += beds["bed_room"][:n] * 0.004
    sig += echo_of(reply, 0.18, n)
    at = int(1.4 * SR)
    sig[at:at + len(barge)] += barge
    cases.append(("user_barge (over echo)", sig, (at, at + len(barge))))

    # 4. a backchannel over the echo
    sig = np.zeros(n, np.float32)
    sig += beds["bed_room"][:n] * 0.004
    sig += echo_of(reply, 0.18, n)
    sig[at:at + len(back)] += back
    cases.append(("user_backchannel (over echo)", sig, (at, at + len(back))))

    print(f"\n{'case':32} {'class':10} {'p50 p':>7} {'p90 p':>7} "
          f"{'p>=.5':>6} {'dB over floor':>14}")
    print("-" * 82)

    for name, sig, span in cases:
        vad.reset_states()
        floor = None
        rows = {"other": [], "user": []}
        for i, fr in frames(sig):
            p = float(vad(torch.from_numpy(fr.copy()), SR).item())
            e = rms(fr)
            # noise floor: slow-tracking minimum-ish statistic
            floor = e if floor is None else (
                0.995 * floor + 0.005 * e if e > floor else
                0.85 * floor + 0.15 * e)
            snr = 20 * np.log10(e / floor) if floor > 0 else 0.0
            in_user = span is not None and span[0] <= i < span[1]
            rows["user" if in_user else "other"].append((p, snr))

        for cls, vals in rows.items():
            if not vals:
                continue
            ps = np.array([v[0] for v in vals])
            sn = np.array([v[1] for v in vals])
            print(f"{name:32} {cls:10} {np.percentile(ps,50):7.2f} "
                  f"{np.percentile(ps,90):7.2f} {np.mean(ps>=0.5)*100:5.0f}% "
                  f"{np.percentile(sn,50):7.1f} /{np.percentile(sn,90):6.1f}")

    print("\nColumns: median p, 90th-pct p, %% of frames Silero calls speech,")
    print("and median/90th-pct frame energy above the tracked noise floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
