"""Headless validation of realtime.py's TurnDetector + speculative STT
+ smart-turn v3 semantic end-of-turn — device-configurable.

Streams synthetic utterances (with Kokoro's natural mid-utterance pauses)
through TurnDetector at REAL-TIME pacing, launching actual speculative STT
threads exactly like run_live does. Uses the production Models class, so
--stt-device/--tts-device benchmark the exact configs realtime.py runs.

Examples:
  python test_turndetector.py                                  # hybrid (default)
  python test_turndetector.py --stt-device cpu --tts-device cpu   # all-CPU
  python test_turndetector.py --stt-device cuda --tts-device cuda # all-GPU
"""
import argparse
import subprocess
import threading
import time

import numpy as np
import torch

from realtime import Models, SmartTurn, TurnDetector, FRAME, SR

UTTERANCES = [
    ("short", "Hey, how are you?"),
    ("medium", "Can you remind me what I have going on this afternoon?"),
    ("long", "So I was thinking about heading to the store later today, "
             "and I wanted to know if you think it's going to rain, "
             "or if I should just walk there instead of driving."),
]
REPLY = "Hey bud, good to hear from you. Hope your day is going well."


def log(m):
    print(m, flush=True)


def gpu_mem():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        return out
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-silence", type=int, default=600)
    ap.add_argument("--stt-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--tts-device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    log(f"gpu mem before load: {gpu_mem()}")
    models = Models(tts_device=args.tts_device, stt_device=args.stt_device)
    smart = SmartTurn()
    log(f"gpu mem after load : {gpu_mem()}\n")

    # measured canned-reply TTS first-audio on the chosen device (best of 3)
    tts_first_ms = min(
        models.speak(REPLY, None)[0] for _ in range(3)) * 1000
    log(f"TTS first-audio (canned reply, {args.tts_device}): {tts_first_ms:.0f} ms\n")

    def synth16(text):
        chunks = [a for _, _, a in models.tts(text, voice="am_michael")]
        a = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        a = np.asarray(a, np.float32)
        n = int(round(len(a) * SR / 24000))
        a = np.interp(np.linspace(0, 1, n, endpoint=False),
                      np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)
        idx = np.where(np.abs(a) > 1e-3)[0]
        return a[: idx[-1] + 1]

    log(f"=== TurnDetector @ min_silence={args.min_silence} ms, "
        f"STT={args.stt_device}, TTS={args.tts_device}, real-time paced ===")
    results = []
    for name, text in UTTERANCES:
        utt = synth16(text)
        stream = np.concatenate(
            [np.zeros(SR // 2, np.float32), utt, np.zeros(2 * SR, np.float32)])
        speech_end = (SR // 2 + len(utt)) / SR  # stream-time, seconds

        det = TurnDetector(models.vad_model, min_silence_ms=args.min_silence)
        spec = None
        resumes = 0
        turn = None

        t0 = time.perf_counter()
        for i, off in enumerate(range(0, len(stream) - FRAME + 1, FRAME)):
            target = t0 + i * FRAME / SR  # real-time pacing
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            event, payload = det.process(stream[off:off + FRAME])
            confirmed = None  # (why, utt)
            if event == "spec":
                out = {}

                def _run(a=payload, o=out):
                    t = time.perf_counter()
                    o["text"] = models.stt.recognize(a, sample_rate=SR)
                    o["ms"] = (time.perf_counter() - t) * 1000
                spec = {"thread": threading.Thread(target=_run), "out": out}
                spec["thread"].start()
                # semantic end-of-turn check, exactly as run_live does
                t_st = time.perf_counter()
                p = smart.predict(payload)
                st_ms = (time.perf_counter() - t_st) * 1000
                stream_pos = (off + FRAME) / SR
                is_final_pause = stream_pos > speech_end  # ground truth
                verdict = "complete" if p > smart.THRESHOLD else "incomplete"
                truth = "true-end" if is_final_pause else "mid-pause"
                ok = (p > smart.THRESHOLD) == is_final_pause
                log(f"  spec@{stream_pos:5.2f}s [{truth:9}] smart-turn p={p:.2f} "
                    f"({verdict}, {st_ms:.0f} ms) {'OK' if ok else '** WRONG **'}")
                if p > smart.THRESHOLD:
                    confirmed = (f"smart-turn ({st_ms:.0f} ms)", payload)
            elif event == "resume":
                spec = None
                resumes += 1
            elif event == "end":
                confirmed = ("silence timeout", payload)

            if confirmed:
                why, utt_audio = confirmed
                t_confirm = time.perf_counter()
                stream_pos = (off + FRAME) / SR
                vad_delay_ms = (stream_pos - speech_end) * 1000
                if why.startswith("smart-turn"):
                    vad_delay_ms += st_ms  # inference delays the confirm
                if spec:
                    spec["thread"].join()
                    wait_ms = (time.perf_counter() - t_confirm) * 1000
                    stt_ms = spec["out"]["ms"]
                    text_out = spec["out"]["text"]
                else:
                    t = time.perf_counter()
                    text_out = models.stt.recognize(utt_audio, sample_rate=SR)
                    wait_ms = stt_ms = (time.perf_counter() - t) * 1000
                turn = (len(utt) / SR, vad_delay_ms, stt_ms, wait_ms,
                        resumes, text_out.strip(), why)
                break

        if turn is None:
            log(f"[{name}] NO TURN DETECTED (min_silence too high?)")
            continue
        dur, vad_ms, stt_full, wait_ms, resumes, text_out, why = turn
        v2v = vad_ms + max(wait_ms, tts_first_ms)
        results.append((name, dur, vad_ms, stt_full, wait_ms, v2v, why))
        log(f"[{name}] {dur:.1f}s  resumes(discarded specs)={resumes}  via {why}")
        log(f'  "{text_out}"')
        log(f"  confirm delay     : {vad_ms:.0f} ms after true end of speech")
        log(f"  STT full          : {stt_full:.0f} ms (ran during wait)")
        log(f"  STT wait@confirm  : {wait_ms:.0f} ms   <-- hidden if ~0")
        log(f"  projected v2v     : {v2v:.0f} ms  (confirm + max(sttWait, "
            f"{tts_first_ms:.0f}))\n")

    log(f"=== summary  [STT={args.stt_device} TTS={args.tts_device} "
        f"ttsFirst={tts_first_ms:.0f}ms] ===")
    log(f"{'utt':8} {'len':>5} {'confirm':>8} {'sttFull':>8} {'sttWait':>8} {'v2v':>6}  via")
    for name, dur, v, sf, w, v2, why in results:
        log(f"{name:8} {dur:4.1f}s {v:7.0f} {sf:7.0f} {w:7.0f} {v2:5.0f}  {why}")
    log(f"\ngpu mem at end: {gpu_mem()}")


if __name__ == "__main__":
    main()
