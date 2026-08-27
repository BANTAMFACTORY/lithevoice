"""Headless end-to-end test of the realtime pipeline.

Simulates the live mic path without audio hardware:
  1. Kokoro (GPU) generates test utterances of several lengths.
  2. Each is streamed frame-by-frame through the SAME Silero VADIterator loop
     realtime.py uses (with leading/trailing silence), verifying start/end
     detection and measuring the VAD's end-of-speech detection delay
     (stream-time: how much silence must elapse before "end" fires).
  3. The captured segment goes through Parakeet STT (timed vs length).
  4. Kokoro synthesizes the canned reply (timed to first audio).

Prints a stage-by-stage latency budget and the projected voice-to-voice for:
  - serial   : VAD delay + STT + TTS-first   (current realtime.py)
  - parallel : VAD delay + max(STT, TTS-first)  (canned reply doesn't need
               the transcript, so TTS can start immediately at end-of-speech)
"""
import time

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # the heavy stack is optional; see scripts/setup.sh
    # Must happen at module level: a missing import breaks pytest COLLECTION,
    # which no in-test skip can rescue. Touching pytest only on the failure path
    # keeps `python tests/<file>.py` working unchanged when torch IS installed.
    import pytest

    pytest.skip(
        "needs torch and the full voice stack (scripts/setup.sh)",
        allow_module_level=True,
    )

SR = 16000
FRAME = 512  # 32 ms
MIN_SILENCE_SWEEP = [200, 400, 600, 800]  # find smallest that survives mid-pauses
SPEECH_PAD_MS = 30

UTTERANCES = [
    ("short", "Hey, how are you?"),
    ("medium", "Can you remind me what I have going on this afternoon?"),
    ("long", "So I was thinking about heading to the store later today, "
             "and I wanted to know if you think it's going to rain, "
             "or if I should just walk there instead of driving."),
]
REPLY = "Hey bud, good to hear from you. Hope your day is going well."
SENTENCE_SPLIT = r"(?<=[.!?])\s+"


def log(m):
    print(m, flush=True)


def resample_24k_to_16k(audio):
    n = int(round(len(audio) * SR / 24000))
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, len(audio), endpoint=False),
                     audio).astype(np.float32)


def trim_trailing_silence(audio, thresh=1e-3):
    """Kokoro pads silence after speech; find where speech really ends."""
    idx = np.where(np.abs(audio) > thresh)[0]
    return audio[: idx[-1] + 1] if len(idx) else audio


def main():
    log("Loading models...")
    from silero_vad import load_silero_vad, VADIterator
    import onnx_asr
    from kokoro import KPipeline

    vad_model = load_silero_vad()
    stt = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")
    tts = KPipeline(lang_code="a", device="cuda")

    # warmups
    stt.recognize(np.zeros(SR, np.float32), sample_rate=SR)
    for _ in tts("Warm up.", voice="af_heart"):
        pass
    torch.cuda.synchronize()
    log("Models ready.\n")

    # --- TTS reply timing (canned reply, sentence-streamed) ---
    def tts_first_audio():
        t0 = time.perf_counter()
        first = None
        for _, _, a in tts(REPLY, voice="af_heart", split_pattern=SENTENCE_SPLIT):
            if first is None:
                torch.cuda.synchronize()
                first = time.perf_counter() - t0
        return first

    tts_first = min(tts_first_audio() for _ in range(3))
    log(f"TTS first-audio (canned reply, best of 3): {tts_first*1000:.0f} ms\n")

    def stream_vad(utt16, min_silence_ms):
        """Stream utterance+silence through VADIterator; return
        (vad_delay_ms, captured_segment, cutoff?) or None if no end fired."""
        stream = np.concatenate([
            np.zeros(SR // 2, np.float32), utt16, np.zeros(2 * SR, np.float32)])
        speech_end_sample = SR // 2 + len(utt16)  # true end (silence trimmed)
        vad = VADIterator(vad_model, sampling_rate=SR,
                          min_silence_duration_ms=min_silence_ms,
                          speech_pad_ms=SPEECH_PAD_MS)
        started = False
        captured = []
        result = None
        for off in range(0, len(stream) - FRAME + 1, FRAME):
            frame = stream[off:off + FRAME]
            res = vad(torch.from_numpy(frame), return_seconds=False)
            if res and "start" in res:
                started = True
            if started:
                captured.append(frame)
            if res and "end" in res:
                end_fired_at = off + FRAME
                seg = np.concatenate(captured)
                cutoff = end_fired_at < speech_end_sample  # fired mid-speech
                result = ((end_fired_at - speech_end_sample) / SR * 1000, seg, cutoff)
                break
        vad.reset_states()
        return result

    # --- sweep min_silence per utterance: does it survive natural pauses? ---
    utts = {}
    for name, text in UTTERANCES:
        chunks = [a for _, _, a in tts(text, voice="am_michael")]
        utt = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        utt16 = trim_trailing_silence(resample_24k_to_16k(np.asarray(utt, np.float32)))
        utts[name] = utt16

    log(f"=== min_silence sweep (cutoff = VAD fired mid-utterance) ===")
    log(f"{'utt':8}" + "".join(f" {ms:>7}ms" for ms in MIN_SILENCE_SWEEP))
    safe = {}
    delays = {}
    for name, utt16 in utts.items():
        row = f"{name:8}"
        for ms in MIN_SILENCE_SWEEP:
            r = stream_vad(utt16, ms)
            if r is None:
                row += f" {'none':>9}"
                continue
            delay, seg, cutoff = r
            row += f" {'CUT':>9}" if cutoff else f" {delay:7.0f}ms"
            if not cutoff and name not in safe:
                safe[name] = ms
                delays[name] = (delay, seg)
        log(row)

    # smallest min_silence safe for ALL utterances
    overall = max(safe.values()) if len(safe) == len(utts) else MIN_SILENCE_SWEEP[-1]
    log(f"\nsmallest cutoff-safe min_silence across utterances: {overall} ms\n")

    # --- full budget at the safe setting ---
    log(f"=== latency budget @ min_silence={overall} ms ===")
    log(f"{'utt':8} {'len':>5} {'VAD':>6} {'STT':>6} {'serial':>8} {'parallel':>9}")
    for name, utt16 in utts.items():
        r = stream_vad(utt16, overall)
        if r is None:
            log(f"{name:8}  VAD failed")
            continue
        vad_delay, seg, cutoff = r
        t0 = time.perf_counter()
        transcript = stt.recognize(seg, sample_rate=SR)
        stt_ms = (time.perf_counter() - t0) * 1000
        serial = vad_delay + stt_ms + tts_first * 1000
        parallel = vad_delay + max(stt_ms, tts_first * 1000)
        flag = "  [CUT!]" if cutoff else ""
        log(f"{name:8} {len(utt16)/SR:4.1f}s {vad_delay:5.0f} {stt_ms:5.0f} "
            f"{serial:7.0f} {parallel:8.0f}{flag}")
        log(f'         "{transcript.strip()}"')


if __name__ == "__main__":
    main()
