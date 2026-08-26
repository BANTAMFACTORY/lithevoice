"""Compare Kokoro synthesis backends on CPU: PyTorch vs ONNX Runtime.

Measured on CPU-only runs, text-to-speech is ~85-90% of the latency to first
audio (1509 ms against a 39 ms LLM first token), so it is the only part of the
CPU path worth optimising. This answers whether swapping the runtime helps,
with numbers rather than opinion.

Both backends are given the same phonemes, the same voice style vector and the
same thread budget, so the only variable is the inference engine. The G2P
frontend is shared and timed separately.

    ./.venv/bin/python tests/bench_tts.py
    ./.venv/bin/python tests/bench_tts.py --threads 4 --repeats 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))

ONNX_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
ONNX_REVISION = "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
# fp32 first; the quantised ones are tried and skipped if this build of
# onnxruntime lacks a kernel for them.
VARIANTS = ["onnx/model.onnx", "onnx/model_q8f16.onnx", "onnx/model_quantized.onnx"]

TTS_SR = 24000
VOICE = "af_heart"

# A short opening clause and a full sentence: the first is what a
# latency-oriented chunking strategy would emit, the second is what the
# current sentence splitter emits.
TEXTS = [
    ("clause", "Sure, here is the part that matters."),
    ("sentence", "Hey there, good to hear from you today. I hope the "
                 "afternoon is treating you kindly so far."),
]


def median_ms(values):
    return statistics.median(values) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=4,
                    help="thread budget given to BOTH backends (default 4)")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline

    print(f"Kokoro CPU synthesis: PyTorch vs ONNX Runtime "
          f"({args.threads} threads, median of {args.repeats})\n")

    pipeline = KPipeline(lang_code="a", device="cpu")
    pack = pipeline.load_voice(VOICE)
    vocab = pipeline.model.vocab

    sessions = {}
    for rel in VARIANTS:
        try:
            path = hf_hub_download(ONNX_REPO, rel, revision=ONNX_REVISION)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = args.threads
            opts.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessions[rel.split("/")[-1]] = ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:
            print(f"  skipping {rel}: {str(exc)[:90]}")
    print()

    rows = []
    for label, text in TEXTS:
        # --- shared frontend -------------------------------------------
        t0 = time.perf_counter()
        ps, _ = pipeline.g2p(text)
        g2p_ms = (time.perf_counter() - t0) * 1000

        ids = [vocab.get(p) for p in ps]
        ids = [i for i in ids if i is not None]
        input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
        style = pack[len(ps) - 1].numpy().astype(np.float32).reshape(1, -1)

        # --- PyTorch ----------------------------------------------------
        torch_times, audio_s = [], None
        for _ in range(args.repeats + 1):
            t0 = time.perf_counter()
            out = pipeline.model(ps, pack[len(ps) - 1], 1.0, return_output=True)
            torch_times.append(time.perf_counter() - t0)
            audio_s = len(out.audio) / TTS_SR
        torch_times = torch_times[1:]                     # drop warm-up
        ref_audio = np.asarray(out.audio, np.float32)

        rows.append((label, "PyTorch (current)", median_ms(torch_times),
                     audio_s, g2p_ms, 1.0))

        # --- ONNX variants ----------------------------------------------
        for name, sess in sessions.items():
            try:
                times, wav = [], None
                for _ in range(args.repeats + 1):
                    t0 = time.perf_counter()
                    wav = sess.run(None, {"input_ids": input_ids,
                                          "style": style,
                                          "speed": np.array([1.0], np.float32)})[0]
                    times.append(time.perf_counter() - t0)
                times = times[1:]
                wav = np.asarray(wav, np.float32).reshape(-1)
                # Sanity: does it produce comparable audio at all?
                ratio = len(wav) / max(1, len(ref_audio))
                rows.append((label, f"ONNX {name}", median_ms(times),
                             len(wav) / TTS_SR, g2p_ms, ratio))
            except Exception as exc:
                print(f"  {label}/{name} failed: {str(exc)[:90]}")

    # --- report ---------------------------------------------------------
    hdr = (f"{'text':10} {'backend':26} {'synth':>9} {'+g2p':>8} "
           f"{'audio':>7} {'RTF':>6} {'len vs torch':>13}")
    print(hdr); print("-" * len(hdr))
    baseline = {}
    for label, backend, ms, audio, g2p_ms, ratio in rows:
        if backend.startswith("PyTorch"):
            baseline[label] = ms
        rtf = (ms / 1000) / audio if audio else float("nan")
        speed = f"{baseline[label]/ms:.2f}x" if label in baseline and ms else ""
        print(f"{label:10} {backend:26} {ms:7.0f}ms {ms+g2p_ms:6.0f}ms "
              f"{audio:6.2f}s {rtf:6.2f} {ratio:12.2f}  {speed}")

    print("\nsynth = model inference only; +g2p adds the shared text frontend.")
    print("RTF < 1 means faster than real time. 'len vs torch' near 1.00 means "
          "the backend produced equivalent audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
