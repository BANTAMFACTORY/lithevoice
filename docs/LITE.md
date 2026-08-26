# The Lite Profile — Running Without PyTorch

LitheVoice scales in two directions. Up, it becomes a service handling many
callers ([SCALING.md](SCALING.md)). Down, it becomes something you can put on
a laptop, a thin container, or eventually a phone. This document is the second
direction, and it is about **deployment size, not speed**.

---

## 1. Why: the runtime costs as much as the models

Measured on the reference machine:

| | Size |
|---|---|
| **Virtual environment, default stack** | **6059 MB** |
| — `nvidia/*` (CUDA libraries) | 2742 MB |
| — `torch` | 1544 MB |
| — `triton` (a torch dependency) | 440 MB |
| — `onnxruntime-gpu` | 438 MB |
| — everything else | ~895 MB |
| **Pinned model weights** | **5.1 GB** |

Nearly 4.7 GB of that environment exists to run **one 82M-parameter model**.
Parakeet, Smart Turn and the speaker verifier are already ONNX; Gemma is a
llama.cpp binary. PyTorch is in the tree for Kokoro synthesis and the Silero
VAD wrapper, and nothing else.

## 2. Result

A genuinely torch-free environment, built and executed:

```
torch available in this interpreter: False
VAD  prob(silence): 0.0006
VAD  prob(noise)  : 0.0017
TTS  2 chunks, 4.62s of audio in 728 ms (RTF 0.16)
torch imported at end: False
```

| | Default | Lite |
|---|---|---|
| venv size | 6059 MB | **390 MB** |
| PyTorch | required | **absent** |
| Synthesis RTF (CPU) | 0.16 | **0.16** |
| GPU synthesis | yes | no |
| Speaker lock | available | unavailable (§5) |

**≈15× smaller, with the same measured synthesis throughput.**

## 3. Using it

```bash
./run.sh --lite                      # ONNX synthesis + ONNX VAD
./run.sh --tts-backend onnx          # just the synthesiser
```

Both are opt-in. Without them the default path is byte-for-byte what it was;
`Models(tts_backend="torch", vad_backend="torch")` is still the default and the
PyTorch code path is untouched.

To build an actually small install, use `requirements-lite.txt` rather than
`requirements.txt`, and fetch the two extra artifacts it names.

## 4. What it swaps, and the dependency archaeology

Each of these was a separate obstacle, and the reasons are not obvious from
reading `requirements.txt`.

**Kokoro → `lite_backends.OnnxKokoro`.** `onnx-community/Kokoro-82M-v1.0-ONNX`,
pinned in `scripts/models.json` and fetched with
`download_models.py --include-lite`. It presents the same generator contract as
`kokoro.KPipeline` — yielding `(graphemes, phonemes, audio)` — so
`speak_stream()` cannot tell the difference. Style vectors come from the
repo's `voices/*.bin` as raw float32, because reading the `.pt` voices would
need torch.

**Silero → `lite_backends.OnnxSilero`.** The `silero-vad` package ships the
`.onnx` model, but `silero_vad/model.py` opens with an *unconditional*
`import torch` even on its ONNX path, so the package cannot be used torch-free.
Worse, `pip install silero-vad` drags torch in as a declared dependency. The
model file is therefore used directly and located **without importing the
package** (`importlib.util.find_spec`, which does not execute it), or supplied
via `LITHEVOICE_SILERO_ONNX`. `TurnDetector` calls `.prob()` when the VAD
object has one, so its logic is unchanged.

**`misaki[en]` → `misaki` + `spacy`.** This one is easy to miss. The `[en]`
extra pulls `spacy-curated-transformers`, which **hard-depends on torch** — so
a lite venv built with `misaki[en]` silently reinstalls the 4.7 GB you were
trying to remove. That extra exists for the transformer G2P path (`trf=True`),
which is not used here; `en.G2P(trf=False)` needs only spaCy and
`en_core_web_sm`.

**spaCy itself is fine.** It reaches torch through `thinc/compat.py`, inside a
`try/except ImportError` that sets `has_torch = False` when absent. Verified by
reading the source and then by running with no torch installed at all.

## 5. Feature parity

Lite is not a stripped-down build. Everything except two things works
identically, because the rest of the stack was never PyTorch to begin with.

| Feature | Default | Lite |
|---|---|---|
| Staged barge-in (duck / hold / cancel) | yes | **yes** — pure NumPy |
| Backchannel + wordless-capture handling | yes | **yes** |
| Echo cancellation (`--aec`) | yes | **yes** — pure NumPy |
| Smart Turn semantic endpointing | yes | **yes** — already ONNX |
| Parakeet STT, speculative | yes | **yes** — already ONNX |
| Gemma / llama.cpp, on GPU | yes | **yes** — a separate binary |
| Web dashboard, personas, all 12 voices, speed | yes | **yes** |
| Direct-audio mode, key barge-in | yes | **yes** |
| **GPU synthesis** | yes | **no** — CPU only |
| **Speaker lock** (`--speaker-lock`) | yes | **no** — needs torchaudio |

Only two capabilities are lost, and one of them is off by default. The
practical cost is latency, not features: first audio is ~700-800 ms instead of
~145 ms, because synthesis is on the CPU.

## 6. What it costs

**No speaker lock.** `SpeakerVerifier` uses
`torchaudio.compliance.kaldi.fbank`. Reproducing Kaldi's filterbank in NumPy
exactly is fiddly, and a subtle mismatch would silently corrupt embeddings
rather than fail loudly, so `--lite` disables speaker lock and says so. It is
off by default anyway (see [BARGE_IN.md §7](BARGE_IN.md)). Writing a verified
NumPy filterbank is the obvious next step — `whisper_features.py` already
demonstrates the pattern for Smart Turn.

**CPU only.** `onnxruntime-gpu` would need CUDA libraries, which in the current
install it finds *inside* torch's bundled lib directory. Wanting GPU synthesis
means wanting torch or the `nvidia/*` wheels back, and most of the size returns.

**No speed win on x86-64.** From `tests/bench_tts.py`, matched threads:

| threads | PyTorch | ONNX | winner |
|---|---|---|---|
| 1 | 5048 ms | 3900 ms | ONNX 1.29× |
| 4 | 1487 ms | 1372 ms | ONNX 1.08× |
| 8 | 987 ms | 921 ms | ONNX 1.07× |
| 12 | 910 ms | 1068 ms | **PyTorch 1.17×** |

A wash in the middle, and PyTorch wins once threads are plentiful. **Choose
this profile for size, not speed.** Quantised ONNX Kokoro is ~3× *slower*
(§ CHANGELOG, *Measured and rejected*) — do not reach for it.

## 7. Not done yet

- **NumPy Kaldi filterbank**, to restore speaker lock under `--lite`.
- **aarch64 benchmarking.** Every timing here is x86-64. PyTorch's x86 kernels
  are what won at 12 threads, which says nothing about ARM — and ARM is where
  this profile is aimed. The comparison could plausibly invert.
- **A pinned source for `silero_vad.onnx`** that does not involve installing a
  torch-dependent package.
- **Smaller Kokoro variants.** The pinned ONNX model is fp32 (325 MB); the
  repo also publishes fp16 (~163 MB) and int8 (~82 MB). Given quantised
  inference measured *slower*, these are interesting for size on a phone, not
  for speed — and quality is unverified.
- **End-to-end on a real low-power device.** Thread scaling says 1 core is
  ~5 s per sentence and 8 cores ~1 s, so roughly 4 cores looks like the
  practical floor. That is arithmetic, not a measurement on real hardware.
