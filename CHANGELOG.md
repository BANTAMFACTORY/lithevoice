# Changelog

## Unreleased — Linux support and a rebuilt turn-taking gate

Two pieces of work: making the project run on Linux, and replacing the
barge-in logic after measurement showed the original approach could not work.

### Linux x64 support

The project was Windows/PowerShell only. It now installs and runs on Linux
without disturbing the Windows path.

- `scripts/setup.sh` — Linux port of `setup.ps1`; same flags, same resumable
  and idempotent behaviour.
- `run.sh` — Linux port of `run.ps1`.
- `scripts/models.json` gained `linux_assets` (ubuntu-x64 and vulkan) with
  SHA256s; `download_models.py` handles `.tar.gz`, platform-aware asset
  selection, and executable bits.
- `realtime.py` — binary name, `LD_LIBRARY_PATH`, a `/proc`-based PID guard
  giving Linux the same "never kill an unrelated process" guarantee Windows
  had, and an X11 `XQueryKeymap` implementation of the global `--barge-key`
  hotkey (no root, no `input` group).

Two things worth knowing:

- **Upstream ships no prebuilt Linux CUDA llama.cpp.** `setup.sh` resolves the
  backend to CPU even on NVIDIA; use `--llama-backend vulkan` or build with
  `-DGGML_CUDA=ON` and point `LITHEVOICE_LLAMA_DIR` at it.
- The Linux release tarball resolves SONAMEs through **symlinks**. A flat
  extraction that skips them produces a `llama-server` that will not load.

### Turn-taking and barge-in

The old gate fired on three consecutive Silero frames above 0.5. Measurement
(`tests/bargein_probe.py`) showed why that cannot work: 28% of echo-only frames
score ≥ 0.5, a backchannel is acoustically identical to a real interruption,
and a television scores 77% of frames at p = 1.00 while sitting *further* above
the noise floor than the real user.

- **`SpeechAdmit`** — one verdict per frame from three signals: VAD
  probability, energy above a percentile noise floor, and energy above the echo
  predicted from the playback reference. Consumed by *both* turn-taking and
  barge-in, because the largest source of destroyed replies turned out to be
  noise opening a **turn**, whose answer then cancelled the reply in progress.
- **`BargeGate`** — staged, mostly reversible response: duck at ~96 ms, hold at
  ~352 ms, abandon only at ~1400 ms. Perceived responsiveness comes from the
  first stage, which buys the last one time to be sure. The accumulator leaks
  rather than resetting, so the pause after a comma no longer abandons a
  genuine interruption.
- **`classify_utterance`** — at the turn boundary, short acknowledgements
  release the hold and the reply resumes where it stopped; captures with no
  recognised words are discarded instead of answered.
- **`SpeakerVerifier`** — WeSpeaker ResNet34-LM (26.5 MB ONNX, CPU, ~12 ms per
  second of audio) behind `--speaker-lock`. **Off by default**; see Known
  issues.
- Playback: duck gain is *ramped*, not stepped (a step mid-waveform is a
  click); the echo reference is pushed *before* `write()`, since pushing after
  makes it lag the echo it is meant to predict and quietly disables both the
  AEC and the double-talk test; output buffer raised to 80 ms after measuring
  36 dropouts of 2–60 ms in 13 s of speech.

Measured on the same scenarios and metric:

| | original gate | now |
|---|---|---|
| replies destroyed by noise/echo/TV | 59 | 3–15 (nearly all the television) |
| replies destroyed by a backchannel | 2 | 0–1 |
| real interruptions honoured | 5/5 | 5/5 |
| backchannels kept off the floor | 0 | 3–6 |
| time to go quiet | n/a (no duck) | 64–86 ms |

### Testing

- `tests/bargein_sim.py` — drives the **real** `run_live()` loop through a
  virtual full-duplex device: modelled room echo, background beds, and user
  speech scheduled relative to the reply. Silent, repeatable, no audio
  hardware. `--legacy` and `--oracle-speaker` bound the comparison from both
  sides.
- `tests/bargein_probe.py` — reports what the gate actually sees, which is how
  the thresholds were chosen.
- `tests/bargein_assets.py` — generates the simulated voices and noise beds.
- `tests/test_bargein.py` — rewritten for the staged gate.
- `LITHEVOICE_BARGE_DEBUG=1` names the test that rejected each frame.

### Web UI

- The transcript no longer pushes the controls off-screen. `body` had only
  `min-height`, so `grid-template-rows: minmax(0, 1fr)` had nothing to divide
  and the feed grew the page instead of scrolling. Measured before the fix:
  controls at y=3297 on a 1226 px viewport.
- The controls column scrolls independently, and the visualiser yields height
  before the controls do on short windows.
- **Learn my voice** button for speaker enrolment.

### Lite profile — running without PyTorch

`--lite` (or `--tts-backend onnx`) swaps Kokoro and Silero to ONNX and removes
the last PyTorch dependency. Opt-in; the default path is unchanged.

| | Default | Lite |
|---|---|---|
| venv size | 6059 MB | **390 MB** |
| PyTorch | required | absent |
| synthesis RTF (CPU) | 0.16 | 0.16 |

Verified by building a venv with no torch installed and running both backends
in it. Three non-obvious obstacles, all documented in `docs/LITE.md`: the
`silero-vad` package hard-imports torch even on its ONNX path (and depends on
it), the `misaki[en]` extra pulls `spacy-curated-transformers` which
hard-depends on torch, and spaCy's own torch import is optional
(`thinc/compat.py`, inside `try/except`).

Costs: CPU only, no speaker lock (needs a NumPy Kaldi filterbank), and **no
speed benefit on x86-64** — it is a wash at 4-8 threads and slower at 12.
Choose it for size, not speed. ARM is unmeasured.

### Phone as microphone and speaker

`--phone` turns a browser on the local network into the audio device.
`PhoneInputStream`/`PhoneOutputStream` replace `sounddevice` at the same seam
`tests/bargein_sim.py` already used, so the loop, the gate and the AEC are
untouched. Measured over Wi-Fi from a phone: **494-754 ms voice-to-voice**,
first audio 117-180 ms, with ducking and barge-in working.

WebSocket rather than WebRTC: `aiortc` brings PyAV and ~50 MB, against ~2 MB
for `websockets`, and the browser supplies echo cancellation through
`getUserMedia` regardless of transport. WebRTC remains the right answer off a
LAN.

TLS is not optional — `getUserMedia` requires a secure context, so a
self-signed certificate covering localhost and the LAN IP is generated into
`certs/`. Non-loopback binds also require a token; requests from the machine
itself skip it so `http://localhost:7860` keeps working. See `docs/PHONE.md`.

### Barge-in policy, measured against other stacks

LiveKit's and Pipecat's published interruption defaults were reimplemented as
alternative policies (`tests/policies.py`) and run through the identical ten
scenarios with the identical acoustic front end.

| | ours | ours-v2 | livekit | pipecat |
|---|---|---|---|---|
| replies destroyed | 19 | 18 | 26 | 19 |
| backchannels destroyed | 1 | 1 | 1 | **4** |
| real interruptions honoured | 6/5 | 6/5 | **3/5** | 5/5 |
| time to go quiet | **65 ms** | 67 ms | no duck | no duck |

Neither alternative has a reversible first stage, and both failure modes
follow from that: Pipecat's 0.2 s trigger destroys backchannels, LiveKit's
0.5 s bar misses real interruptions. This compares *policies*, not products —
LiveKit's ML backchannel classifier was not reproduced, and both frameworks
bring transport and multi-user support this project does not have. Single
runs; ±3 on "destroyed" is noise. No shipped default changed.

### Measured and rejected

**ONNX Runtime for Kokoro** (`tests/bench_tts.py`). Same phonemes, same style
vector, same thread budget; only the engine differs.

| threads | PyTorch | ONNX fp32 | ONNX int8 / q8f16 |
|---|---|---|---|
| 1 | 5048 ms | 3900 ms (1.29×) | — |
| 4 | 1487 ms | 1372 ms (1.08×) | ~4750 ms (0.31×) |
| 8 | 987 ms | 921 ms (1.07×) | — |
| 12 | 910 ms | 1068 ms (**0.85×**) | — |

ONNX wins only single-threaded and *loses* by 12 threads; PyTorch's CPU kernels
parallelise better on this model. The quantised variants are ~3× slower —
onnxruntime has no efficient int8 convolution kernel for this architecture and
falls back to slow paths, the same failure that made the WeSpeaker int8 export
unusable. A second TTS backend is not worth carrying for ±8%.

**Smaller first chunk to cut time-to-first-audio.** The benchmark showed
Kokoro runs at RTF ≈ 0.16 on CPU — roughly 6× faster than real time — so once
the first chunk is out, streaming stays comfortably ahead for a single
conversation. The opening chunk is also already short by construction:
`_SPOKEN_STYLE` instructs the model to "Open with a short sentence". Splitting
below sentence granularity would only matter for concurrent streams, where
throughput rather than first-bite latency becomes the constraint.

What the benchmark *did* establish is that thread allocation matters a great
deal — 1 → 8 threads is a 5× speedup, and past 8 there is nothing left. Nothing
currently budgets cores: `llama-server` runs with no `-t`, Smart Turn requests
`os.cpu_count()`, and Kokoro takes what it likes, all concurrently on a
12-core part. That oversubscription is a plausible contributor to the audio
dropouts noted below.

### Documentation

- `docs/BARGE_IN.md` — the turn-taking and barge-in subsystem in full.
- `docs/SCALING.md` — multi-user analysis: measured batching economics for
  Parakeet (45x -> 1207x RTFx) and Kokoro (169x), the ~900 ms request-batching
  latency trap, capacity arithmetic showing the LLM is the constraint rather
  than speech, what is single-user by construction, and a near drop-in path to
  multilingual Parakeet v3.

### Known issues

- **Speaker lock is off by default.** Its `0.40` threshold was calibrated on
  synthetic speech and rejected a real user at `0.32` — the separation is real
  (0.32 against 0.15 for a second person) but the operating point needs
  measuring against a real microphone. It needs a calibration tool before it
  can be trusted on by default.
- **The television case is unsolved** without identity; the acoustic gates
  cannot win it. With a perfect identifier the scenario goes from 13 destroyed
  replies to 1.
- **The 80 ms output buffer fix is unverified.** The dropout count has not been
  re-measured since the change.
- **Multi-speaker conversation is not supported.** `speaker_ok` returns a
  boolean, not an identity; turn attribution, per-speaker history and a floor
  policy would all be new work.
- Synthesis is not interruptible mid-sentence: `cancel` stops playback within
  one 100 ms block, but Kokoro finishes the sentence it started. Nothing is
  heard; it costs CPU, not perceived latency.
