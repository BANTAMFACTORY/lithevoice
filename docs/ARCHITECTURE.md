# LitheVoice Architecture

This document describes the runtime boundaries and the decisions that keep the
voice loop responsive. See the root README for installation and user-facing
commands.

## Runtime Processes

LitheVoice consists of two local processes:

1. `realtime.py` owns audio capture/playback, VAD, turn detection, Parakeet,
   Kokoro, conversation state, interruption, and the localhost web server.
2. `llama-server` owns Gemma inference and its KV cache. LitheVoice starts it
   when no healthy server answers on the configured URL.

The runtime records the PID of the server it starts. Restart operations target
that PID only. An unrelated llama-server process is never terminated by image
name.

## Standard Turn

1. The microphone callback queues 512-sample float32 blocks at 16 kHz.
2. Silero scores every 32 ms block. Hysteresis separates speech start and
   continuation thresholds.
3. After roughly 200 ms of silence, Parakeet begins speculative recognition in
   a worker while Smart Turn scores semantic completeness.
4. A complete score dispatches immediately. An incomplete score waits for
   resumed speech or the configured silence timeout.
5. The transcript is usually ready by dispatch because recognition overlapped
   the turn-decision window.
6. Gemma streams response fragments through llama.cpp's `/completion`
   endpoint. The system prompt, few-shot persona, and recent history use a
   stable prompt prefix with `cache_prompt` enabled.
7. Completed sentences flow into Kokoro without waiting for the full reply.
8. Kokoro audio is written in 100 ms blocks so cancellation takes effect
   within one block.

## Direct Audio Turn

`--direct-audio` encodes the captured utterance as PCM16 WAV, sends it through
the OpenAI-compatible chat endpoint as `input_audio`, and lets Gemma's mmproj
produce the audio embeddings. Parakeet runs after the response as a reference
and is not on the critical path.

Past audio turns stay in conversation history. With one llama-server slot, the
unchanged prompt/audio prefix can reuse the KV cache.

## Device Policy

The setup script selects one of two tested profiles:

| Profile | llama.cpp | Kokoro | Parakeet | VAD / Smart Turn |
|---|---|---|---|---|
| NVIDIA | CUDA, automatic layer fit | CUDA | CPU int8 | CPU |
| CPU-only | CPU | CPU | CPU int8 | CPU |

Parakeet's quantized operators do not map efficiently to the tested CUDA
execution provider. The optional fp32 GPU graph is much larger and did not win
the latency comparison, so it is not downloaded by default.

llama.cpp runs with `-ngl auto` and its fit behavior enabled. Explicit
`--llm-device cpu` maps to zero GPU layers; `gpu` maps to all layers.

## llama.cpp Contract

The pinned build is `b9867`. The server is started with:

```text
-m <model> --mmproj <projector>
-ngl auto -c 4096 -np 1
--jinja --chat-template-file gemma.jinja
--host 127.0.0.1 --port 8080
```

The fixed template is required for the selected fine-tune. Its embedded
template can introduce a thinking preamble that is unsuitable for low-latency
spoken output. A single slot prevents turns from rotating across independent
KV caches.

The CUDA release and matching runtime archive are both pinned and verified.
PyTorch's bundled CUDA libraries are also added to the child process PATH.

## Interruption

Three mechanisms share one cancellation event:

- The web interrupt button sets the event directly.
- Key interruption polls the backtick/tilde key and ignores microphone input
  while playback is active.
- Voice barge-in escalates through the staged gate described below.

Cancellation stops playback between 100 ms blocks. Closing the streaming HTTP
response also aborts in-flight LLM generation. Only the text actually spoken
is committed to conversation history.

## Speech Admission And Voice Barge-In

One verdict per 32 ms frame, produced by `SpeechAdmit`, answers "is this the
user talking to us?" and is consumed by both turn-taking and barge-in. Sharing
it is deliberate: measurement showed the barge gate was not the dominant
problem. A television or a desk fan would open a *turn*, and dispatching that
turn cancelled whatever the assistant was already saying, so fixing only the
barge path left replies dying by another route.

The verdict combines the VAD probability, frame energy above an adaptive noise
floor, and frame energy above the echo predicted from the playback reference.
The floor is a low percentile of the recent past rather than an average of
quiet frames — an average never initialises while a television is talking,
which silently disables the test exactly when it is needed. Reference coupling
is learned on echo-only frames, which makes the third signal a double-talk
detector independent of whether the AEC is enabled.

`BargeGate` stages the response over a leaky accumulator of qualifying frames:

| Stage | Threshold | Effect | Reversible |
|---|---|---|---|
| duck | ~96 ms | reply attenuated to `DUCK_GAIN` | yes |
| hold | ~352 ms | reply silent, position kept | yes |
| cancel | ~1400 ms | turn abandoned | no |

The accumulator leaks instead of resetting, because requiring consecutive
frames abandons a genuine interruption at the pause after a comma. Perceived
responsiveness comes from the first stage, so the destructive one can afford
to wait. At the turn boundary the transcript resolves what acoustics cannot:
short acknowledgements release the hold and the reply resumes, and captures
with no words are discarded rather than answered.

Speaker identity is resolved at the turn boundary rather than per frame.
`SpeakerVerifier` (WeSpeaker ResNet34-LM, 26.5 MB ONNX, CPU, ~12 ms per second
of audio) embeds the captured utterance and compares it to an enrolled
centroid; a mismatch discards the turn. It is opt-in via `--speaker-lock`
because the default threshold is not yet calibrated against real microphones.
`SpeechAdmit` also takes a per-frame `speaker_ok` hook, but nothing fast
enough to use it ships today: an embedding needs about a second of audio and
the duck fires in a tenth of one, so identity deliberately does not gate the
fast path.

`tests/bargein_sim.py` exercises all of this against the real loop through a
virtual full-duplex device, with `--legacy` and `--oracle-speaker` to bound
the comparison from both sides.

[docs/BARGE_IN.md](BARGE_IN.md) documents the whole subsystem in detail: the
measurements behind each threshold, the failure modes they fix, and the tuning
knobs. [docs/SCALING.md](SCALING.md) records what is single-user by
construction — module-level singletons, the `-np 1` llama.cpp slot, one
`Models` instance and one audio device pair — and where the bottleneck moves
once there is more than one stream.

## Echo Cancellation

The optional AEC is a partitioned frequency-domain NLMS filter. Playback blocks
feed a reference buffer before acoustic output; microphone frames are cleaned
against that reference. Double-talk gating protects the learned path.

The algorithm performs well against the synthetic room path in `tests/test_aec.py`.
It is experimental for real microphone arrays because driver DSP can make the
captured path non-linear or time-varying. Key interruption remains the reliable
open-speaker mode.

## Web Protocol

`webui.py` binds a `ThreadingHTTPServer` to localhost only.

- `GET /` serves the self-contained dashboard.
- `GET /events` provides Server-Sent Events.
- `POST /control` changes persona, voice, speed, interruption, or shutdown.

The bus retains only the latest config/state snapshot for newly connected
clients. Conversation and audio-spectrum events are transient.

## Files And Caches

All paths are repository-relative unless a `LITHEVOICE_*` environment override
is present.

- `models/gemma_4_e2b`: verified Gemma and mmproj files
- `models/parakeet-int8`: pinned local Parakeet CPU graph
- `models/parakeet-fp32`: optional GPU graph
- `models/huggingface`: project-local Kokoro and Smart Turn cache
- `llama.cpp/bin`: pinned native runtime
- `llama.cpp/backend.json`: installed build metadata
- `llama.cpp/server.pid`: PID ownership marker

These directories are installation products and are excluded from Git.

## Timing

`handle_turn` measures processing from dispatch. The live loop separately
passes the elapsed turn-decision interval:

```text
true voice-to-voice = turn decision + dispatch-to-first-audio
```

For Smart Turn, the decision interval includes accumulated silence plus Smart
Turn inference. For timeout completion, it uses the frame-rounded silence
threshold. Simulated WAV runs have no live end-of-speech clock and therefore
report pipeline-to-audio instead.

## Download Integrity

`scripts/models.json` pins repository revisions and the critical artifact
hashes. `download_models.py` uses Hugging Face's resumable cache for model
files, a resumable range request for GitHub archives, and SHA256 validation
before extraction. `.complete` sentinels prevent interrupted local Parakeet
directories from being treated as valid offline models.
