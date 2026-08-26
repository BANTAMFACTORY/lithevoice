# Scaling Beyond One Conversation

LitheVoice serves a single local speaker on one machine, and everything about
it is tuned for that: the shortest possible path from end-of-speech to first
sound. Serving many conversations at once is a different problem with a
different bottleneck, and most of the single-stream tuning stops mattering.

This document records what was measured, what already scales, what would have
to change, and — importantly — which optimisations were tried and rejected, so
a future upgrade does not rediscover them the hard way.

---

## 1. Two axes that "realtime factor" blurs

Almost every speech benchmark quotes a realtime factor, and it hides the
distinction that decides this whole design.

| | Meaning | What moves it |
|---|---|---|
| **First-bite latency** | how long after you stop talking before the assistant makes a sound | model speed at batch 1, chunk size, thread count, transport |
| **Throughput** | seconds of audio processed per second of wall clock, across everyone | batching, GPU, concurrency |

One conversation cares only about the first. A hundred conversations care
almost only about the second. **Optimisations for one are frequently harmful
to the other** — batching multiplies throughput and either does nothing for
first-bite latency or actively worsens it, because you wait to fill a batch.

Everything below is a consequence of that tension.

---

## 2. Measured batching economics

Both models this system uses batch extremely well on a single RTX 4090.

### Speech recognition — Parakeet TDT 0.6B

From prior Parakeet fast-batching measurements,
transcribing a 56.54 s chunk replicated across the batch:

| Batch | Time (s) | RTFx | Speedup | Marginal efficiency |
|---|---|---|---|---|
| 1 | 1.233 | 45.9× | 1.00× | >90% |
| 4 | 1.322 | 171.1× | 3.73× | >90% |
| 8 | 1.498 | 301.9× | 6.58× | 82.3% |
| 16 | 1.731 | 522.5× | 11.39× | 71.2% |
| 32 | 2.243 | 806.8× | 17.59× | 55.0% |
| 64 | 3.612 | 1001.9× | 21.84× | 34.1% |
| 128 | 5.994 | **1207.4×** | 26.33× | 20.6% |

### Speech synthesis — Kokoro 82M

From prior Kokoro batching measurements, batch 32
at `max_chars=200` on a 4090: **276.15 s of audio in 1.63 s ≈ 169× realtime.**

For comparison, this repo measures Kokoro single-stream on CPU at
**RTF ≈ 0.16** (~6× realtime, `tests/bench_tts.py`).

### The reading that matters

Peak RTFx is the wrong target for conversation. Marginal efficiency collapses
past batch 16–32, so **batch 8–32 is the conversational sweet spot** — 300–800×
realtime at 55–82% efficiency, without the queueing delay that batch 128
implies. Batch 128 is for offline archive processing, not for people waiting
to be answered.

---

## 3. The latency floor, and why it is the whole design problem

The same Parakeet repo measures end-to-end request latency:

| Audio length | Mean latency | Median | p95 |
|---|---|---|---|
| 10 s | 932.6 ms | 933.2 ms | 956.0 ms |
| 60 s | 993.1 ms | 993.4 ms | 1021.9 ms |

Six times the audio costs **6% more latency**. That is not a model that scales
with input length — it is a **~900 ms fixed overhead** per request: HTTP,
serialisation, scheduling, padding, model invocation.

Now compare against what LitheVoice does today. STT costs approximately
**zero** wall-clock, because recognition runs *speculatively during the silence
window* while Smart Turn is still deciding whether the turn ended — by the time
the turn is confirmed the transcript is usually already in hand. Real logs
routinely show `STT wait: 0 ms`, and total voice-to-voice of **516–597 ms**.

> **Naively replacing local STT with a batched HTTP service would add ~900 ms
> to a 550 ms budget — nearly tripling it — while making the throughput number
> look twenty times better.** This is the single most important trap in this
> document.

The resolution is that batching is not the problem; *request-response*
batching is. What a multi-user voice server needs is **continuous (dynamic)
micro-batching in-process**: a scheduler that accumulates whatever utterances
arrived in the last 10–30 ms, runs them as one batch, and returns them
individually. That keeps per-request latency near the batch-1 GPU time (tens of
ms for a 3 s utterance) while still getting most of the throughput multiple.
The same repo's `/transcribe` endpoint — "automatic batching of incoming
requests", low VRAM — is that pattern; `/transcribe_batch` is the offline one.

Design rule: **the batch window is a latency budget, not a throughput knob.**
Pick it from the latency you can afford (10–30 ms), and take whatever batch
size that yields.

---

## 4. Where the time actually goes in Kokoro

From the batched profile, per forward pass:

```
Final processing and inverse STFT : 1.1648 s   ← 71%
Resblocks 1                       : 0.2484 s
Source generation / STFT / rest   : ~0.22 s
```

**Kokoro is iSTFT- and convolution-bound, not matmul-bound.** That single fact
explains two independent local measurements:

- ONNX Runtime fp32 was a wash against PyTorch (1.07× at 8 threads, *slower*
  at 12) — there is no matmul-heavy region for a different runtime to win.
- Quantised ONNX was ~3× **slower**: int8 buys nothing on an inverse STFT, and
  onnxruntime has no efficient int8 convolution kernel for this architecture.
  The identical failure made the WeSpeaker int8 export unloadable
  (`NOT_IMPLEMENTED: ConvInteger`).

Anyone optimising Kokoro should aim at the iSTFT and the resblocks — batching,
a fused iSTFT, fp16 on GPU — and should expect nothing from quantisation or a
runtime swap. See CHANGELOG, *Measured and rejected*.

---

## 5. Capacity arithmetic

Per-stream CPU work that does **not** batch, measured on this machine
(Ryzen 9 5900X, 2 torch threads):

| Component | Median | Cadence | CPU per user |
|---|---|---|---|
| Silero VAD + `SpeechAdmit` + `TurnDetector` | 0.48 ms | every 32 ms | **1.5%** of a core |
| `BargeGate` arithmetic | ~0 ms | every 32 ms | ~0% |
| Smart Turn | 112.8 ms | ~1 per turn | 0.94% of a core |
| `SpeakerVerifier` | 18.0 ms | ~1 per turn | 0.15% of a core |

**≈2.6% of one core per concurrent user**, assuming a turn every ~12 s. On a
12-core box that is a few hundred users before the always-on work saturates —
so the per-stream conversational machinery is *not* the constraint.

GPU side, per user, assuming a 3 s utterance and a 4 s reply every 12 s:

- **STT:** 0.25 audio-seconds/second/user ÷ ~500× RTFx ≈ **0.05%** of the GPU
- **TTS:** 0.33 audio-seconds/second/user ÷ ~169× ≈ **0.2%** of the GPU

Both are rounding errors. **On one 4090, speech in and speech out are not what
limits how many people you can serve.** The LLM is.

### 5.1 Concurrent-session envelope

Extending that per-user cost to a population. Assumes a 3 s utterance and a 4 s
reply per turn, Parakeet at 500× (batch ~16) and Kokoro at 169× (batch 32).

| Users | Turns/s | STT GPU | TTS GPU | **Speech GPU** | CPU cores | of 12 cores |
|---|---|---|---|---|---|---|
| **Busy — a turn every 8 s** ||||||
| 20 | 2.5 | 1.50% | 5.92% | **7.4%** | 0.5 | 4% |
| 50 | 6.2 | 3.75% | 14.79% | **18.5%** | 1.3 | 11% |
| 100 | 12.5 | 7.50% | 29.59% | **37.1%** | 2.6 | 22% |
| **Normal — a turn every 12 s** ||||||
| 20 | 1.7 | 1.00% | 3.94% | **4.9%** | 0.5 | 4% |
| 50 | 4.2 | 2.50% | 9.86% | **12.4%** | 1.3 | 11% |
| 100 | 8.3 | 5.00% | 19.72% | **24.7%** | 2.6 | 22% |

**100 concurrent voice-to-voice conversations consume roughly a quarter of one
4090 for speech, and about a fifth of a 12-core CPU.** Even at a brisk
eight-second turn cadence it is under 40%. VRAM on the speech side is
similarly undemanding: Kokoro measured at 1.07 GB, and Parakeet's 20 GB figure
belongs to the batch-64–128 *offline* profile, not the batch-8–32 band
conversation actually wants.

So the headline is fair: **with the LLM offloaded, 20–100 simultaneous
realtime conversations on a single 24 GB device is arithmetically comfortable,
and speech is not what stops you.**

### 5.2 What actually binds first

The averages above are the easy part. In rough order of what would break:

1. **Latency budget, not throughput.** This system is at 516–597 ms
   voice-to-voice today. Add an API LLM's TTFT (200–500 ms), network transport
   and a jitter buffer (50–150 ms), and one micro-batch window per stage
   (10–30 ms each) and you land near 800–1200 ms. Still conversational, but the
   margin is gone — and *tail* latency under load, not the mean, is what makes
   a voice agent feel broken.
2. **Burstiness at turn boundaries.** Duty cycle is an average; turns are not
   Poisson-smooth. Many sessions finishing at once produce a thundering herd of
   simultaneous STT-then-TTS starts. Micro-batching is exactly the mitigation,
   but queue depth and admission control are what need load-testing.
3. **LLM KV cache, if local.** The real VRAM competitor. Naively, 100 sessions
   × 4096 context is prohibitive. Three things make it tractable here:
   personas are a shared prefix (cached once, not per session), replies are
   short by construction, and per-session history is a few hundred tokens. Cut
   `-c`, quantise the KV cache, and lean on automatic prefix caching.
4. **The Python frame loop.** Each stream steps the gate at 31 Hz; 100 streams
   is ~3,100 interpreter iterations per second. The heavy parts (Silero,
   numpy) release the GIL, so this may well be fine — but it is single-process
   today and it is the first thing to shard across workers, or to replace with
   batched GPU VAD.

> All of the above are component benchmarks and arithmetic, **not a load
> test**. They establish an ordering — LLM ≫ TTS > STT > per-stream CPU — and
> an envelope. Nothing here has been measured under real concurrency, and the
> figures that would matter most (p95/p99 voice-to-voice at N sessions) do not
> exist yet.

---

## 6. The LLM is the constraint

This is why the interesting version of the question is "what if we are *not*
providing the LLM in the middle".

### 6.1 Today: deliberately single-user

`llama-server` is started with `-np 1`, and the comment explains why:

> `-np 1`: single slot, so every turn reuses the same KV prefix cache
> (4 default slots rotated → TTFT crept 45→450 ms live).

That is correct for one user and exactly wrong for many: concurrent sessions
serialise behind one slot. Raising `-np` restores concurrency but splits the KV
cache, reproducing the original **10× first-token regression**. llama.cpp is
not the right server for this shape of load.

### 6.2 External API

Removes the GPU cost entirely and makes the box a pure audio front-end —
VAD, turn detection, STT, TTS and barge-in, all of which batch or are cheap.
This is the configuration where one 4090 plausibly serves *hundreds* of
concurrent voice users.

The cost is network TTFT, typically 200–500 ms, which lands directly in the
voice-to-voice budget. Mitigations that already exist here: sentence-level
streaming means only the *first* sentence's TTFT is on the critical path, and
speculative STT still hides recognition entirely.

### 6.3 Local, batched, continuous

A server built for continuous batching (vLLM, SGLang, TensorRT-LLM) rather
than llama.cpp. A small MoE — active-parameter count far below total — is the
right shape: high throughput per unit VRAM, and cheap per-token cost at batch.

Two properties of *this* system make it unusually friendly to that:

- **Personas are a shared prefix.** Every user on the same persona shares an
  identical system prompt plus few-shot block. Automatic prefix caching turns
  that into one cached prefix serving all of them, which is exactly what `-np 1`
  was hand-rolling for a single user.
- **Replies are short by construction.** `_SPOKEN_STYLE` asks for "one or two
  short spoken sentences", so generated tokens per turn are small — the regime
  where batched decode shines.

### 6.4 Ordering

1. Move the LLM to something built for concurrency (API or continuous-batching
   server). Nothing else matters until this is done.
2. Continuous micro-batching for TTS (§3), then STT.
3. Only then CPU thread budgeting — which does not currently exist at all:
   `llama-server` runs with no `-t`, Smart Turn requests `os.cpu_count()`, and
   Kokoro takes what it likes, concurrently on a 12-core part. Measured, 1 → 8
   threads is a 5× speedup and past 8 there is nothing left, so the current
   arrangement is pure contention.

Not on the list, because all three were measured and rejected: sub-sentence
chunking, a different inference runtime, and quantisation.

### Batching cannot help inside one turn

Worth stating because it looks like an easy win. Sentences reach the
synthesiser one at a time from `sentence_stream()` as the LLM emits them —
sentence 2 does not exist when sentence 1 begins synthesising. Within a single
reply the pipeline is **LLM-bound for chunk arrival**, not TTS-bound. Batching
only has material to work with once there are multiple independent *streams*.

---

## 7. What already scales

`run_live()` was written with per-call state, so the conversational machinery
is close to per-stream already. All of this is constructed inside the function
and holds no cross-conversation state:

`TurnDetector`, `SmartTurn`, `BargeGate`, `SpeechAdmit`, `SpeakerVerifier`,
`RefBuffer`, `EchoCanceller`, the `speaking` dict, the frame queue, `idx`, the
speculative-STT handle and the enrolment buffer.

The echo model also transfers cleanly. `RefBuffer` plus the coupling estimate
in `SpeechAdmit` express "what we sent" versus "what we hear", which is exactly
the relationship a server has with a remote client — the server knows the audio
it streamed out, so double-talk detection works per session without the client
doing anything special.

---

## 8. What is single-user by construction

Verified against the source.

### 8.1 Module-level singletons

```
WEB              = None              # one dashboard bus
INTERRUPT        = threading.Event() # "the user pressed stop" — which user?
TURN_ACTIVE      = threading.Event() # one global "a turn is in flight"
ENROLL_REQUEST   = threading.Event() # one global "learn a voice"
```

Each assumes exactly one conversation. They become per-session objects passed
in rather than reached for — mechanical, but it touches every call site.

### 8.2 Conversation state

`LLM.history` and `LLM.audio_history` are lists on a single `LLM` instance,
alongside `system`, `shots` and `persona_name`. **One `LLM` object is one
conversation.** Multi-user needs one per session, or a session key threaded
through.

### 8.3 One `Models` instance, one audio device pair

`Models` is constructed once and holds `KPipeline` (Kokoro), the `onnx_asr`
Parakeet session and Silero. Sharing those across threads is not obviously
safe. More fundamentally the audio path is hard-wired to one local device:

```
sd.InputStream(...)    # one microphone
sd.OutputStream(...)   # one speaker
```

A multi-user deployment is not local audio at all — it is streams over the
network — so this layer is *replaced*, not multiplied. `tests/bargein_sim.py`
already demonstrates substituting it: it swaps a virtual full-duplex device
into `sys.modules` and drives the real `run_live()` loop unmodified. That shim
is the template for a transport layer.

### 8.4 Transport

Local `sounddevice` becomes WebRTC or WebSocket audio, which brings its own
work: Opus encode/decode, a jitter buffer, packet loss concealment, and clock
drift between client and server. Note that the 32 ms frame cadence the whole
gate assumes must be preserved across that transport, and a jitter buffer adds
latency that comes straight out of the barge-in budget.

### 8.5 The dashboard

`webui.Bus` keeps one client list and one `_snapshot` replayed to every new
connection. It broadcasts; it does not route. Multi-user needs per-session
topics and an access model.

---

## 9. Model upgrades worth taking

### Parakeet v3 — multilingual, near drop-in

[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
is the same 0.6B TDT FastConformer architecture as the pinned v2, so the
batching economics in §2 carry over, but covers **25 European languages**
(en, es, fr, de, it, pt, nl, pl, ru, uk, sv, da, fi, el, cs, …) instead of
English only. CC-BY-4.0.

It is close to a manifest-only change:

- `istupakov/parakeet-tdt-0.6b-v3-onnx` exists with the **same file layout**
  as the pinned v2 repo (`encoder-model.int8.onnx`,
  `decoder_joint-model.int8.onnx`), from the same publisher.
- `onnx_asr` 0.11.0 — already installed — knows the id
  `nemo-parakeet-tdt-0.6b-v3`.

So the change is `scripts/models.json` (repo id, revision, sizes, SHA256s) plus
the model name passed to `onnx_asr.load_model` in `Models.__init__`. What would
still need checking: int8 accuracy across languages, whether v3's larger
vocabulary changes latency at batch 1, and that Kokoro has a voice for the
target language — **the TTS side is the real multilingual constraint**, since
the curated voice set here is English.

### Others worth a look

- **Speaker identity** — a proper embedding model already ships behind
  `--speaker-lock` (WeSpeaker ResNet34-LM). In a multi-user server this stops
  being optional: it is how you keep one session's audio from being attributed
  to another when several people share a room or a device.
- **Streaming diarization** — `nvidia/diar_streaming_sortformer_4spk` for
  several speakers per session. Its ~1.1 s context rules it out of the 96 ms
  duck decision, but turn confirmation already spends about that long, so it
  belongs at the turn boundary. See [BARGE_IN.md §7](BARGE_IN.md).

---

## 10. Turn-taking with more than one person

Distinct from multi-user, and worth keeping separate: several people talking
to *one* assistant.

`SpeechAdmit` takes a `speaker_ok` hook, and `SpeakerVerifier` already answers
"is this the enrolled speaker?" — enough to **lock onto one person and reject
everyone else**, the useful behaviour in a noisy room. Off by default pending
threshold calibration; see [BARGE_IN.md §7](BARGE_IN.md).

Genuinely addressing several speakers needs more than the hook, because the
hook returns a boolean rather than an identity:

- identity must survive past admission and attach to the captured utterance;
- `handle_turn` and history need per-speaker attribution, where today they are
  one flat `[(user, assistant)]` list;
- a floor policy — with several admitted speakers, "should this interrupt?"
  stops being one threshold and becomes a question of who holds the floor;
- transcript and dashboard need speaker labels.

Measured ceiling for identity, from `tests/bargein_sim.py --oracle-speaker`: a
perfect identifier takes the television scenario from 13 destroyed replies
to 1.

---

## 11. Summary for a future upgrade

- **100 concurrent conversations cost ~25% of one 4090 for speech** and ~20%
  of a 12-core CPU. With the LLM offloaded or batched, 20-100 simultaneous
  realtime voice-to-voice sessions on a single 24 GB device is arithmetically
  comfortable (section 5.1).
- The audio models are **not** the constraint; the LLM is. Fix that first.
- What binds before throughput does: the latency budget, burstiness at turn
  boundaries, LLM KV cache, and the single-process Python frame loop
  (section 5.2).
- Batch with a **latency budget** (10–30 ms micro-batches), never a
  request-response batch endpoint. A ~900 ms fixed overhead would triple a
  550 ms voice-to-voice budget while making throughput look twenty times
  better.
- Keep batch sizes in the **8–32** band for conversation; past that, marginal
  efficiency collapses and only offline work benefits.
- The per-stream gate machinery is already per-call and cheap (~2.6% of a core
  per user). Leave it alone.
- Replace the transport layer, do not multiply it — `tests/bargein_sim.py`
  shows the seam.
- Do not spend time on quantisation, runtime swaps, or sub-sentence chunking.
  All three were measured; none helps.
