# Turn-Taking And Barge-In

How LitheVoice decides that someone is talking to it, and what it does about
it while it is already speaking.

This document covers `SpeechAdmit`, `BargeGate`, `SpeakerVerifier`,
`classify_utterance`, and the turn-boundary resolution in `run_live` — all in
`realtime.py`. The test harness is `tests/bargein_sim.py`, the measurement
tool is `tests/bargein_probe.py`.

---

## 1. The problem

A voice agent that only listens while it is silent is a walkie-talkie. To feel
like a conversation it has to keep listening while it speaks, which means
answering a hard question thirty-one times a second:

> Is this 32 ms of microphone audio *my user talking to me*?

Getting it wrong is expensive in both directions:

| Failure | What the user experiences |
|---|---|
| False positive | The assistant cuts itself off mid-sentence for a cough, a keyboard, a television, or its own voice through the speakers. |
| False negative | The user talks over it and is ignored, then has to repeat themselves. |
| Over-eager | Saying "mm-hmm" destroys the assistant's turn and it starts over. |

The original implementation used one signal: three consecutive Silero VAD
frames scoring ≥ 0.5. That is the obvious design, and measurement showed it
does not work — see §2.

---

## 2. What the measurements actually showed

`tests/bargein_probe.py` builds controlled microphone signals with known
ground truth and reports what the gate sees. Three results shaped everything
that follows.

**Echo reads as speech.** With the reply coming back through open speakers at
−15 dB, **28% of echo-only frames score p ≥ 0.5**. The VAD is not wrong — that
*is* speech, it is just ours. Any gate keyed on VAD confidence alone will make
the assistant interrupt itself, reliably.

**A backchannel is acoustically identical to an interruption.** "Mm-hmm" and
"Wait, that's wrong" both sit at p = 1.00 for essentially every voiced frame,
at the same level above the noise floor. No amount of spectral or energy
cleverness separates them. Only *duration* and, at the turn boundary, *the
words* can.

**A television is not distinguishable by level.** Recorded speech from another
room scores **77% of frames at p = 1.00** and sits *further* above the noise
floor than the real user. Every acoustic signal says "someone is talking",
because someone is. Separating it requires identity, not thresholds. This is
the one case the acoustic gates cannot win; see §7.

A fourth result came from live use rather than the probe, and is the reason
§5.3 exists: **the largest source of destroyed replies was not barge-in at
all.** A fan or a television would open a *turn*; dispatching the answer to
that turn cancelled whatever the assistant was already saying. Fixing only the
barge path left replies dying by another route.

---

## 3. Shape of the solution

One verdict per frame, produced by `SpeechAdmit`, consumed by two clients:

```
mic frame (32 ms)
      │
      ├─ Silero VAD ──► p
      │
      ▼
  SpeechAdmit.update(p, frame, ref_frame) ──► bool
      │                                        (+ .reason for diagnostics)
      ├────────────────► TurnDetector.process(frame, admit)
      │                     failing frames count as silence, so noise
      │                     and echo cannot open a turn
      │
      └────────────────► BargeGate.update(ok, playing)
                            duck → hold → cancel, staged and mostly reversible
```

Sharing the verdict is deliberate. Turn admission and barge-in are the same
question asked at two moments, and answering them differently is what let
noise destroy replies through the back door.

At the end of an utterance a third stage runs, using the transcript that
speculative STT has already produced (§6).

---

## 4. `SpeechAdmit` — is this the user?

Three tests, all a few microseconds per frame. A frame must pass all of them.

### 4.1 Voice activity

`p >= thresh` (default `0.5`, matching `TurnDetector.START_P`). Necessary,
nowhere near sufficient. Everything below exists because this test alone
produced §2's failures.

### 4.2 Above the room

`e >= floor * 10^(snr_db/20)` where `e` is the frame's RMS. Default
`snr_db = 6.0`.

The floor is a **20th percentile of the last ~4 seconds** (`_floor_win`,
125 frames), not an average of quiet frames. That choice is load-bearing:

> An "average of frames where `p < 0.35`" floor never initialises at all when a
> television is talking continuously — there are no quiet frames to learn from,
> the floor stays `None`, and the test is skipped entirely. It disables itself
> exactly when it is needed. A percentile always produces an answer, and steady
> babble simply *becomes* the floor, which is the correct behaviour.

Frames are only fed to the floor while the assistant is **not** playing, so
the floor describes the room rather than slowly absorbing our own echo.

This test is what rejects fans, hum, and steady background noise.

### 4.3 Above our own echo

While playing, `e >= coupling * ref_e * 10^(echo_margin_db/20)`. Default
`echo_margin_db = 4.0`.

`ref_e` is the envelope of the playback reference — the audio we handed to the
speaker — taken as the **maximum over the last 8 frames** (~256 ms), because
the reference leads the echo by the output latency and an envelope comparison
must tolerate a few frames of skew.

`coupling` estimates how much of what we play comes back into the microphone.
It is learned only on frames where the assistant is playing, the reference is
genuinely loud (`ref_e > REF_ACTIVE = 1e-3`), and the frame is not speech
(`p < 0.35`). It is the **90th percentile of a bounded ~3 s window**
(`_ratios`, 100 frames).

> Three versions of this line, and the history is the point.
>
> 1. An **unbounded `max()` ratchet**. Pinned at its cap on a headset (see
>    below) and made the gate demand the user shout over the reference.
> 2. A **90th percentile plus the `REF_ACTIVE` guard**. Fixed that, and was
>    confirmed good in live use on a headset.
> 3. **`max()` over a bounded window**, to recover an open-speaker regression
>    seen in the simulator — five scenarios had gone from 0 destroyed replies
>    to between 1 and 4 (13 → 23 across the suite), and the bounded max
>    brought those five back to 3 instead of 9.
>
> Version 3 was never tried on real hardware, and it **reintroduced version
> 1's symptom**: max lets a single loud frame set the bar, so on a headset the
> user's own speech is rejected as echo and the front of an interruption goes
> missing. Live evidence: a full sentence captured as `0.6s` of "works
> better." It is back to version 2.
>
> The lesson is not about quantiles. It is that a simulated gain in scenarios
> the operator does not use does not justify a regression on the hardware they
> do use. The simulator is good at catching what cannot be heard; it does not
> overrule a confirmed live result. The residual open-speaker cost is a few
> echo-triggered turns in the hardest scenarios, and `--barge-snr-db` and
> `--aec` are the levers there.

> **This was a real bug, and it is worth understanding.** The first version
> learned `coupling = min(4.0, max(coupling * 0.999, e / ref_e))` — a running
> maximum. On a **headset**, where the microphone genuinely hears none of the
> reply, the reference passes through near-zero at the start and end of every
> reply. One frame of `ref_e = 1e-4` against ambient `e = 1e-3` yields a ratio
> of 10, instantly clamped to the 4.0 cap, and `max()` with a 0.999 decay never
> let it fall. The echo test then demanded the user speak roughly 20 dB louder
> than the reference before being believed.
>
> Live symptom: barge-in appeared to work, but the captured utterance was
> missing its beginning — the user was only heard in the gaps *between*
> sentences. The diagnostic trace showed it plainly:
>
> ```
> [gate] echo  p=1.00 lvl=0.0355 ref=0.0584 coup=3.92   ← user's voice, rejected
> [gate] ok    p=1.00 lvl=0.0115 ref=0.0003 coup=3.92   ← heard only in a gap
> ```
>
> A coupling of 4.0 is also physically absurd: it claims the microphone hears
> the speaker 12 dB louder than the signal we sent it. A maximum over a
> *bounded* window can fall as well as rise, and the `REF_ACTIVE` guard stops
> near-silent reference frames from generating meaningless ratios.

`playing` is **sticky** for 12 frames (~380 ms) after the reference last went
quiet. On CPU the synthesiser routinely fails to stay ahead of the speaker, so
the reference goes briefly empty while the room is still ringing with the last
block; treating those frames as "not playing" would file our own echo away as
room noise and skip this test on exactly the frames that need it.

This test is what stops the assistant interrupting itself, and it works with
or without the AEC running in front of it.

### 4.4 Is it *this* speaker?

Optional, via the `speaker_ok` hook. See §7.

---

## 5. `BargeGate` — what to do about it

### 5.1 Staged response

`BargeGate` does not decide whether someone is speaking; it decides what to do
about a *run* of frames that `SpeechAdmit` accepted.

| Stage | Default | Effect | Destructive |
|---|---|---|---|
| `duck` | 96 ms | reply attenuated to `DUCK_GAIN` (0.15, −16 dB) | no |
| `hold` | 352 ms | reply goes silent, keeps its position | no |
| `cancel` | 1400 ms | turn abandoned | **yes** |
| `release` | accumulator hits 0 | un-duck, un-hold, reply continues | no |

The whole design rests on one observation: **perceived responsiveness is set
by the first stage, and only the last one throws anything away.** The assistant
is quiet ~76–140 ms after the user opens their mouth, which already feels like
yielding. That buys more than a second to decide whether they actually meant to
take the floor — and it costs nothing, because the reply cannot become useful
again until they stop talking anyway.

Ducking has a second, useful effect: it drops the echo by the same 16 dB,
which sharpens §4.2 and §4.3 on the very frames where the decision is being
made.

A `grace_ms` window (250 ms) suppresses all decisions until the reply is
actually *audible* — measured from the first written block, not from the
decision to speak, because on CPU those can be two seconds apart. It covers
the echo transient at reply onset and the moment when the coupling estimate is
still settling.

### 5.2 The accumulator leaks; it does not reset

Qualifying frames add 32 ms; non-qualifying frames subtract `32 * decay` ms
(default `decay = 0.5`), floored at zero.

> Requiring N *consecutive* frames looks reasonable and is wrong. "Wait,
> actually I meant next week" has a real pause after the comma. A consecutive
> counter gives up in the middle of a genuine interruption — measured, the
> reply survived a deliberate barge and only died a second later at the turn
> boundary. Leaking tolerates prosodic gaps while still demanding that speech
> *dominate* the window.

### 5.3 Turn admission uses the same verdict

`TurnDetector.process(frame, admit)` takes a bool or a `callable(p) -> bool`.
Frames that fail are treated as silence for turn purposes. The callable form
exists because Silero is a stateful RNN and must be stepped exactly once per
frame — the admission test runs on the probability the detector just computed.

The raw probability is still exposed as `last_p` for the microphone
visualisation, so the UI shows what the room is doing even when the gate is
ignoring it.

---

## 6. The turn boundary — cheap semantics

Acoustics cannot separate "mm-hmm" from "wait, stop" (§2). Words can, and
speculative STT has usually already produced them by the time the turn is
confirmed, so consulting them is nearly free.

When an utterance completes, `run_live` classifies it with
`classify_utterance(text)`:

- **`empty`** — no words recognised. Discarded. Fans, keyboards and door
  clicks reach the VAD but carry no words; answering them is worse than
  ignoring them.
- **`backchannel`** — ≤ 4 words, all in `BACKCHANNEL_WORDS` (`mm-hmm`, `yeah`,
  `okay`, `right`, `sure`, `got it`, …). If a reply is in flight and the gate
  has not committed to a cancel, the duck and hold are released and **the reply
  resumes where it left off**. The user hears it dip and carry on.
- **`speech`** — anything else. Takes the floor.

`FLOOR_TAKING_WORDS` (`stop`, `wait`, `no`, `hold`, `enough`, `actually`, …)
override the backchannel rule: they are as short as an acknowledgement but
unambiguously mean "stop talking".

Direct-audio mode (`--direct-audio`) has no transcript available here and
would pay for one on the critical path, so it keeps the older unconditional
behaviour.

---

## 7. Knowing *whose* voice it is

### 7.1 The hook

`SpeechAdmit` takes `speaker_ok: callable(frame) -> bool`. It is consulted
last, after the cheap tests, and a `False` makes the frame count as silence for
both turn-taking and barge-in.

`run_live(..., speaker_ok=...)` overrides whatever is built in. That parameter
is the seam: it is how a real speaker identifier, or a test oracle, replaces
the default.

### 7.2 What ships: `SpeakerVerifier`

**WeSpeaker ResNet34-LM** (VoxCeleb), 26.5 MB of ONNX, run through the
onnxruntime the project already depends on. It turns ~1 s of audio into a
256-d embedding in about **12 ms on four CPU threads**, and a turn is accepted
only if its cosine against the enrolled centroid clears `threshold`
(default `0.40`).

Preprocessing has to match the model or the embeddings are meaningless:
Kaldi-style 80-bin fbank (25 ms window, 10 ms hop, Hamming, no dither), input
scaled to int16 range, followed by cepstral mean normalisation. `torchaudio`'s
`compliance.kaldi.fbank` does this exactly.

Measured separation, enrolling on two utterances and scoring held-out clips:

| clip | cosine | verdict |
|---|---|---|
| same speaker, full sentence | +0.688 … +0.697 | accept |
| same speaker, short ("right, sure") | +0.404 | accept |
| **different speaker** | +0.103 … +0.200 | reject |
| fan noise | +0.004 | reject |

> **This replaced a spectral match that did not work.** The previous version
> compared L2-normalised log-band energies by cosine. Measured live it rejected
> **4 frames out of 429**: every human voice scores 0.94–1.00 against every
> other one on that representation, so the threshold sat *inside* the enrolled
> speaker's own spread and tightening it would have started rejecting the user
> before it rejected anybody else. Log-compressed band energies describe
> "speech", not "whose speech". It is deleted, not disabled — leaving it in
> place would invite someone to try tuning a threshold that cannot work.

**Enrolment is explicit and required.** Nothing takes a turn until a voice has
been enrolled, which is the only way to be certain the profile is *yours*
rather than whoever happened to speak first. Two utterances are collected, the
centroid is saved to `models/voice_profile.npz`, and later runs load it.

- **Dashboard:** *Learn my voice* → say a sentence, twice.
- **CLI:** `--enroll` re-learns at startup, replacing the saved profile.
- First run with no saved profile enters enrolment automatically and says so.

**Abstention.** Rejecting somebody needs more evidence than embedding them
does. Below `min_verify_s` (0.8 s) of voiced audio the verifier returns `None`
and the identity gate stands down, deferring to the backchannel and
floor-word logic. Without that, a 0.5 s clip scores 0.29 against its *own*
speaker — and the user's own short "stop" would be thrown away as a stranger.

`--speaker-lock` enables the mechanism (off by default, see 7.3);
`--speaker-threshold` moves the operating point; `--enroll` re-learns.

### 7.3 Answering the actual question: one speaker, or several?

**Locking onto one speaker in a noisy room: built, working, and OFF by
default** pending calibration (`--speaker-lock` enables it). Enrol once and
only that voice can take a turn; other people, a television, or a voice on a
call may briefly duck the reply and are then discarded at the turn boundary
with `[other voice] ignored`.

> It is off because the default threshold is not trustworthy yet. Verified in
> simulation it works cleanly — the enrolled speaker is accepted and a second
> person scores +0.15/+0.16 against a +0.40 bar. Tried live it **rejected the
> operator's own voice at +0.32**. Both facts are consistent: real speech
> through a real microphone scores far lower than the clean synthetic TTS the
> threshold was measured on, and 0.40 sits inside the enrolled speaker's own
> spread. The separation is still there (0.32 against 0.15) — the operating
> point is simply in the wrong place, and picking it needs a recording of the
> actual user in the actual room, not a synthetic proxy. Shipping a
> synthetic-calibrated threshold as a default was the mistake.

The deliberate gap is that identity does **not** gate the 96 ms duck, because
an embedding needs about a second of audio and the duck needs a tenth of one.
A stranger can therefore make the reply dip momentarily before being rejected.
That is the safe direction for the error to point: if identity is ever wrong,
the user is still heard. Closing it would mean waiting ~1 s before yielding to
*anyone*, including the enrolled speaker, which would destroy the thing that
makes barge-in feel good.

**Addressing several speakers at once: not supported today, and the hook alone
is not enough.** `speaker_ok` returns a *boolean*, not an identity. It can
answer "should this frame be admitted?" but not "who is this?". Multi-party
operation would additionally need:

- identity to survive past admission — a label attached to the captured
  utterance, not discarded at the gate;
- turn attribution — `handle_turn` and the conversation history are currently
  a single `[(user, assistant)]` list with no notion of who spoke;
- a floor policy — with several admitted speakers, "should this interrupt?"
  stops being one threshold and becomes a question about who currently holds
  the floor;
- the web UI and transcript to carry speaker tags.

None of that plumbing exists. The gate is the right *place* for it and does not
stand in the way, but calling the system multi-speaker-ready would overstate
it.

**On `Parakeet_Multitalk` / Sortformer specifically.** A streaming diarizer
such as `nvidia/diar_streaming_sortformer_4spk` is the principled answer to
identity, and its natural home is the turn boundary rather than the duck. Its
`att_context_size [70, 13]` implies roughly **1.1 s of context**, which is far
too slow to gate a 96 ms duck — but the duck is reversible and the turn
boundary is not, and turn confirmation already happens at end-of-utterance
where a second of latency is spent regardless. Putting identity there fixes
the layer that was doing the real damage (§2, fourth result) without touching
the fast path.

Measured ceiling, from `tests/bargein_sim.py --oracle-speaker`: a perfect
speaker identifier takes the television scenario from **13 destroyed replies
to 1**.

---

## 8. Playback mechanics

`speak_stream` writes 100 ms blocks and re-checks the flags between each, so
the reply reacts within one block.

- **duck** multiplies the block by `DUCK_GAIN`, **ramped** across the block
  with `np.linspace` rather than stepped. A gain change applied as a step
  part-way through a waveform is a discontinuity, and a discontinuity is an
  audible click.
- **hold** spins in 20 ms sleeps without consuming audio, so the reply keeps
  its place and can resume verbatim.
- **cancel** breaks out; only text actually spoken is committed to history.
- the reference is pushed to `RefBuffer` **before** `play_stream.write()`.
  `write()` blocks until the device has room, so pushing afterwards makes the
  reference *lag* the echo it is supposed to predict — which quietly disables
  both the AEC and §4.3.

The output stream uses `latency = OUTPUT_LATENCY_S` (80 ms). `latency="low"`
asks the backend for the smallest buffer it will grant, which on PipeWire is
small enough that the GIL — shared with Silero every 32 ms and with Kokoro
synthesis — starves the device: measured **36 dropouts of 2–60 ms in 13 s of
speech**, heard as a crackle. Buffering is additive with the duck latency, so
the value is deliberately modest rather than generous.

---

## 9. Tuning

| Flag | Default | Raise it when |
|---|---|---|
| `--barge-duck-ms` | 96 | noise makes it flinch; lower for a snappier yield |
| `--barge-cancel-ms` | 1400 | short acknowledgements still cost you a turn |
| `--barge-snr-db` | 6 | the room is loud and it reacts to nothing in particular |
| `--speaker-lock` | off | you want only your voice answered — calibrate the threshold first (§7.3) |
| `--speaker-threshold` | 0.40 | lower it if you are rejected; raise it if others get through |
| `--enroll` | — | re-learn your voice, replacing the saved profile |

Constants worth knowing: `DUCK_GAIN` (0.15), `REF_ACTIVE` (1e-3),
`OUTPUT_LATENCY_S` (0.08), `BargeGate(hold_ms=352, decay=0.5, grace_ms=250)`,
`SpeechAdmit(echo_margin_db=4.0)`.

`LITHEVOICE_BARGE_DEBUG=1` logs one line per 8 frames while a reply is
audible, naming the test that rejected the frame:

```
[gate] echo    p=1.00 lvl=0.0355 floor=0.0016 ref=0.0584 coup=3.92 acc=0ms spk=0.995
[gate] ok      p=1.00 lvl=0.0115 floor=0.0016 ref=0.0003 coup=3.92 acc=32ms spk=0.994
```

`reason` is one of `vad`, `floor`, `echo`, `speaker`, `ok` — which is how the
coupling bug in §4.3 was found. Stage transitions log as `[barge] ducking`,
`[barge] holding`, `[barge] released`.

---

## 10. Testing

`tests/bargein_sim.py` drives the **real** `run_live()` loop through a virtual
full-duplex device. Kokoro's audio is played into a modelled room — multi-tap
reflections, 60 ms output latency, speaker distortion — and returns to the
microphone as echo, mixed with a background bed and user speech scheduled
relative to the reply. Silero, `TurnDetector`, Smart Turn, the AEC and Parakeet
are all real. Only the sound card is fake, so it is silent and repeatable.

```bash
./.venv/bin/python tests/bargein_assets.py            # generate voices + beds
./.venv/bin/python tests/bargein_sim.py --list
./.venv/bin/python tests/bargein_sim.py               # the suite
./.venv/bin/python tests/bargein_sim.py --legacy          # original gate
./.venv/bin/python tests/bargein_sim.py --oracle-speaker  # identity ceiling
./.venv/bin/python tests/bargein_probe.py             # what the gate sees
```

Ten scenarios: headset, open speakers at two levels with and without AEC,
backchannels (both), keyboard, fan, television, and double-talk over loud
speakers.

### Scoring, and a mistake worth repeating

The suite scores **how a reply died, by whichever path** — not whether the
gate fired. A gate that never triggers but lets `finish_turn` kill the reply a
second later has not solved anything.

That metric was itself the source of a missed bug. "Real interruptions
honoured" counted the reply being destroyed by *any* route, so a barge that
failed at the gate and only died later at the turn boundary scored as success.
The headset coupling bug in §4.3 passed the whole suite and was caught in live
use. Counting outcomes is right; counting them so coarsely that two very
different mechanisms look identical is not.

### Measured results

Identical scenarios and metric, tuned build vs. the original gate:

| | original gate | tuned | perfect identity |
|---|---|---|---|
| replies destroyed by noise/echo/TV | 59 | 13–15 (nearly all the TV) | 1 |
| replies destroyed by a backchannel | 2 | 0–1 | 1 |
| real interruptions honoured | 5/5 | 5/5 | 5/5 |
| backchannels kept off the floor | 0 | 3–6 | 3 |
| time to go quiet | n/a (no duck) | 76–175 ms | 126 ms |

Clean scenarios went 1/10 → 8/10. The suite is real-time and CPU-bound, so
figures vary between runs; these come from a small number of runs, not a large
sample.

---

## 11. Measured against other stacks

Two production voice frameworks were reimplemented as alternative policies and
run through the identical ten scenarios, with the identical `SpeechAdmit`
front end. Only the policy differs, so the comparison isolates turn-taking
behaviour. Defaults were read from their sources, not from memory:

* **livekit** — `livekit-agents`, `voice/turn.py` `_INTERRUPTION_DEFAULTS`:
  `min_duration` 0.5 s, `resume_false_interruption` True,
  `false_interruption_timeout` 2.0 s, `backchannel_boundary` (1.0, 1.0).
* **pipecat** — `audio/vad/vad_analyzer.py`: `VAD_CONFIDENCE` 0.7,
  `VAD_START_SECS` 0.2, `VAD_STOP_SECS` 0.2.

Both are in `tests/policies.py`; select with
`tests/bargein_sim.py --policy {ours,ours-v2,livekit,pipecat}`.

| | ours | ours-v2 | livekit | pipecat |
|---|---|---|---|---|
| replies destroyed by noise/echo/TV | 19 | **18** | 26 | 19 |
| **backchannels destroyed** | 1 | 1 | 1 | **4** |
| **real interruptions honoured** | 6/5 | 6/5 | **3/5** | 5/5 |
| backchannels kept off the floor | 7 | **8** | 7 | **1** |
| ducks while nobody spoke | 3 | 2 | 0 | 0 |
| **time to go quiet** | **65 ms** | 67 ms | *no duck stage* | *no duck stage* |
| time to abandon the turn | 1597 ms | 1267 ms | — | 200 ms |

**The staged duck is the structural difference.** Neither of the other
policies has a reversible first step: LiveKit goes straight to silent at
0.5 s, Pipecat straight to cancelled at 0.2 s. Ours is the only one that goes
quiet in under 100 ms, and it can do so precisely *because* that step is
reversible — being wrong is cheap, so it can afford to be fast.

The two failure modes that follows from having no duck are visible in the
table. Pipecat's 0.2 s trigger is quick but irreversible, so it destroyed
**four backchannels** and kept only one off the floor: saying "mm-hmm" costs
the turn. LiveKit's 0.5 s bar avoids that but **misses 3 of 5 real
interruptions** — an interruption that does not sustain past half a second
never registers at all — and it destroyed the most replies overall, because
when it does fire it commits to silence for a 2 s window.

`ours-v2` (confidence-weighted accumulation plus tail softening, in
`tests/policies.py`) is marginally ahead of `ours` — fewest destroyed, most
backchannels kept, and it abandons real interruptions 330 ms sooner. It is not
the default, because the gap is inside the noise band described below.

**What this comparison does and does not show.** It compares *policies*, not
products. LiveKit ships an ML "adaptive" backchannel classifier that was not
reproduced here, and that is their answer to the same problem our word list
solves; reproducing it would likely improve their row. Both frameworks also
bring production WebRTC transport, multi-user support and vendor integrations
that this project does not have at all. The claim supported by this table is
narrow: **for a single local speaker, this policy yields faster and destroys
fewer turns than the defaults of either framework.**

**Statistical honesty.** These are single runs. Repeats of an identical build
varied 13 → 15 → 19 on "replies destroyed", so treat ±3 as noise. The large
gaps — the duck column, LiveKit's 3/5, Pipecat's 4 backchannels — are well
outside that; the ours-vs-ours-v2 difference is not. The "6/5" figure is also
a counting artifact (it should not exceed the denominator) and should not be
leaned on.

## 12. Known limitations

- **The television is unsolved acoustically.** §2 and §7.3. It needs identity.
- **Speaker lock is off by default**, because its threshold is calibrated
  on synthetic audio and rejected a real user at +0.32. §7.2, §7.3.
- **Single speaker only.** §7.3.
- **Synthesis is not interruptible mid-sentence.** `cancel` stops playback
  within one 100 ms block, but Kokoro finishes the sentence it had already
  started. Nothing is heard; it costs CPU, not perceived latency. On CPU that
  can be seconds.
- **The AEC remains experimental** on real microphone arrays whose driver DSP
  makes the captured path non-linear. §4.3 is designed to work without it, and
  `--barge-key` remains the guaranteed-safe open-speaker mode.
