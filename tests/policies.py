"""Interchangeable barge-in policies, for A/B measurement.

Every policy presents the interface `BargeGate` presents, so `run_live()` can
be driven with any of them and scored by `tests/bargein_sim.py` on identical
scenarios:

    reset()                     new reply
    relax()                     interruption resolved as not taking the floor
    committed -> bool           has the reply been abandoned
    update(ok, playing) -> str  "duck" | "hold" | "cancel" | "release" | None

`ok` is one frame's SpeechAdmit verdict, so all policies see exactly the same
acoustic front end. What differs is only what they *do* about a run of
qualifying frames, which is the thing worth comparing.

Defaults for the third-party arms are taken from their sources, not from
memory:

  livekit  livekit-agents, voice/turn.py _INTERRUPTION_DEFAULTS:
           min_duration 0.5 s, resume_false_interruption True,
           false_interruption_timeout 2.0 s, backchannel_boundary (1.0, 1.0)
  pipecat  pipecat audio/vad/vad_analyzer.py:
           VAD_CONFIDENCE 0.7, VAD_START_SECS 0.2, VAD_STOP_SECS 0.2
"""

from __future__ import annotations

FRAME_MS = 32


class PipecatLike:
    """Interrupt as soon as VAD has been confident for `start_secs`.

    The simplest thing that works, and the baseline most voice agents ship:
    one threshold, one irreversible action. No ducking, no resume. Included to
    show what the staged designs are actually buying.
    """

    name = "pipecat"

    def __init__(self, start_secs=0.2):
        self.start_ms = start_secs * 1000
        self.reset()

    def reset(self):
        self.run_ms = 0.0
        self.stage = 0

    def relax(self):
        self.reset()

    @property
    def committed(self):
        return self.stage >= 3

    def update(self, ok, playing=False):
        if not ok:
            self.run_ms = 0.0
            return None
        self.run_ms += FRAME_MS
        if self.stage < 3 and self.run_ms >= self.start_ms:
            self.stage = 3
            return "cancel"
        return None


class LiveKitLike:
    """Stop at `min_duration`, and resume if the interruption proves false.

    LiveKit reaches the same conclusion this project did — an interruption
    should be reversible — but expresses it as one binary stop with a generous
    2 s window to decide it was spurious, rather than a graded response. It
    also suppresses backchannel-classified speech near the boundaries of the
    agent's turn, which is the idea `OursV2` borrows below.
    """

    name = "livekit"

    def __init__(self, min_duration=0.5, false_interruption_timeout=2.0,
                 backchannel_boundary=(1.0, 1.0)):
        self.min_ms = min_duration * 1000
        self.false_ms = false_interruption_timeout * 1000
        self.head_ms, self.tail_ms = (b * 1000 for b in backchannel_boundary)
        self.reset()

    def reset(self):
        self.run_ms = 0.0
        self.quiet_ms = 0.0
        self.played_ms = 0.0
        self.stage = 0

    def relax(self):
        self.run_ms = 0.0
        self.quiet_ms = 0.0
        self.stage = 0

    @property
    def committed(self):
        return self.stage >= 3

    def update(self, ok, playing=False):
        if playing:
            self.played_ms += FRAME_MS
        if ok:
            self.quiet_ms = 0.0
            self.run_ms += FRAME_MS
        else:
            self.run_ms = 0.0
            if self.stage:
                self.quiet_ms += FRAME_MS
                if self.quiet_ms >= self.false_ms:
                    # Decided it was a false interruption: resume.
                    self.stage = 0
                    return "release"
            return None
        # Suppress interruptions in the opening window of the agent turn.
        if self.played_ms < self.head_ms:
            return None
        if self.stage < 2 and self.run_ms >= self.min_ms:
            self.stage = 2          # binary: straight to silent
            return "hold"
        return None


class OursV2:
    """LitheVoice's staged gate, with two ideas the comparison suggested.

    The base design stays: duck early because that is what the person hears as
    yielding, hold next, and only abandon the turn once. Two changes:

    1. **Confidence-weighted accumulation.** Frames no longer count equally.
       A run of unambiguous speech accrues faster than a marginal one, so a
       decisive interruption commits sooner while a hesitant one gets more
       time to prove itself. The old gate had one speed for both.

    2. **Tail softening**, adapted from LiveKit's `backchannel_boundary`.
       Near the end of a reply the assistant is about to stop anyway, so
       destroying the turn buys nothing and costs the tail of a sentence. In
       the last `tail_ms` the bar to *cancel* rises; ducking is untouched, so
       it still feels instantly responsive.

    Ducking and holding stay cheap and reversible; only the irreversible step
    is made more conservative, which is the same principle as the original
    staging, applied along a second axis.
    """

    name = "ours-v2"

    def __init__(self, duck_ms=96, hold_ms=352, cancel_ms=1400, decay=0.5,
                 grace_ms=250, tail_ms=900, tail_factor=1.6, fast_gain=1.5):
        self.duck_ms = duck_ms
        self.hold_ms = hold_ms
        self.cancel_ms = cancel_ms
        self.decay = decay
        self.grace_ms = grace_ms
        self.tail_ms = tail_ms
        self.tail_factor = tail_factor
        self.fast_gain = fast_gain
        self.reply_ms = None      # total audible length, when known
        self.reset()

    def reset(self):
        self.acc_ms = 0.0
        self.stage = 0
        self.played_ms = 0.0
        self.streak = 0

    def relax(self):
        self.acc_ms = 0.0
        self.stage = 0
        self.streak = 0

    @property
    def committed(self):
        return self.stage >= 3

    def update(self, ok, playing=False):
        if playing:
            self.played_ms += FRAME_MS

        if ok:
            self.streak += 1
            # Unambiguous speech accrues faster, up to fast_gain. Ramps in
            # over ~10 frames so a click cannot reach the bonus.
            gain = 1.0 + (self.fast_gain - 1.0) * min(1.0, self.streak / 10.0)
            self.acc_ms += FRAME_MS * gain
        else:
            self.streak = 0
            self.acc_ms = max(0.0, self.acc_ms - FRAME_MS * self.decay)
            if self.acc_ms == 0.0 and self.stage in (1, 2):
                self.stage = 0
                return "release"
            return None

        if self.played_ms < self.grace_ms:
            return None

        cancel_at = self.cancel_ms
        if self.reply_ms is not None and \
                self.reply_ms - self.played_ms < self.tail_ms:
            cancel_at *= self.tail_factor

        if self.stage < 3 and self.acc_ms >= cancel_at:
            self.stage = 3
            return "cancel"
        if self.stage < 2 and self.acc_ms >= self.hold_ms:
            self.stage = 2
            return "hold"
        if self.stage < 1 and self.acc_ms >= self.duck_ms:
            self.stage = 1
            return "duck"
        return None


POLICIES = {
    "pipecat": PipecatLike,
    "livekit": LiveKitLike,
    "ours-v2": OursV2,
}
