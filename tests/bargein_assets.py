"""Generate the audio assets used by the silent barge-in simulation.

User-side speech is synthesized with Kokoro on CPU using a voice distinct from
the assistant's, so the simulated conversation is a genuine two-voice exchange.
Background beds are synthetic. Everything is written to outputs/bargein/ at the
mic sample rate (16 kHz), which is what the VAD and the barge gate consume.

    ./.venv/bin/python tests/bargein_assets.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime import SR, TTS_SR  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "bargein"

# The user's voice — deliberately not the assistant's af_heart.
USER_VOICE = "am_adam"
# A third voice for the "television in the next room" bed, so the babble is
# real speech rather than noise shaped like it. This is the case that most
# often makes open-speaker barge-in feel twitchy.
TV_VOICE = "af_sarah"
# A second PERSON at the same microphone — the case a level or spectrum test
# cannot solve, and the reason speaker verification exists.
INTRUDER_VOICE = "am_michael"

# Full opening turns that start a conversation.
TURNS = {
    "turn_weather": "Hey, can you tell me what the weather is like today?",
    "turn_store": "What time does the store close on Sunday?",
    "turn_reminder": "Can you set a reminder for me tomorrow morning?",
}

# Genuine interruptions — these SHOULD cut the reply off.
BARGES = {
    "barge_wait": "Wait, actually I meant next week, not today.",
    "barge_stop": "No, stop, that is not what I asked you about.",
    "barge_rephrase": "Hold on, let me rephrase that question for you.",
}

# Backchannels — short acknowledgements a listener makes WHILE the assistant
# talks. Cutting the reply off for these is the single most uncomfortable
# failure mode of a naive voice barge-in.
BACKCHANNELS = {
    "back_mhm": "Mm-hmm.",
    "back_yeah": "Yeah.",
    "back_okay": "Okay.",
    "back_right": "Right, sure.",
}

TV_SCRIPT = ("and in financial news the markets closed higher again today "
             "as investors weighed the latest round of earnings reports")

# Spoken by INTRUDER_VOICE: full, well-formed turns that the system would
# happily answer if it could not tell who was talking.
INTRUDER = {
    "other_question": "Hey, what is the capital of France by the way?",
    "other_command": "Actually, forget all that and tell me a joke instead.",
}


def to_mic_rate(audio: np.ndarray, sr: int = TTS_SR) -> np.ndarray:
    """Linear-resample a Kokoro block to the 16 kHz mic domain."""
    if sr == SR:
        return np.asarray(audio, np.float32)
    n = int(round(len(audio) * SR / sr))
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, len(audio), endpoint=False),
                     audio).astype(np.float32)


def synth(models, text: str, voice: str) -> np.ndarray:
    parts = [np.asarray(a, np.float32)
             for _, _, a in models.tts(text, voice=voice, speed=1.0)]
    return to_mic_rate(np.concatenate(parts))


def normalize(audio: np.ndarray, peak: float = 0.5) -> np.ndarray:
    m = float(np.max(np.abs(audio)))
    return (audio * (peak / m)).astype(np.float32) if m > 1e-9 else audio


# --- background beds -------------------------------------------------------

def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise — a much better model of room tone than white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    out = np.fft.irfft(spec * scale, n)
    return normalize(out.astype(np.float32), 1.0)


def fan_hum(n: int, rng: np.random.Generator) -> np.ndarray:
    """Computer/AC hum: a low fundamental plus harmonics under filtered noise."""
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for f, a in ((58.0, 1.0), (116.0, 0.45), (174.0, 0.2), (232.0, 0.1)):
        drift = 1.0 + 0.002 * np.sin(2 * np.pi * 0.07 * t)
        out += (a * np.sin(2 * np.pi * f * drift * t)).astype(np.float32)
    rumble = pink_noise(n, rng) * 0.35
    return normalize(out + rumble, 1.0)


def keyboard_clicks(n: int, rng: np.random.Generator, per_sec: float = 3.5
                    ) -> np.ndarray:
    """Sharp broadband transients — the classic false-trigger for a naive
    energy gate, and a good check that the gate is not click-sensitive."""
    out = np.zeros(n, np.float32)
    count = int(n / SR * per_sec)
    for _ in range(count):
        start = int(rng.uniform(0, max(1, n - 400)))
        length = int(rng.uniform(60, 220))
        env = np.exp(-np.linspace(0, 9, length))
        click = rng.standard_normal(length) * env
        out[start:start + length] += click.astype(np.float32)
    return normalize(out, 1.0)


def reverb(audio: np.ndarray, rng: np.random.Generator, decay: float = 0.35,
           taps: int = 12) -> np.ndarray:
    """Cheap multi-tap reverb, used to push the TV bed 'into the next room'."""
    out = audio.copy()
    for i in range(1, taps):
        delay = int(rng.uniform(0.01, 0.09) * SR) * i
        if delay >= len(audio):
            break
        out[delay:] += (audio[:len(audio) - delay] * (decay ** i)).astype(np.float32)
    return normalize(out, 1.0)


def lowpass(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Zero out everything above the cutoff — muffles the TV through a wall."""
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    spec[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(spec, len(audio)).astype(np.float32)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260727)

    os.environ.setdefault("HF_HOME", str(
        Path(__file__).resolve().parents[1] / "models" / "huggingface"))
    from realtime import Models

    print("Loading Kokoro + Parakeet on CPU...")
    models = Models(tts_device="cpu", stt_device="cpu")

    speech = {**TURNS, **BARGES, **BACKCHANNELS}
    for name, text in speech.items():
        audio = normalize(synth(models, text, USER_VOICE), 0.5)
        sf.write(ASSETS / f"{name}.wav", audio, SR)
        print(f"  {name:16} {len(audio)/SR:5.2f}s  \"{text}\"")

    for name, text in INTRUDER.items():
        audio = normalize(synth(models, text, INTRUDER_VOICE), 0.5)
        sf.write(ASSETS / f"{name}.wav", audio, SR)
        print(f"  {name:16} {len(audio)/SR:5.2f}s  [{INTRUDER_VOICE}] \"{text}\"")

    # 60 s beds, long enough to loop under any scenario.
    n = 60 * SR
    beds = {
        "bed_room": pink_noise(n, rng),
        "bed_fan": fan_hum(n, rng),
        "bed_keyboard": keyboard_clicks(n, rng),
    }
    tv = synth(models, TV_SCRIPT, TV_VOICE)
    tv = np.tile(tv, int(np.ceil(n / len(tv))))[:n]
    beds["bed_tv"] = normalize(reverb(lowpass(tv, 3200.0), rng), 1.0)

    for name, bed in beds.items():
        sf.write(ASSETS / f"{name}.wav", bed, SR)
        print(f"  {name:16} {len(bed)/SR:5.1f}s bed")

    print(f"\nAssets written to {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
