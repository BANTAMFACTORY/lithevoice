"""Validate the torch-free backends against the PyTorch ones they replace.

These tests exist because of a bug that produced no error at all. The first
OnnxSilero fed the model a bare 512-sample frame; the model actually expects
the previous frame's last 64 samples prepended, for 576. Given 512 it returned
a near-zero probability for *everything* — so the runtime came up, the
dashboard worked, and the microphone was simply deaf. Nothing raised.

The lesson is in the shape of the tests: checking that a VAD stays quiet on
silence and noise passes happily against a model that is broken. The only
useful check is real speech, compared frame by frame against the reference.

    ./.venv/bin/python -m unittest tests.test_lite
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000
FRAME = 512


def speech_frames():
    audio, sr = sf.read(str(Path(__file__).with_name("roundtrip_1.wav")),
                        dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        n = int(len(audio) * SR / sr)
        audio = np.interp(np.linspace(0, 1, n, endpoint=False),
                          np.linspace(0, 1, len(audio), endpoint=False),
                          audio).astype(np.float32)
    return [audio[i:i + FRAME]
            for i in range(0, len(audio) - FRAME + 1, FRAME)]


def _require(module: str, why: str):
    """Skip rather than fail when an optional backend is not installed.

    A missing dependency is a fact about the machine, not a defect in the code
    under test, and a red suite on a fresh clone teaches contributors to ignore
    red suites."""
    import importlib

    try:
        return importlib.import_module(module)
    except Exception as exc:  # ImportError, or the ONNX weights being absent
        raise unittest.SkipTest(f"{module} unavailable ({exc}); {why}")


class OnnxVadMatchesTorch(unittest.TestCase):
    def test_agrees_with_reference_on_real_speech(self):
        torch = _require("torch", "install requirements.txt")
        load_silero_vad = _require(
            "silero_vad", "install requirements.txt"
        ).load_silero_vad
        from lite_backends import OnnxSilero

        frames = speech_frames()
        reference = load_silero_vad()
        reference.reset_states()
        ref = np.array([float(reference(torch.from_numpy(f.copy()), SR).item())
                        for f in frames])
        lite = np.array([OnnxSilero().prob(f) for f in [frames[0]]])  # warm
        vad = OnnxSilero()
        lite = np.array([vad.prob(f) for f in frames])

        # The clip must actually contain speech, or this proves nothing.
        self.assertGreater((ref >= 0.5).sum(), 20,
                           "reference VAD found little speech; bad fixture")
        self.assertLess(float(np.abs(ref - lite).max()), 1e-4)
        self.assertEqual(((ref >= 0.5) == (lite >= 0.5)).all(), True)

    def test_detects_speech_at_all(self):
        """The specific regression: a broken wrapper returns ~0 forever."""
        from lite_backends import OnnxSilero
        try:
            vad = OnnxSilero()
        except Exception as exc:
            raise unittest.SkipTest(f"OnnxSilero unavailable ({exc}); run scripts/setup.sh")
        probs = [vad.prob(f) for f in speech_frames()]
        self.assertGreater(max(probs), 0.5,
                           "ONNX VAD never saw speech — check the 64-sample "
                           "context prepended to each frame")


class OnnxKokoroProducesAudio(unittest.TestCase):
    def test_synthesises_comparable_audio(self):
        from lite_backends import OnnxKokoro
        try:
            tts = OnnxKokoro()
        except Exception as exc:  # missing onnxruntime, misaki, or the weights
            raise unittest.SkipTest(f"OnnxKokoro unavailable ({exc}); run scripts/setup.sh")
        chunks = list(tts("Hello there. This is a test of the lite backend.",
                          voice="af_heart", speed=1.0,
                          split_pattern=r"(?<=[.!?])\s+"))
        self.assertEqual(len(chunks), 2, "sentence splitting failed")
        audio = np.concatenate([a for _, _, a in chunks])
        seconds = len(audio) / 24000
        self.assertGreater(seconds, 1.5)
        self.assertLess(seconds, 12.0)
        self.assertGreater(float(np.max(np.abs(audio))), 0.05,
                           "synthesised audio is silent")


if __name__ == "__main__":
    unittest.main()
