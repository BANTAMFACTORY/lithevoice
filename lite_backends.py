"""Torch-free backends for a small deployable build.

The default stack pulls PyTorch in for exactly two things — Kokoro synthesis
and the Silero VAD wrapper — and pays a great deal for it. Measured on this
machine, the virtual environment is 6.0 GB, of which 4.7 GB is `torch`,
`nvidia/*` CUDA libraries and `triton`, to run one 82M-parameter model. The
pinned weights are 5.1 GB, so the runtime costs roughly as much as the models.

Everything else is already native or ONNX: Parakeet through `onnx_asr`, Smart
Turn through onnxruntime with vendored NumPy features, Gemma through a
llama.cpp binary. Replacing these two removes the last PyTorch dependency and
takes the environment to roughly 600 MB, which is the difference between "runs
on a workstation" and "ships to a laptop, a container or a phone".

Nothing here is used unless asked for: `--tts-backend onnx` selects the
synthesiser and `--lite` selects both. The default path is untouched.

Two things this deliberately does NOT do:

* It does not replace `torchaudio.compliance.kaldi.fbank`, which
  `SpeakerVerifier` uses. Reproducing Kaldi's filterbank in NumPy exactly is
  fiddly and a silent mismatch would corrupt embeddings rather than fail
  loudly, so speaker lock is simply unavailable under `--lite`. It is off by
  default anyway.
* It does not claim to be faster. Measured on x86-64 it is a wash against
  PyTorch at 4-8 threads and slower at 12 (`tests/bench_tts.py`). The reason
  to use it is size, not speed. On aarch64 the comparison is unmeasured and
  could go either way.
"""

from __future__ import annotations

import os
import re

import numpy as np

SR = 16000
TTS_SR = 24000

# Pinned alongside everything else in scripts/models.json.
KOKORO_ONNX_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_ONNX_REVISION = "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
KOKORO_ONNX_FILE = "onnx/model.onnx"


def _session(path, threads=None):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads or min(8, os.cpu_count() or 4)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


class OnnxSilero:
    """Silero VAD straight from the .onnx the `silero-vad` package ships.

    The package itself cannot be used torch-free — `silero_vad/model.py` opens
    with an unconditional `import torch`, even on the ONNX path — but the model
    file beside it is self-contained: audio in, speech probability and a
    recurrent state out. `TurnDetector` calls `.prob()` when the object has it,
    so this drops in without changing the detector's logic.
    """

    STATE = (2, 1, 128)
    # The model is fed the previous frame's last 64 samples followed by the
    # current 512, i.e. 576 samples. Feeding it a bare 512 does not fail — it
    # returns a near-zero probability for everything, so speech is simply never
    # detected. Missing this is why the first version of this class was silent.
    CONTEXT = 64

    def __init__(self, threads=1, path=None):
        # find_spec() locates the package WITHOUT executing it. Importing
        # silero_vad would defeat the whole exercise: its model.py opens with
        # an unconditional `import torch`, so on a genuinely torch-free
        # install the import fails even though the .onnx beside it is fine.
        #
        # A fully torch-free install does not have the package at all — pip
        # would drag torch in as its dependency — so `path` (or
        # LITHEVOICE_SILERO_ONNX) supplies the model file directly.
        path = path or os.environ.get("LITHEVOICE_SILERO_ONNX")
        if not path:
            import importlib.util
            spec = importlib.util.find_spec("silero_vad")
            if spec is None or not spec.origin:
                raise RuntimeError(
                    "no Silero model: install silero-vad, or set "
                    "LITHEVOICE_SILERO_ONNX to a silero_vad.onnx file")
            path = os.path.join(os.path.dirname(spec.origin),
                                "data", "silero_vad.onnx")
        if not os.path.isfile(path):
            raise RuntimeError(f"silero_vad.onnx not found at {path}")
        # One thread: the model is tiny and per-frame, so thread hand-off
        # costs more than it saves.
        self._sess = _session(path, threads=threads)
        self.reset_states()

    def reset_states(self):
        self._state = np.zeros(self.STATE, np.float32)
        self._context = np.zeros((1, self.CONTEXT), np.float32)

    def prob(self, frame):
        """Speech probability for one 512-sample frame at 16 kHz."""
        frame = np.asarray(frame, np.float32).reshape(1, -1)
        x = np.concatenate([self._context, frame], axis=1)
        out, self._state = self._sess.run(
            None, {"input": x, "state": self._state,
                   "sr": np.array(SR, dtype=np.int64)})
        self._context = x[:, -self.CONTEXT:]
        return float(out[0][0])


class OnnxKokoro:
    """Kokoro synthesis through onnxruntime, with misaki for grapheme-to-phoneme.

    Presents the same call signature as `kokoro.KPipeline` — it yields
    `(graphemes, phonemes, audio)` per sentence — so `Models.speak_stream` does
    not know the difference.

    misaki reaches PyTorch only through spaCy/thinc, which import it inside a
    `try/except ImportError` and set `has_torch = False` when it is absent, so
    the frontend works in an environment with no torch at all.
    """

    def __init__(self, threads=None):
        from huggingface_hub import hf_hub_download
        from misaki import en

        self._sess = _session(
            hf_hub_download(KOKORO_ONNX_REPO, KOKORO_ONNX_FILE,
                            revision=KOKORO_ONNX_REVISION),
            threads=threads)
        self._g2p = en.G2P(trf=False, british=False)
        self._vocab = self._load_vocab()
        self._voices = {}

    def _load_vocab(self):
        import json
        from huggingface_hub import hf_hub_download
        cfg = hf_hub_download(KOKORO_ONNX_REPO, "config.json",
                              revision=KOKORO_ONNX_REVISION)
        with open(cfg, encoding="utf-8") as handle:
            data = json.load(handle)
        vocab = data.get("vocab")
        if vocab:
            return vocab
        tok = hf_hub_download(KOKORO_ONNX_REPO, "tokenizer.json",
                              revision=KOKORO_ONNX_REVISION)
        with open(tok, encoding="utf-8") as handle:
            return json.load(handle)["model"]["vocab"]

    def _voice(self, name):
        """Style vectors as raw float32, so no torch is needed to read them."""
        if name not in self._voices:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(KOKORO_ONNX_REPO, f"voices/{name}.bin",
                                   revision=KOKORO_ONNX_REVISION)
            self._voices[name] = np.fromfile(
                path, dtype=np.float32).reshape(-1, 1, 256)
        return self._voices[name]

    def __call__(self, text, voice="af_heart", speed=1.0, split_pattern=None):
        parts = (re.split(split_pattern, text) if split_pattern else [text])
        pack = self._voice(voice)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            phonemes, _ = self._g2p(part)
            ids = [self._vocab.get(p) for p in phonemes]
            ids = [i for i in ids if i is not None]
            if not ids:
                continue
            # Kokoro brackets the sequence with the padding token, and indexes
            # the style pack by phoneme count.
            input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
            style = pack[min(len(phonemes) - 1, len(pack) - 1)]
            audio = self._sess.run(None, {
                "input_ids": input_ids,
                "style": style.astype(np.float32).reshape(1, -1),
                "speed": np.array([speed], dtype=np.float32),
            })[0]
            yield part, phonemes, np.asarray(audio, np.float32).reshape(-1)
