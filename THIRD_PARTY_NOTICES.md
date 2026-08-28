# Third-Party Notices

LitheVoice downloads third-party models and native binaries during setup. They
are not committed to this repository and remain governed by their upstream
licenses and terms. Exact sources and revisions are listed in
`scripts/models.json` and the README.

## Downloaded artifacts

LitheVoice ships **no** model weights. `scripts/download_models.py` fetches these
pinned, hash-verified artifacts from their upstream sources at setup time. Each
remains governed by its own licence; the pins below are the exact revisions this
release was built and measured against (generated from `scripts/models.json`,
and checked against it by `tests/test_release.py`).

| Artifact | Source | Pinned revision | Licence |
| --- | --- | --- | --- |
| LLM (Gemma 4 E2B, GGUF) | [`ggml-org/gemma-4-E2B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) | `b4243c156154` | apache-2.0 (stock Gemma 4 E2B instruction-tuned; llama.cpp-project conversion) |
| Speech recognition (Parakeet TDT) | [`istupakov/parakeet-tdt-0.6b-v2-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx) | `0bbb45a33658` | cc-by-4.0 |
| Speech synthesis (Kokoro-82M) | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) | `f3ff3571791e` | apache-2.0 |
| Speaker embedding (WeSpeaker) | [`onnx-community/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM) | `6a61a1833ff2` | apache-2.0 |
| Turn detection (Smart Turn v3) | [`pipecat-ai/smart-turn-v3`](https://huggingface.co/pipecat-ai/smart-turn-v3) | `f766f81d3cfd` | bsd-2-clause |
| Inference runtime (llama.cpp prebuilt) | github.com/ggml-org/llama.cpp (release binary) | pinned release tag | mit |
| Speech synthesis, ONNX (Kokoro-82M) | [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) | `1939ad2a8e41` | apache-2.0 |

The Parakeet ONNX conversion is distributed under **CC BY 4.0**, which requires
attribution: credit `istupakov/parakeet-tdt-0.6b-v2-onnx` (a conversion of
NVIDIA's Parakeet TDT 0.6B v2) in anything you build on this pipeline that
carries its transcripts.

## Vendored Smart Turn Feature Extraction

`whisper_features.py` contains NumPy feature-extraction code vendored from the
Smart Turn project and retains this notice:

> Copyright (c) 2024-2026, Daily

> SPDX-License-Identifier: BSD 2-Clause License

BSD 2-Clause License:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The implementation also mirrors behavior from Hugging Face Transformers'
`WhisperFeatureExtractor` and `audio_utils`, distributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
