"""Check whether a LitheVoice installation is complete and usable."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("models.json")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-llm", action="store_true",
                        help="do not check for the bundled Gemma LLM")
    parser.add_argument("--full", action="store_true", help="also hash all multi-GB model files")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    print(f"Python: {platform.python_version()} ({platform.architecture()[0]})")
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        failures.append("Python 3.12 x64 is required")

    modules = [
        ("numpy", "numpy"),
        ("soundfile", "soundfile"),
        ("sounddevice", "sounddevice"),
        ("silero_vad", "silero-vad"),
        ("onnx_asr", "onnx-asr"),
        ("onnxruntime", "onnxruntime"),
        ("kokoro", "kokoro"),
        ("huggingface_hub", "huggingface-hub"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
    ]
    loaded = {}
    for module_name, distribution in modules:
        try:
            loaded[module_name] = importlib.import_module(module_name)
            version = package_version(distribution)
            if module_name == "onnxruntime" and version == "missing":
                version = package_version("onnxruntime-gpu")
            print(f"{module_name}: {version}")
        except Exception as exc:
            failures.append(f"cannot import {module_name}: {exc}")

    torch = loaded.get("torch")
    torchaudio = loaded.get("torchaudio")
    if torch is not None and torchaudio is not None:
        torch_base = str(torch.__version__).split("+")[0]
        audio_base = str(torchaudio.__version__).split("+")[0]
        if torch_base != audio_base:
            failures.append(f"torch {torch.__version__} and torchaudio {torchaudio.__version__} do not match")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    ort = loaded.get("onnxruntime")
    if ort is not None:
        providers = ort.get_available_providers()
        print(f"ONNX providers: {', '.join(providers)}")
        if "CPUExecutionProvider" not in providers:
            failures.append("ONNX Runtime has no CPUExecutionProvider")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not args.skip_models:
        model_dir = Path(os.environ.get("LITHEVOICE_MODELS_DIR", PROJECT_ROOT / "models"))
        llm_dir = model_dir / "gemma_4_e2b"
        for entry in ([] if args.skip_llm else manifest["llm"]["files"]):
            path = llm_dir / entry["filename"]
            if not path.is_file():
                failures.append(f"missing model: {path}")
                continue
            if path.stat().st_size != entry["size"]:
                failures.append(f"wrong model size: {path}")
            elif args.full and digest(path).lower() != entry["sha256"].lower():
                failures.append(f"wrong model SHA256: {path}")
        parakeet = model_dir / "parakeet-int8" / ".complete"
        if not parakeet.is_file():
            failures.append(f"missing pinned Parakeet model: {parakeet.parent}")
        else:
            print(f"Models: {model_dir}")

    if not args.skip_llama:
        llama_dir = Path(os.environ.get("LITHEVOICE_LLAMA_DIR", PROJECT_ROOT / "llama.cpp"))
        server = llama_dir / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
        marker = llama_dir / "backend.json"
        if not server.is_file():
            failures.append(f"missing llama-server: {server}")
        if not marker.is_file():
            failures.append(f"missing llama.cpp backend marker: {marker}")
        else:
            backend = json.loads(marker.read_text(encoding="utf-8")).get("backend")
            print(f"llama.cpp backend: {backend}")
            if backend == "cuda" and torch is not None and not torch.cuda.is_available():
                failures.append("CUDA llama.cpp was installed, but PyTorch cannot use CUDA")

    sounddevice = loaded.get("sounddevice")
    if sounddevice is not None:
        try:
            devices = sounddevice.query_devices()
            inputs = sum(int(device["max_input_channels"]) > 0 for device in devices)
            outputs = sum(int(device["max_output_channels"]) > 0 for device in devices)
            print(f"Audio devices: {inputs} input, {outputs} output")
            if not inputs or not outputs:
                warnings.append("no usable default input/output audio devices were found")
        except Exception as exc:
            warnings.append(f"could not enumerate audio devices: {exc}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("LitheVoice doctor: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
