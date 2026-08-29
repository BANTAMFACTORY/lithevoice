#!/usr/bin/env bash
# Hardware-aware Linux installation. Mirrors scripts/setup.ps1: it creates
# .venv, installs matching PyTorch/torchaudio and ONNX Runtime builds,
# downloads the pinned model revisions, installs llama.cpp, and runs the
# installation doctor. Resumable and idempotent.
set -euo pipefail

LLAMA_BACKEND=auto
CPU_ONLY=0
INCLUDE_GPU_STT=0
SKIP_MODELS=0
SKIP_LLAMA=0
SKIP_LLM=0
FORCE_DOWNLOADS=0

usage() {
    cat <<'EOF'
Usage: scripts/setup.sh [options]

  --llama-backend BACKEND   auto (default), cpu, or vulkan
  --cpu-only                force CPU PyTorch/ONNX Runtime and CPU llama.cpp
  --include-gpu-stt         also fetch the optional fp32 Parakeet (~2.5 GB)
  --skip-models             do not download model weights
  --skip-llama              do not download the llama.cpp runtime
  --skip-llm                do not download the bundled Gemma LLM
                            (voice pipeline only; bring your own brain)
  --force-downloads         re-download and re-verify everything
  -h, --help                show this message

llama.cpp publishes no prebuilt Linux CUDA binary, so `auto` resolves the
llama.cpp backend to cpu even on NVIDIA. Kokoro and Parakeet still use the
GPU through PyTorch/ONNX Runtime when CUDA validates. For GPU llama.cpp,
either pass --llama-backend vulkan or build llama.cpp with -DGGML_CUDA=ON and
point LITHEVOICE_LLAMA_DIR at that build.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --llama-backend) LLAMA_BACKEND="${2:?--llama-backend needs a value}"; shift 2 ;;
        --cpu-only)        CPU_ONLY=1; shift ;;
        --include-gpu-stt) INCLUDE_GPU_STT=1; shift ;;
        --skip-models)     SKIP_MODELS=1; shift ;;
        --skip-llama)      SKIP_LLAMA=1; shift ;;
        --skip-llm)        SKIP_LLM=1; shift ;;
        --force-downloads) FORCE_DOWNLOADS=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$LLAMA_BACKEND" in
    auto|cpu|vulkan) ;;
    cuda)
        echo "llama.cpp has no prebuilt Linux CUDA binary; see --help." >&2
        exit 2 ;;
    *) echo "invalid --llama-backend: $LLAMA_BACKEND" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

export PYTHONUTF8=1
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

cd "$PROJECT_ROOT"

# --- interpreter -----------------------------------------------------------
BASE_PYTHON=""
for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
        BASE_PYTHON="$candidate"
        break
    fi
done
if [ -z "$BASE_PYTHON" ]; then
    echo "Python 3.12 is required. Install it (for example: sudo apt install python3.12 python3.12-venv) and rerun setup." >&2
    exit 1
fi

if [ "$SKIP_MODELS" -eq 0 ]; then
    FREE_KB="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')"
    if [ "$FREE_KB" -lt $((12 * 1024 * 1024)) ]; then
        echo "At least 12 GB free is required for models and installation; only $((FREE_KB / 1024 / 1024)) GB is available." >&2
        exit 1
    fi
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Creating .venv with $("$BASE_PYTHON" --version)..."
    "$BASE_PYTHON" -m venv "$PROJECT_ROOT/.venv"
fi

echo 'Updating Python packaging tools...'
"$VENV_PYTHON" -m pip install --upgrade 'pip==25.0.1' 'setuptools>=75,<82' wheel

# --- accelerator selection -------------------------------------------------
NVIDIA_DETECTED=0
if [ "$CPU_ONLY" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 &&
   nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    NVIDIA_DETECTED=1
fi

if [ "$CPU_ONLY" -eq 1 ]; then
    RESOLVED_BACKEND=cpu
elif [ "$LLAMA_BACKEND" = auto ]; then
    RESOLVED_BACKEND=cpu   # no prebuilt Linux CUDA binary exists
else
    RESOLVED_BACKEND="$LLAMA_BACKEND"
fi

if [ "$NVIDIA_DETECTED" -eq 1 ]; then
    TORCH_INDEX=https://download.pytorch.org/whl/cu124
    EXPECTED_TORCH_TAG='+cu124'
else
    TORCH_INDEX=https://download.pytorch.org/whl/cpu
    EXPECTED_TORCH_TAG='+cpu'
fi

echo "Installing PyTorch 2.6.0 from $TORCH_INDEX..."
INSTALLED_TORCH="$("$VENV_PYTHON" -c '
try:
    import torch
    print(torch.__version__)
except Exception:
    print("")
' | tr -d '[:space:]')"

TORCH_ARGS=(-m pip install --upgrade)
# The CPU wheels carry no local version tag, so treat a bare version as +cpu.
CURRENT_TAG="+cpu"
case "$INSTALLED_TORCH" in *+*) CURRENT_TAG="+${INSTALLED_TORCH#*+}" ;; esac
if [ -n "$INSTALLED_TORCH" ] && [ "$CURRENT_TAG" != "$EXPECTED_TORCH_TAG" ]; then
    echo "Replacing incompatible Torch build $INSTALLED_TORCH with $EXPECTED_TORCH_TAG..."
    TORCH_ARGS+=(--force-reinstall)
fi
TORCH_ARGS+=(torch==2.6.0 torchaudio==2.6.0 --index-url "$TORCH_INDEX")
"$VENV_PYTHON" "${TORCH_ARGS[@]}"

# PyTorch's import-time NumPy bridge warns when NumPy is not present yet.
# Install the project pin before validating CUDA and before ONNX resolves it.
"$VENV_PYTHON" -m pip install 'numpy==2.5.0'

if [ "$NVIDIA_DETECTED" -eq 1 ] &&
   ! "$VENV_PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo 'WARNING: the NVIDIA driver was found, but PyTorch CUDA validation failed. Falling back to CPU.' >&2
    NVIDIA_DETECTED=0
    RESOLVED_BACKEND=cpu
    "$VENV_PYTHON" -m pip install --upgrade --force-reinstall \
        torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
fi

# --- onnx runtime ----------------------------------------------------------
echo 'Installing the matching ONNX Runtime...'
if [ "$NVIDIA_DETECTED" -eq 1 ]; then
    EXPECTED_ORT=onnxruntime-gpu
    OTHER_ORT=onnxruntime
else
    EXPECTED_ORT=onnxruntime
    OTHER_ORT=onnxruntime-gpu
fi
if "$VENV_PYTHON" - "$EXPECTED_ORT" "$OTHER_ORT" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

expected, other = sys.argv[1], sys.argv[2]
try:
    ready = version(expected) == "1.22.0"
except PackageNotFoundError:
    ready = False
try:
    version(other)
    ready = False          # the two builds must never coexist
except PackageNotFoundError:
    pass
raise SystemExit(0 if ready else 1)
PY
then
    echo "$EXPECTED_ORT 1.22.0 is already installed."
else
    "$VENV_PYTHON" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    "$VENV_PYTHON" -m pip install "$EXPECTED_ORT==1.22.0"
fi

# --- project dependencies --------------------------------------------------
echo 'Installing the LitheVoice Python dependencies...'
"$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt"

if ! "$VENV_PYTHON" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("en_core_web_sm") else 1)'; then
    echo 'Installing the Kokoro English language pipeline...'
    "$VENV_PYTHON" -m pip install \
        'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
fi

# --- models and runtime ----------------------------------------------------
DOWNLOAD_ARGS=("$SCRIPT_DIR/download_models.py" --backend "$RESOLVED_BACKEND")
[ "$INCLUDE_GPU_STT" -eq 1 ] && DOWNLOAD_ARGS+=(--include-gpu-stt)
[ "$SKIP_MODELS" -eq 1 ]     && DOWNLOAD_ARGS+=(--skip-models)
[ "$SKIP_LLAMA" -eq 1 ]      && DOWNLOAD_ARGS+=(--skip-llama)
[ "$SKIP_LLM" -eq 1 ]        && DOWNLOAD_ARGS+=(--skip-llm)
[ "$FORCE_DOWNLOADS" -eq 1 ] && DOWNLOAD_ARGS+=(--force)
echo "Downloading pinned models and llama.cpp ($RESOLVED_BACKEND)..."
"$VENV_PYTHON" "${DOWNLOAD_ARGS[@]}"

DOCTOR_ARGS=("$SCRIPT_DIR/doctor.py")
[ "$SKIP_MODELS" -eq 1 ] && DOCTOR_ARGS+=(--skip-models)
[ "$SKIP_LLAMA" -eq 1 ]  && DOCTOR_ARGS+=(--skip-llama)
[ "$SKIP_LLM" -eq 1 ]    && DOCTOR_ARGS+=(--skip-llm)
echo 'Checking the installation...'
"$VENV_PYTHON" "${DOCTOR_ARGS[@]}"

echo
echo 'LitheVoice is ready.'
echo 'Run: ./run.sh --barge-key'
