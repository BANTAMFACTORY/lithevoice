#!/usr/bin/env bash
# Linux launcher. Mirrors run.ps1.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo 'LitheVoice is not installed. Run ./scripts/setup.sh first.' >&2
    exit 1
fi

export PYTHONUTF8=1
export HF_HOME="$PROJECT_ROOT/models/huggingface"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

exec "$PYTHON" "$PROJECT_ROOT/realtime.py" "$@"
