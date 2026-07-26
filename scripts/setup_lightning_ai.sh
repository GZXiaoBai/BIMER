#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LIGHTNING_STUDIO_ROOT="${LIGHTNING_STUDIO_ROOT:-/teamspace/studios/this_studio/bimer}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run mkdir -p \
  "$LIGHTNING_STUDIO_ROOT/data" \
  "$LIGHTNING_STUDIO_ROOT/output" \
  "$LIGHTNING_STUDIO_ROOT/model-cache" \
  "$LIGHTNING_STUDIO_ROOT/features-emotiontalk-train-v4"

if [[ "$DRY_RUN" == "1" ]] || ! command -v ffmpeg >/dev/null 2>&1; then
  run sudo apt-get update
  run sudo apt-get install -y ffmpeg
fi

run "$PYTHON" -m pip install --upgrade pip
run "$PYTHON" -m pip install --no-deps --editable "$PROJECT_ROOT"
run "$PYTHON" -m pip install --upgrade \
  'numpy>=2.0,<3' \
  'pandas>=2.2,<3' \
  'scikit-learn>=1.5,<2' \
  'transformers==4.49.0' \
  'huggingface_hub[hf_xet]>=0.30,<1.0' \
  'opencv-python-headless>=4.10,<5' \
  'pytest>=8,<9'

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" - <<'PY'
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
fi

echo "Lightning environment ready: $LIGHTNING_STUDIO_ROOT"
