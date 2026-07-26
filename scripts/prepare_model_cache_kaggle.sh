#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/kaggle/working/bimer-model-cache}"
MODEL_CACHE_READY_PATH="${MODEL_CACHE_READY_PATH:-$MODEL_CACHE_ROOT/ready.json}"
YUNET_MODEL_PATH="${YUNET_MODEL_PATH:-$MODEL_CACHE_ROOT/yunet/face_detection_yunet_2023mar.onnx}"
HF_HOME="${HF_HOME:-$MODEL_CACHE_ROOT/huggingface}"
TORCH_HOME="${TORCH_HOME:-$MODEL_CACHE_ROOT/torch}"
MODEL_DOWNLOAD_MAX_ATTEMPTS="${MODEL_DOWNLOAD_MAX_ATTEMPTS:-3}"
MODEL_DOWNLOAD_MAX_SECONDS="${MODEL_DOWNLOAD_MAX_SECONDS:-3600}"
MODEL_DOWNLOAD_RETRY_DELAY_SECONDS="${MODEL_DOWNLOAD_RETRY_DELAY_SECONDS:-10}"
PYTHON="${PYTHON:-python}"

export MODEL_CACHE_ROOT MODEL_CACHE_READY_PATH YUNET_MODEL_PATH HF_HOME TORCH_HOME
mkdir -p "$MODEL_CACHE_ROOT" "$HF_HOME" "$TORCH_HOME"
rm -f "$MODEL_CACHE_READY_PATH"

retry_hf_download() {
  local repo="$1"
  local attempt
  for ((attempt = 1; attempt <= MODEL_DOWNLOAD_MAX_ATTEMPTS; attempt++)); do
    echo "Downloading $repo (attempt $attempt/$MODEL_DOWNLOAD_MAX_ATTEMPTS)" >&2
    if HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
      timeout --signal=TERM --kill-after=30 "$MODEL_DOWNLOAD_MAX_SECONDS" \
      hf download "$repo" \
        --max-workers 1 \
        --include '*.json' '*.model' '*.safetensors' 'pytorch_model.bin' '*.txt'; then
      return 0
    fi
    echo "$repo attempt $attempt/$MODEL_DOWNLOAD_MAX_ATTEMPTS failed" >&2
    if ((attempt < MODEL_DOWNLOAD_MAX_ATTEMPTS)); then
      sleep "$MODEL_DOWNLOAD_RETRY_DELAY_SECONDS"
    fi
  done
  echo "Failed to cache $repo after $MODEL_DOWNLOAD_MAX_ATTEMPTS attempts" >&2
  return 1
}

retry_python_step() {
  local description="$1"
  local program="$2"
  local attempt
  for ((attempt = 1; attempt <= MODEL_DOWNLOAD_MAX_ATTEMPTS; attempt++)); do
    echo "$description (attempt $attempt/$MODEL_DOWNLOAD_MAX_ATTEMPTS)" >&2
    if "$PYTHON" -c "$program"; then
      return 0
    fi
    echo "$description attempt $attempt/$MODEL_DOWNLOAD_MAX_ATTEMPTS failed" >&2
  done
  echo "$description failed after $MODEL_DOWNLOAD_MAX_ATTEMPTS attempts" >&2
  return 1
}

retry_hf_download "xlm-roberta-base"
retry_hf_download "facebook/wav2vec2-xls-r-300m"

if [[ ! -f "$YUNET_MODEL_PATH" ]]; then
  "$SCRIPT_DIR/download_yunet.sh" "$YUNET_MODEL_PATH"
fi

R3D_CACHE_PROGRAM=$(cat <<'PY'
from torchvision.models.video import R3D_18_Weights, r3d_18

r3d_18(weights=R3D_18_Weights.DEFAULT).eval()
PY
)
retry_python_step "Caching R3D-18 weights" "$R3D_CACHE_PROGRAM"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON" - <<'PY'
import gc
import json
import os
from pathlib import Path

import cv2
from torchvision.models.video import R3D_18_Weights, r3d_18
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer


def release(model):
    del model
    gc.collect()


AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
model = AutoModel.from_pretrained("xlm-roberta-base", local_files_only=True).eval()
release(model)

AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    local_files_only=True,
)
model = AutoModel.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    local_files_only=True,
).eval()
release(model)

model = r3d_18(weights=R3D_18_Weights.DEFAULT).eval()
release(model)

yunet_path = Path(os.environ["YUNET_MODEL_PATH"])
cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.8, 0.3, 5000)

ready_path = Path(os.environ["MODEL_CACHE_READY_PATH"])
ready_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = ready_path.with_suffix(ready_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "offline_verified": True,
            "models": [
                "xlm-roberta-base",
                "facebook/wav2vec2-xls-r-300m",
                "torchvision/r3d_18",
                "opencv/face_detection_yunet_2023mar",
            ],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
temporary_path.replace(ready_path)
PY

if [[ ! -f "$MODEL_CACHE_READY_PATH" ]]; then
  echo "Offline verification did not create $MODEL_CACHE_READY_PATH" >&2
  exit 1
fi

echo "Model cache ready: $MODEL_CACHE_READY_PATH"
