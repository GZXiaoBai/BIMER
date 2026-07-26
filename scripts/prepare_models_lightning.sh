#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTNING_STUDIO_ROOT="${LIGHTNING_STUDIO_ROOT:-/teamspace/studios/this_studio/bimer}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$LIGHTNING_STUDIO_ROOT/model-cache}"
MODEL_CACHE_READY_PATH="$MODEL_CACHE_ROOT/ready.json"

export MODEL_CACHE_ROOT MODEL_CACHE_READY_PATH
export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export YUNET_MODEL_PATH="$MODEL_CACHE_ROOT/yunet/face_detection_yunet_2023mar.onnx"

if [[ -f "$MODEL_CACHE_READY_PATH" ]]; then
  echo "Lightning model cache already ready: $MODEL_CACHE_READY_PATH"
  exit 0
fi

bash "$SCRIPT_DIR/prepare_model_cache_kaggle.sh"
