#!/usr/bin/env bash
set -euo pipefail

LIGHTNING_STUDIO_ROOT="${LIGHTNING_STUDIO_ROOT:-/teamspace/studios/this_studio/bimer}"
FEATURE_ROOT="${FEATURE_ROOT:-$LIGHTNING_STUDIO_ROOT/features-emotiontalk-train-v4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LIGHTNING_STUDIO_ROOT/output}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$LIGHTNING_STUDIO_ROOT/model-cache}"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/emotiontalk.jsonl}"
DRY_RUN="${DRY_RUN:-0}"

export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export YUNET_MODEL_PATH="$MODEL_CACHE_ROOT/yunet/face_detection_yunet_2023mar.onnx"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -f "$MANIFEST" ]]; then
    echo "EmotionTalk manifest is missing: $MANIFEST" >&2
    exit 3
  fi
  if [[ ! -f "$MODEL_CACHE_ROOT/ready.json" ]]; then
    echo "Model cache is missing: $MODEL_CACHE_ROOT/ready.json" >&2
    exit 4
  fi
  python - <<'PY'
import torch

if torch.cuda.device_count() < 1:
    raise RuntimeError("Lightning extraction requires one CUDA GPU")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
PY
fi

selected_start=""
selected_end=""
for start in 480 500 520 540 560 580; do
  end=$((start + 20))
  completion="$FEATURE_ROOT/ranges/range-$(printf '%05d' "$start")-$(printf '%05d' "$end").json"
  if [[ ! -f "$completion" ]]; then
    selected_start="$start"
    selected_end="$end"
    break
  fi
done

if [[ -z "$selected_start" ]]; then
  echo "All Lightning fifth-segment ranges are complete: [480,600)"
  exit 0
fi

extract_command=(
  bimer extract-features
  --manifest "$MANIFEST"
  --features "$FEATURE_ROOT"
  --staging "$FEATURE_ROOT"
  --yunet-model "$YUNET_MODEL_PATH"
  --dataset emotiontalk
  --split train
  --mode parallel
  --text-audio-device cuda:0
  --vision-device cuda:0
  --text-batch-size "${TEXT_BATCH_SIZE:-64}"
  --audio-batch-size "${AUDIO_BATCH_SIZE:-8}"
  --vision-batch-size "${VISION_BATCH_SIZE:-8}"
  --audio-workers "${AUDIO_WORKERS:-4}"
  --vision-workers "${VISION_WORKERS:-4}"
  --queue-capacity "${QUEUE_CAPACITY:-8}"
  --shard-size 16
  --start-shard "$selected_start"
  --end-shard "$selected_end"
)

verify_command=(
  bimer verify-features
  --manifest "$MANIFEST"
  --features "$FEATURE_ROOT"
  --dataset emotiontalk
  --split train
  --shard-size 16
  --start-shard "$selected_start"
  --end-shard "$selected_end"
  --write-completion
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY-RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1'
  printf ' %q' "${extract_command[@]}"
  printf '\n'
  printf 'DRY-RUN'
  printf ' %q' "${verify_command[@]}"
  printf '\n'
else
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${extract_command[@]}"
  "${verify_command[@]}"
fi

echo "Lightning range complete: [$selected_start,$selected_end)"
