#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-${LIGHTNING_STUDIO_ROOT:-/root/autodl-tmp/bimer-runtime}}"
FEATURE_ROOT="${FEATURE_ROOT:-$RUNTIME_ROOT/features-emotiontalk-train-v4}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$RUNTIME_ROOT/model-cache}"
MANIFEST="${MANIFEST:-$RUNTIME_ROOT/output/emotiontalk.jsonl}"
DATA_LOG="${DATA_LOG:-$RUNTIME_ROOT/data-http.log}"
DATA_PID="${DATA_PID:-}"
DRY_RUN="${DRY_RUN:-0}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RUNNER_SCRIPT="$SCRIPT_DIR/run_emotiontalk_lightning.sh"
if [[ "$DRY_RUN" == "1" ]]; then
  RUNNER_SCRIPT="scripts/run_emotiontalk_lightning.sh"
fi

export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export YUNET_MODEL_PATH="$MODEL_CACHE_ROOT/yunet/face_detection_yunet_2023mar.onnx"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN'
    LC_ALL=C printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

report_failure() {
  local status=$?
  if ((status != 0)); then
    echo "AUTODL_PIPELINE_FAILED status=$status" >&2
  fi
}
trap report_failure EXIT

if [[ "$DRY_RUN" != "1" ]]; then
  while ! grep -Fxq "DATA_COMPLETE" "$DATA_LOG" 2>/dev/null; do
    if [[ -n "$DATA_PID" ]] && ! kill -0 "$DATA_PID" 2>/dev/null; then
      echo "EmotionTalk data preparation stopped before DATA_COMPLETE" >&2
      exit 20
    fi
    sleep "$POLL_SECONDS"
  done
  if [[ ! -f "$MANIFEST" ]]; then
    echo "EmotionTalk manifest is missing after data preparation: $MANIFEST" >&2
    exit 21
  fi
  if [[ ! -f "$MODEL_CACHE_ROOT/ready.json" ]]; then
    echo "Model cache is not ready: $MODEL_CACHE_ROOT/ready.json" >&2
    exit 22
  fi
  rm -f "$RUNTIME_ROOT/.hf_token"
fi

smoke_completion="$FEATURE_ROOT/ranges/range-00480-00482.json"
if [[ "$DRY_RUN" == "1" || ! -f "$smoke_completion" ]]; then
  run env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bimer extract-features \
    --manifest "$MANIFEST" \
    --features "$FEATURE_ROOT" \
    --staging "$FEATURE_ROOT" \
    --yunet-model "$YUNET_MODEL_PATH" \
    --dataset emotiontalk \
    --split train \
    --mode parallel \
    --text-audio-device cuda:0 \
    --vision-device cuda:0 \
    --text-batch-size 64 \
    --audio-batch-size 8 \
    --vision-batch-size 8 \
    --audio-workers 4 \
    --vision-workers 4 \
    --queue-capacity 8 \
    --shard-size 16 \
    --start-shard 480 \
    --end-shard 482
  run bimer verify-features \
    --manifest "$MANIFEST" \
    --features "$FEATURE_ROOT" \
    --dataset emotiontalk \
    --split train \
    --shard-size 16 \
    --start-shard 480 \
    --end-shard 482 \
    --write-completion
fi

for _ in 1 2 3 4 5 6; do
  run env \
    LIGHTNING_STUDIO_ROOT="$RUNTIME_ROOT" \
    FEATURE_ROOT="$FEATURE_ROOT" \
    MODEL_CACHE_ROOT="$MODEL_CACHE_ROOT" \
    MANIFEST="$MANIFEST" \
    DRY_RUN="$DRY_RUN" \
    TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-64}" \
    AUDIO_BATCH_SIZE="${AUDIO_BATCH_SIZE:-8}" \
    VISION_BATCH_SIZE="${VISION_BATCH_SIZE:-8}" \
    AUDIO_WORKERS="${AUDIO_WORKERS:-4}" \
    VISION_WORKERS="${VISION_WORKERS:-4}" \
    QUEUE_CAPACITY="${QUEUE_CAPACITY:-8}" \
    bash "$RUNNER_SCRIPT"
done

run bimer verify-features \
  --manifest "$MANIFEST" \
  --features "$FEATURE_ROOT" \
  --dataset emotiontalk \
  --split train \
  --shard-size 16 \
  --start-shard 480 \
  --end-shard 600 \
  --write-completion

echo "AUTODL_PIPELINE_COMPLETE"
