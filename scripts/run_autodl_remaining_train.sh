#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-${LIGHTNING_STUDIO_ROOT:-/root/autodl-tmp/bimer-runtime}}"
FEATURE_ROOT="${FEATURE_ROOT:-$RUNTIME_ROOT/features-emotiontalk-train-v4}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$RUNTIME_ROOT/model-cache}"
MANIFEST="${MANIFEST:-$RUNTIME_ROOT/output/emotiontalk.jsonl}"
CURRENT_PIPELINE_LOG="${CURRENT_PIPELINE_LOG:-$RUNTIME_ROOT/pipeline.log}"
DRY_RUN="${DRY_RUN:-0}"
POLL_SECONDS="${POLL_SECONDS:-30}"
TOTAL_SHARDS=964
SHARD_SIZE=16

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
    echo "AUTODL_REMAINING_TRAIN_FAILED status=$status" >&2
  fi
}
trap report_failure EXIT

current_completion="$FEATURE_ROOT/ranges/range-00480-00600.json"
echo "WAIT-FOR $(basename "$current_completion")"
if [[ "$DRY_RUN" != "1" ]]; then
  while [[ ! -f "$current_completion" ]]; do
    if grep -Fq "AUTODL_PIPELINE_FAILED" "$CURRENT_PIPELINE_LOG" 2>/dev/null; then
      echo "Current [480,600) pipeline failed before completion" >&2
      exit 20
    fi
    sleep "$POLL_SECONDS"
  done

  if [[ ! -f "$MANIFEST" ]]; then
    echo "EmotionTalk manifest is missing: $MANIFEST" >&2
    exit 21
  fi
  if [[ ! -f "$MODEL_CACHE_ROOT/ready.json" ]]; then
    echo "Model cache is not ready: $MODEL_CACHE_ROOT/ready.json" >&2
    exit 22
  fi
fi

extract_range() {
  local start=$1
  local end=$2
  local completion
  completion="$FEATURE_ROOT/ranges/range-$(printf '%05d' "$start")-$(printf '%05d' "$end").json"

  if [[ "$DRY_RUN" != "1" && -f "$completion" ]]; then
    echo "SKIP completed range [$start,$end)"
    return
  fi

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
    --text-batch-size "${TEXT_BATCH_SIZE:-64}" \
    --audio-batch-size "${AUDIO_BATCH_SIZE:-8}" \
    --vision-batch-size "${VISION_BATCH_SIZE:-8}" \
    --audio-workers "${AUDIO_WORKERS:-4}" \
    --vision-workers "${VISION_WORKERS:-4}" \
    --queue-capacity "${QUEUE_CAPACITY:-8}" \
    --shard-size "$SHARD_SIZE" \
    --start-shard "$start" \
    --end-shard "$end"

  run bimer verify-features \
    --manifest "$MANIFEST" \
    --features "$FEATURE_ROOT" \
    --dataset emotiontalk \
    --split train \
    --shard-size "$SHARD_SIZE" \
    --start-shard "$start" \
    --end-shard "$end" \
    --write-completion
}

verify_major_range() {
  local start=$1
  local end=$2
  local completion
  completion="$FEATURE_ROOT/ranges/range-$(printf '%05d' "$start")-$(printf '%05d' "$end").json"

  if [[ "$DRY_RUN" != "1" && -f "$completion" ]]; then
    echo "SKIP completed major range [$start,$end)"
    return
  fi

  run bimer verify-features \
    --manifest "$MANIFEST" \
    --features "$FEATURE_ROOT" \
    --dataset emotiontalk \
    --split train \
    --shard-size "$SHARD_SIZE" \
    --start-shard "$start" \
    --end-shard "$end" \
    --write-completion
}

major_start=600
for ((start = 600; start < TOTAL_SHARDS; start += 20)); do
  end=$((start + 20))
  if ((end > TOTAL_SHARDS)); then
    end=$TOTAL_SHARDS
  fi

  extract_range "$start" "$end"

  if ((end == 720 || end == 840 || end == 960 || end == TOTAL_SHARDS)); then
    verify_major_range "$major_start" "$end"
    major_start=$end
  fi
done

verify_major_range 600 "$TOTAL_SHARDS"
echo "AUTODL_REMAINING_TRAIN_COMPLETE"
