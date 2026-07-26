#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-${LIGHTNING_STUDIO_ROOT:-/root/autodl-tmp/bimer-runtime}}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$RUNTIME_ROOT/model-cache}"
MANIFEST="${MANIFEST:-$RUNTIME_ROOT/output/emotiontalk.jsonl}"
REPORT_ROOT="${REPORT_ROOT:-$RUNTIME_ROOT/reports}"
DRY_RUN="${DRY_RUN:-0}"
SHARD_SIZE=16
RANGE_SIZE="${RANGE_SIZE:-20}"

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
    echo "AUTODL_EMOTIONTALK_EVAL_FAILED status=$status" >&2
  fi
}
trap report_failure EXIT

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -f "$MANIFEST" ]]; then
    echo "EmotionTalk manifest is missing: $MANIFEST" >&2
    exit 20
  fi
  if [[ ! -f "$MODEL_CACHE_ROOT/ready.json" ]]; then
    echo "Model cache is not ready: $MODEL_CACHE_ROOT/ready.json" >&2
    exit 21
  fi
  if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "A CUDA GPU is required; do not run this queue in no-card mode" >&2
    exit 22
  fi
fi

run mkdir -p "$REPORT_ROOT"

completion_is_valid() {
  local path=$1
  [[ -f "$path" ]] && grep -Eq '"is_valid"[[:space:]]*:[[:space:]]*true' "$path"
}

extract_split() {
  local split=$1
  local total_shards=$2
  local feature_root="$RUNTIME_ROOT/features-emotiontalk-${split}-v4"
  local start end completion

  echo "BEGIN emotiontalk/$split shards=$total_shards"
  for ((start = 0; start < total_shards; start += RANGE_SIZE)); do
    end=$((start + RANGE_SIZE))
    if ((end > total_shards)); then
      end=$total_shards
    fi
    completion="$feature_root/ranges/range-$(printf '%05d' "$start")-$(printf '%05d' "$end").json"

    if completion_is_valid "$completion"; then
      echo "SKIP $split completed range [$start,$end)"
      continue
    fi

    run env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bimer extract-features \
      --manifest "$MANIFEST" \
      --features "$feature_root" \
      --staging "$feature_root" \
      --yunet-model "$YUNET_MODEL_PATH" \
      --dataset emotiontalk \
      --split "$split" \
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
      --features "$feature_root" \
      --dataset emotiontalk \
      --split "$split" \
      --shard-size "$SHARD_SIZE" \
      --start-shard "$start" \
      --end-shard "$end" \
      --write-completion
  done

  run bimer verify-features \
    --manifest "$MANIFEST" \
    --features "$feature_root" \
    --dataset emotiontalk \
    --split "$split" \
    --shard-size "$SHARD_SIZE" \
    --start-shard 0 \
    --end-shard "$total_shards" \
    --write-completion

  if [[ "$DRY_RUN" == "1" ]] || bimer --help 2>&1 | grep -Fq "feature-stats"; then
    run bimer feature-stats \
      --manifest "$MANIFEST" \
      --features "$feature_root" \
      --dataset emotiontalk \
      --split "$split" \
      --output "$REPORT_ROOT/emotiontalk-${split}-feature-statistics.json"
  else
    echo "SKIP $split feature-stats command unavailable; compute after download"
  fi

  echo "COMPLETE emotiontalk/$split"
}

extract_split validation 120
extract_split test 121

echo "AUTODL_EMOTIONTALK_EVAL_COMPLETE"
