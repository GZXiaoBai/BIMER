#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-${LIGHTNING_STUDIO_ROOT:-/root/autodl-tmp/bimer-runtime}}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$RUNTIME_ROOT/model-cache}"
MANIFEST="${MANIFEST:-$RUNTIME_ROOT/output/bilingual.jsonl}"
BASE_FEATURES="${BASE_FEATURES:-$RUNTIME_ROOT/features-bilingual-test-standard}"
ROBUSTNESS_ROOT="${ROBUSTNESS_ROOT:-$RUNTIME_ROOT/robustness}"
ASR_MANIFEST="${ASR_MANIFEST:-$RUNTIME_ROOT/output/whisper-test.jsonl}"
ASR_ERROR_LOG="${ASR_ERROR_LOG:-$RUNTIME_ROOT/reports/whisper-test-errors.jsonl}"
YUNET_MODEL_PATH="${YUNET_MODEL_PATH:-$MODEL_CACHE_ROOT/yunet/face_detection_yunet_2023mar.onnx}"
DRY_RUN="${DRY_RUN:-0}"
SHARD_SIZE=16

export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
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
    echo "AUTODL_WHISPER_ROBUSTNESS_FAILED status=$status" >&2
  fi
}
trap report_failure EXIT

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f "$MANIFEST" ]] || {
    echo "Combined manifest is missing: $MANIFEST" >&2
    exit 20
  }
  [[ -d "$BASE_FEATURES/meld/test" ]] || {
    echo "MELD standard test features are missing: $BASE_FEATURES/meld/test" >&2
    exit 21
  }
  [[ -d "$BASE_FEATURES/emotiontalk/test" ]] || {
    echo "EmotionTalk standard test features are missing: $BASE_FEATURES/emotiontalk/test" >&2
    exit 22
  }
  [[ -f "$MODEL_CACHE_ROOT/ready.json" ]] || {
    echo "Model cache is not ready: $MODEL_CACHE_ROOT/ready.json" >&2
    exit 23
  }
  [[ -f "$YUNET_MODEL_PATH" ]] || {
    echo "YuNet model is missing: $YUNET_MODEL_PATH" >&2
    exit 24
  }
  python -c 'import faster_whisper' || {
    echo "faster-whisper is not installed" >&2
    exit 25
  }
  if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "A CUDA GPU is required for Whisper robustness extraction" >&2
    exit 26
  fi
fi

run mkdir -p "$(dirname "$ASR_MANIFEST")" "$(dirname "$ASR_ERROR_LOG")" "$ROBUSTNESS_ROOT"

run env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m bimer.cli asr-manifest \
    --manifest "$MANIFEST" \
    --output "$ASR_MANIFEST" \
    --split test \
    --device cuda \
    --keep-original-on-error \
    --error-log "$ASR_ERROR_LOG"

run python -c \
  'import sys; from bimer.manifest import read_manifest; source=[row for row in read_manifest(sys.argv[1]) if str(row.split)=="test"]; asr=read_manifest(sys.argv[2]); assert [row.sample_id for row in source]==[row.sample_id for row in asr]; print(f"verified_whisper_rows={len(asr)}")' \
  "$MANIFEST" "$ASR_MANIFEST"

feature_root="$ROBUSTNESS_ROOT/whisper_text"
run env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m bimer.cli extract-features \
    --manifest "$ASR_MANIFEST" \
    --features "$feature_root" \
    --base-features "$BASE_FEATURES" \
    --yunet-model "$YUNET_MODEL_PATH" \
    --split test \
    --mode parallel \
    --only-modality text \
    --condition-name whisper_text \
    --shard-size "$SHARD_SIZE" \
    --text-audio-device cuda:0 \
    --text-batch-size "${TEXT_BATCH_SIZE:-64}"

run python -m bimer.cli verify-features \
  --manifest "$ASR_MANIFEST" \
  --features "$feature_root" \
  --dataset meld \
  --split test \
  --shard-size "$SHARD_SIZE" \
  --start-shard 0 \
  --end-shard 164 \
  --write-completion

run python -m bimer.cli verify-features \
  --manifest "$ASR_MANIFEST" \
  --features "$feature_root" \
  --dataset emotiontalk \
  --split test \
  --shard-size "$SHARD_SIZE" \
  --start-shard 0 \
  --end-shard 121 \
  --write-completion

run touch "$ROBUSTNESS_ROOT/AUTODL_WHISPER_ROBUSTNESS_COMPLETE"
echo "AUTODL_WHISPER_ROBUSTNESS_COMPLETE"
