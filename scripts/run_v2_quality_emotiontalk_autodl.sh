#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-/root/autodl-tmp/bimer-runtime}"
PYTHON="${BIMER_PYTHON:-/root/miniconda3/bin/python3}"
MANIFEST="${BIMER_MANIFEST:-$RUNTIME_ROOT/output/emotiontalk.jsonl}"
BASE="${BIMER_BASE_FEATURES:-$ROOT/artifacts/features/emotiontalk-v4}"
QUALITY="${BIMER_QUALITY_FEATURES:-$ROOT/artifacts/features/bilingual-v2-quality}"
YUNET="${BIMER_YUNET_MODEL:-$RUNTIME_ROOT/model-cache/yunet/face_detection_yunet_2023mar.onnx}"
SELECTED_MANIFEST="$ROOT/data/processed/v2/corruption-train-10pct-emotiontalk.jsonl"
SELECTED_ASR_MANIFEST="$ROOT/data/processed/v2/corruption-train-10pct-emotiontalk-asr.jsonl"
SELECTED_CLEAN="$ROOT/artifacts/features/v2-corruption-clean"
WORKERS="${BIMER_QUALITY_WORKERS:-48}"
QUEUE_CAPACITY="${BIMER_QUALITY_QUEUE_CAPACITY:-96}"
SHARDS_PER_RANGE="${BIMER_QUALITY_SHARDS_PER_RANGE:-120}"
AUGMENT_WORKERS="${BIMER_AUGMENT_WORKERS:-24}"
AUGMENT_QUEUE_CAPACITY="${BIMER_AUGMENT_QUEUE_CAPACITY:-48}"
STATUS="$ROOT/artifacts/experiments/v2-quality-emotiontalk/_status"
LOG_DIR="$ROOT/artifacts/logs"
ARCHIVE="$ROOT/artifacts/bimer-v2-emotiontalk-quality.tar.gz"

mkdir -p "$STATUS" "$LOG_DIR"
rm -f "$STATUS/RUN_FAILED" "$STATUS/DOWNLOAD_READY"

on_exit() {
  status=$?
  trap - EXIT
  if [[ "$status" -eq 0 ]]; then
    tar -C "$ROOT" -czf "$ARCHIVE" \
      data/processed/v2/corruption-train-10pct-emotiontalk.jsonl \
      data/processed/v2/corruption-train-10pct-emotiontalk-asr.jsonl \
      artifacts/features/bilingual-v2-quality/emotiontalk \
      artifacts/features/v2-corruption-clean/emotiontalk \
      artifacts/features/v2-corruption-audio10/emotiontalk \
      artifacts/features/v2-corruption-video50/emotiontalk \
      artifacts/features/v2-corruption-whisper/emotiontalk
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    touch "$STATUS/DOWNLOAD_READY"
  else
    printf '%s\n' "$status" > "$STATUS/RUN_FAILED"
  fi
  if [[ "${AUTODL_AUTO_SHUTDOWN:-0}" == "1" ]]; then
    shutdown -h now || sudo shutdown -h now || true
  fi
  exit "$status"
}
trap on_exit EXIT

export PATH="/root/miniconda3/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/model-cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-$RUNTIME_ROOT/model-cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$ROOT"

"$PYTHON" - "$MANIFEST" <<'PY'
from pathlib import Path
import sys

from bimer.manifest import read_manifest

records = [
    record
    for record in read_manifest(sys.argv[1])
    if record.dataset == "emotiontalk"
]
available = sum(
    record.video_path is not None and Path(record.video_path).is_file()
    for record in records
)
if len(records) != 19_250:
    raise SystemExit(f"expected 19250 EmotionTalk records, found {len(records)}")
if available < int(0.99 * len(records)):
    raise SystemExit(
        f"only {available}/{len(records)} EmotionTalk media paths are accessible"
    )
print(f"EmotionTalk media preflight: {available}/{len(records)} accessible")
PY

for split in train validation test; do
  shard_count=$(
    find "$BASE/emotiontalk/$split" -maxdepth 1 -type f \
      -name 'features-*.npz' | wc -l | tr -d ' '
  )
  if [[ "$shard_count" -le 0 ]]; then
    echo "no base feature shards for emotiontalk/$split" >&2
    exit 22
  fi
  for ((start_shard = 0; start_shard < shard_count; start_shard += SHARDS_PER_RANGE)); do
    end_shard=$((start_shard + SHARDS_PER_RANGE))
    if ((end_shard > shard_count)); then
      end_shard=$shard_count
    fi
    printf 'QUALITY_RANGE dataset=emotiontalk split=%s start=%d end=%d total=%d\n' \
      "$split" "$start_shard" "$end_shard" "$shard_count"
    "$PYTHON" -m bimer.cli attach-quality \
      --manifest "$MANIFEST" \
      --base-features "$BASE" \
      --output-features "$QUALITY" \
      --yunet-model "$YUNET" \
      --dataset emotiontalk \
      --split "$split" \
      --workers "$WORKERS" \
      --queue-capacity "$QUEUE_CAPACITY" \
      --start-shard "$start_shard" \
      --end-shard "$end_shard"
  done
done

"$PYTHON" -m bimer.cli sample-corruption-manifest \
  --manifest "$MANIFEST" \
  --output-manifest "$SELECTED_MANIFEST" \
  --base-features "$QUALITY" \
  --output-features "$SELECTED_CLEAN" \
  --dataset emotiontalk \
  --fraction 0.1 \
  --seed 42

"$PYTHON" -m bimer.cli extract-features \
  --manifest "$SELECTED_MANIFEST" \
  --dataset emotiontalk \
  --split train \
  --base-features "$SELECTED_CLEAN" \
  --features "$ROOT/artifacts/features/v2-corruption-audio10" \
  --yunet-model "$YUNET" \
  --mode parallel \
  --only-modality audio \
  --condition-name train-audio-10db \
  --audio-snr 10 \
  --audio-workers "$AUGMENT_WORKERS" \
  --vision-workers "$AUGMENT_WORKERS" \
  --queue-capacity "$AUGMENT_QUEUE_CAPACITY" \
  --text-audio-device cuda:0 \
  --vision-device cuda:0

"$PYTHON" -m bimer.cli extract-features \
  --manifest "$SELECTED_MANIFEST" \
  --dataset emotiontalk \
  --split train \
  --base-features "$SELECTED_CLEAN" \
  --features "$ROOT/artifacts/features/v2-corruption-video50" \
  --yunet-model "$YUNET" \
  --mode parallel \
  --only-modality vision \
  --condition-name train-video-drop-50 \
  --frame-drop 0.5 \
  --audio-workers "$AUGMENT_WORKERS" \
  --vision-workers "$AUGMENT_WORKERS" \
  --queue-capacity "$AUGMENT_QUEUE_CAPACITY" \
  --text-audio-device cuda:0 \
  --vision-device cuda:0

"$PYTHON" -m bimer.cli asr-manifest \
  --manifest "$SELECTED_MANIFEST" \
  --output "$SELECTED_ASR_MANIFEST" \
  --device cuda \
  --keep-original-on-error \
  --error-log "$ROOT/artifacts/features/v2-corruption-whisper/asr-errors-emotiontalk.jsonl"

"$PYTHON" -m bimer.cli extract-features \
  --manifest "$SELECTED_ASR_MANIFEST" \
  --dataset emotiontalk \
  --split train \
  --base-features "$SELECTED_CLEAN" \
  --features "$ROOT/artifacts/features/v2-corruption-whisper" \
  --yunet-model "$YUNET" \
  --mode parallel \
  --only-modality text \
  --condition-name train-whisper-text \
  --audio-workers "$AUGMENT_WORKERS" \
  --vision-workers "$AUGMENT_WORKERS" \
  --queue-capacity "$AUGMENT_QUEUE_CAPACITY" \
  --text-audio-device cuda:0 \
  --vision-device cuda:0

touch "$ROOT/artifacts/features/v2-quality-emotiontalk.DONE"
