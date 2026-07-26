#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-/root/autodl-tmp/bimer-runtime}"
PYTHON="${BIMER_PYTHON:-/root/miniconda3/bin/python3}"
MANIFEST="${BIMER_MANIFEST:-$ROOT/data/processed/v2/all.jsonl}"
BASE="${BIMER_BASE_FEATURES:-$ROOT/artifacts/features/bilingual-v1}"
QUALITY="${BIMER_QUALITY_FEATURES:-$ROOT/artifacts/features/bilingual-v2-quality}"
YUNET="${BIMER_YUNET_MODEL:-$RUNTIME_ROOT/model-cache/yunet/face_detection_yunet_2023mar.onnx}"
JOINT_MANIFEST="$ROOT/data/processed/v2/corruption-train-10pct-joint.jsonl"
JOINT_ASR_MANIFEST="$ROOT/data/processed/v2/corruption-train-10pct-joint-asr.jsonl"
JOINT_CLEAN="$ROOT/artifacts/features/v2-corruption-joint-clean"
JOINT_AUDIO="$ROOT/artifacts/features/v2-corruption-joint-audio10"
JOINT_VIDEO="$ROOT/artifacts/features/v2-corruption-joint-video50"
JOINT_WHISPER="$ROOT/artifacts/features/v2-corruption-joint-whisper"
OUTPUT="$ROOT/artifacts/experiments/v2"
STATUS="$ROOT/artifacts/experiments/v2-meld-quality-screen/_status"
LOG_DIR="$ROOT/artifacts/logs"
ARCHIVE="$ROOT/artifacts/bimer-v2-meld-quality-screen-seed42.tar.gz"
WORKERS="${BIMER_QUALITY_WORKERS:-32}"
QUEUE_CAPACITY="${BIMER_QUALITY_QUEUE_CAPACITY:-64}"
SHARDS_PER_RANGE="${BIMER_QUALITY_SHARDS_PER_RANGE:-80}"
AUGMENT_WORKERS="${BIMER_AUGMENT_WORKERS:-18}"
AUGMENT_QUEUE_CAPACITY="${BIMER_AUGMENT_QUEUE_CAPACITY:-36}"

mkdir -p "$STATUS" "$LOG_DIR"
rm -f "$STATUS/RUN_FAILED" "$STATUS/DOWNLOAD_READY"

on_exit() {
  status=$?
  trap - EXIT
  if [[ "$status" -eq 0 ]]; then
    tar -C "$ROOT" -czf "$ARCHIVE" \
      data/processed/v2/corruption-train-10pct-joint.jsonl \
      data/processed/v2/corruption-train-10pct-joint-asr.jsonl \
      artifacts/features/bilingual-v2-quality/meld \
      artifacts/features/v2-corruption-joint-clean \
      artifacts/features/v2-corruption-joint-audio10 \
      artifacts/features/v2-corruption-joint-video50 \
      artifacts/features/v2-corruption-joint-whisper \
      artifacts/experiments/v2/screen
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
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-$RUNTIME_ROOT/model-cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$ROOT"

"$PYTHON" - "$MANIFEST" <<'PY'
from pathlib import Path
import sys

from bimer.manifest import read_manifest

records = [record for record in read_manifest(sys.argv[1]) if record.dataset == "meld"]
available = sum(
    record.video_path is not None and Path(record.video_path).is_file()
    for record in records
)
if len(records) != 13_708:
    raise SystemExit(f"expected 13708 MELD records, found {len(records)}")
if available < int(0.99 * len(records)):
    raise SystemExit(f"only {available}/{len(records)} MELD media paths are accessible")
print(f"MELD media preflight: {available}/{len(records)} accessible")
PY

for split in train dev test; do
  shard_count=$(
    find -L "$BASE/meld/$split" -maxdepth 1 -type f \
      -name 'features-*.npz' | wc -l | tr -d ' '
  )
  if [[ "$shard_count" -le 0 ]]; then
    echo "no base feature shards for meld/$split" >&2
    exit 22
  fi
  for ((start_shard = 0; start_shard < shard_count; start_shard += SHARDS_PER_RANGE)); do
    end_shard=$((start_shard + SHARDS_PER_RANGE))
    if ((end_shard > shard_count)); then
      end_shard=$shard_count
    fi
    printf 'QUALITY_RANGE dataset=meld split=%s start=%d end=%d total=%d\n' \
      "$split" "$start_shard" "$end_shard" "$shard_count"
    "$PYTHON" -m bimer.cli attach-quality \
      --manifest "$MANIFEST" \
      --base-features "$BASE" \
      --output-features "$QUALITY" \
      --yunet-model "$YUNET" \
      --dataset meld \
      --split "$split" \
      --workers "$WORKERS" \
      --queue-capacity "$QUEUE_CAPACITY" \
      --start-shard "$start_shard" \
      --end-shard "$end_shard"
  done
done

"$PYTHON" -m bimer.cli sample-corruption-manifest \
  --manifest "$MANIFEST" \
  --output-manifest "$JOINT_MANIFEST" \
  --base-features "$QUALITY" \
  --output-features "$JOINT_CLEAN" \
  --fraction 0.1 \
  --seed 42

for dataset in meld emotiontalk; do
  "$PYTHON" -m bimer.cli extract-features \
    --manifest "$JOINT_MANIFEST" \
    --dataset "$dataset" \
    --split train \
    --base-features "$JOINT_CLEAN" \
    --features "$JOINT_AUDIO" \
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
    --manifest "$JOINT_MANIFEST" \
    --dataset "$dataset" \
    --split train \
    --base-features "$JOINT_CLEAN" \
    --features "$JOINT_VIDEO" \
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
done

if [[ ! -s "$JOINT_ASR_MANIFEST" ]] || \
   [[ "$(wc -l < "$JOINT_ASR_MANIFEST")" != "$(wc -l < "$JOINT_MANIFEST")" ]]; then
  "$PYTHON" -m bimer.cli asr-manifest \
    --manifest "$JOINT_MANIFEST" \
    --output "$JOINT_ASR_MANIFEST" \
    --device cuda \
    --keep-original-on-error \
    --error-log "$JOINT_WHISPER/asr-errors.jsonl"
fi

for dataset in meld emotiontalk; do
  "$PYTHON" -m bimer.cli extract-features \
    --manifest "$JOINT_ASR_MANIFEST" \
    --dataset "$dataset" \
    --split train \
    --base-features "$JOINT_CLEAN" \
    --features "$JOINT_WHISPER" \
    --yunet-model "$YUNET" \
    --mode parallel \
    --only-modality text \
    --condition-name train-whisper-text \
    --audio-workers "$AUGMENT_WORKERS" \
    --vision-workers "$AUGMENT_WORKERS" \
    --queue-capacity "$AUGMENT_QUEUE_CAPACITY" \
    --text-audio-device cuda:0 \
    --vision-device cuda:0
done

touch "$ROOT/artifacts/features/v2-meld-quality-and-joint-views.DONE"

"$PYTHON" scripts/run_v2_experiments.py \
  --stage fusion-screen \
  --variant lagf \
  --variant lagf_no_gates \
  --variant quality_lagf \
  --quality-features "$QUALITY" \
  --output "$OUTPUT" \
  --augmentation-manifest "$JOINT_MANIFEST" \
  --augmentation-features "$JOINT_AUDIO" \
  --augmentation-manifest "$JOINT_MANIFEST" \
  --augmentation-features "$JOINT_VIDEO" \
  --augmentation-manifest "$JOINT_ASR_MANIFEST" \
  --augmentation-features "$JOINT_WHISPER" \
  --device cuda
