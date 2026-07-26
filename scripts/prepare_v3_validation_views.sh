#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
MANIFEST="${BIMER_MANIFEST:-$ROOT/data/processed/v2/all.jsonl}"
BASE="${BIMER_FEATURES:-$ROOT/artifacts/features/bilingual-v2-quality}"
OUTPUT="${BIMER_V3_VALIDATION:-$ROOT/artifacts/features/v3-validation}"
YUNET="${BIMER_YUNET:-$ROOT/artifacts/models/face_detection_yunet_2023mar.onnx}"
DEVICE="${BIMER_DEVICE:-cuda:0}"
WORKERS="${BIMER_CPU_WORKERS:-8}"
SHARD_SIZE="${BIMER_SHARD_SIZE:-16}"

mkdir -p "$OUTPUT/manifests"
VALIDATION_MANIFEST="$OUTPUT/manifests/validation-clean.jsonl"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$ROOT/scripts/build_v3_validation_manifest.py" \
  --manifest "$MANIFEST" \
  --output "$VALIDATION_MANIFEST"

for dataset_split in "meld dev" "emotiontalk validation"; do
  read -r dataset split <<<"$dataset_split"
  python3 -m bimer.cli extract-features \
    --manifest "$VALIDATION_MANIFEST" \
    --features "$OUTPUT/audio-10db" \
    --base-features "$BASE" \
    --yunet-model "$YUNET" \
    --dataset "$dataset" \
    --split "$split" \
    --shard-size "$SHARD_SIZE" \
    --mode parallel \
    --only-modality audio \
    --condition-name validation_audio_10db \
    --audio-snr 10 \
    --text-audio-device "$DEVICE" \
    --vision-device "$DEVICE" \
    --audio-workers "$WORKERS"

  python3 -m bimer.cli extract-features \
    --manifest "$VALIDATION_MANIFEST" \
    --features "$OUTPUT/video-50" \
    --base-features "$BASE" \
    --yunet-model "$YUNET" \
    --dataset "$dataset" \
    --split "$split" \
    --shard-size "$SHARD_SIZE" \
    --mode parallel \
    --only-modality vision \
    --condition-name validation_video_50 \
    --frame-drop 0.5 \
    --text-audio-device "$DEVICE" \
    --vision-device "$DEVICE" \
    --vision-workers "$WORKERS"
done

WHISPER_MANIFEST="$OUTPUT/manifests/validation-whisper.jsonl"
python3 -m bimer.cli asr-manifest \
  --manifest "$VALIDATION_MANIFEST" \
  --output "$WHISPER_MANIFEST" \
  --device cpu \
  --keep-original-on-error \
  --error-log "$OUTPUT/manifests/validation-whisper-errors.jsonl"

for dataset_split in "meld dev" "emotiontalk validation"; do
  read -r dataset split <<<"$dataset_split"
  python3 -m bimer.cli extract-features \
    --manifest "$WHISPER_MANIFEST" \
    --features "$OUTPUT/whisper" \
    --base-features "$BASE" \
    --yunet-model "$YUNET" \
    --dataset "$dataset" \
    --split "$split" \
    --shard-size "$SHARD_SIZE" \
    --mode parallel \
    --only-modality text \
    --condition-name validation_whisper \
    --text-audio-device "$DEVICE" \
    --vision-device "$DEVICE"
done

python3 - "$OUTPUT/validation-views.json" "$VALIDATION_MANIFEST" "$WHISPER_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
payload = {
    "source_split": {"meld": "dev", "emotiontalk": "validation"},
    "sample_identity": ["sample_id", "context_id", "label"],
    "views": {
        "audio_10db": {"corrupted_modality": "audio", "severity": 10.0},
        "video_50": {"corrupted_modality": "vision", "severity": 0.5},
        "whisper": {"corrupted_modality": "text", "severity": 1.0},
    },
    "clean_manifest": sys.argv[2],
    "whisper_manifest": sys.argv[3],
    "test_records_used": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

touch "$OUTPUT/VALIDATION_VIEWS_READY"
