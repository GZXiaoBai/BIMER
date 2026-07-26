#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MANIFEST="${BIMER_MANIFEST:-$ROOT/data/processed/v2/all.jsonl}"
BASE="${BIMER_BASE_FEATURES:-$ROOT/artifacts/features/bilingual-v1}"
QUALITY="${BIMER_QUALITY_FEATURES:-$ROOT/artifacts/features/bilingual-v2-quality}"
YUNET="${BIMER_YUNET_MODEL:-$ROOT/artifacts/models/face_detection_yunet_2023mar.onnx}"
SELECTED_MANIFEST="$ROOT/data/processed/v2/corruption-train-10pct.jsonl"
SELECTED_ASR_MANIFEST="$ROOT/data/processed/v2/corruption-train-10pct-asr.jsonl"
SELECTED_CLEAN="$ROOT/artifacts/features/v2-corruption-clean"
WORKERS="${BIMER_QUALITY_WORKERS:-4}"

cd "$ROOT"
for group in "meld train" "meld dev" "meld test" "emotiontalk train" "emotiontalk validation" "emotiontalk test"; do
  read -r dataset split <<< "$group"
  python3 -m bimer.cli attach-quality \
    --manifest "$MANIFEST" \
    --base-features "$BASE" \
    --output-features "$QUALITY" \
    --yunet-model "$YUNET" \
    --dataset "$dataset" \
    --split "$split" \
    --workers "$WORKERS"
done

python3 -m bimer.cli sample-corruption-manifest \
  --manifest "$MANIFEST" \
  --output-manifest "$SELECTED_MANIFEST" \
  --base-features "$QUALITY" \
  --output-features "$SELECTED_CLEAN" \
  --fraction 0.1 \
  --seed 42

for dataset in meld emotiontalk; do
  python3 -m bimer.cli extract-features \
    --manifest "$SELECTED_MANIFEST" \
    --dataset "$dataset" \
    --split train \
    --base-features "$SELECTED_CLEAN" \
    --features "$ROOT/artifacts/features/v2-corruption-audio10" \
    --yunet-model "$YUNET" \
    --mode parallel \
    --only-modality audio \
    --condition-name train-audio-10db \
    --audio-snr 10 \
    --text-audio-device cuda:0 \
    --vision-device cuda:0

  python3 -m bimer.cli extract-features \
    --manifest "$SELECTED_MANIFEST" \
    --dataset "$dataset" \
    --split train \
    --base-features "$SELECTED_CLEAN" \
    --features "$ROOT/artifacts/features/v2-corruption-video50" \
    --yunet-model "$YUNET" \
    --mode parallel \
    --only-modality vision \
    --condition-name train-video-drop-50 \
    --frame-drop 0.5 \
    --text-audio-device cuda:0 \
    --vision-device cuda:0
done

python3 -m bimer.cli asr-manifest \
  --manifest "$SELECTED_MANIFEST" \
  --output "$SELECTED_ASR_MANIFEST" \
  --device cuda \
  --keep-original-on-error \
  --error-log "$ROOT/artifacts/features/v2-corruption-whisper/asr-errors.jsonl"

for dataset in meld emotiontalk; do
  python3 -m bimer.cli extract-features \
    --manifest "$SELECTED_ASR_MANIFEST" \
    --dataset "$dataset" \
    --split train \
    --base-features "$SELECTED_CLEAN" \
    --features "$ROOT/artifacts/features/v2-corruption-whisper" \
    --yunet-model "$YUNET" \
    --mode parallel \
    --only-modality text \
    --condition-name train-whisper-text \
    --text-audio-device cuda:0 \
    --vision-device cuda:0
done

touch "$ROOT/artifacts/features/v2-quality-views.DONE"
