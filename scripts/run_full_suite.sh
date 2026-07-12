#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the combined JSONL manifest}"
: "${FEATURES:?Set FEATURES to the standard feature cache root}"
: "${OUTPUT:?Set OUTPUT to the experiment output directory}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
SEEDS=(42 123 2026)
MODELS=(majority text audio vision early_mlp early_context lagf)

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    "$PYTHON" -m bimer.cli train \
      --manifest "$MANIFEST" --features "$FEATURES" --output "$OUTPUT" \
      --model "$model" --training-scope joint --seed "$seed" --device "$DEVICE"
  done
done

for scope in meld emotiontalk; do
  for seed in "${SEEDS[@]}"; do
    "$PYTHON" -m bimer.cli train \
      --manifest "$MANIFEST" --features "$FEATURES" --output "$OUTPUT" \
      --model lagf --training-scope "$scope" --seed "$seed" --device "$DEVICE"
  done
done

for ablation in no-language no-gates no-context no-modality-dropout; do
  case "$ablation" in
    no-language) flag="--no-language" ;;
    no-gates) flag="--no-gates" ;;
    no-context) flag="--no-context" ;;
    no-modality-dropout) flag="--no-modality-dropout" ;;
  esac
  for seed in "${SEEDS[@]}"; do
    "$PYTHON" -m bimer.cli train \
      --manifest "$MANIFEST" --features "$FEATURES" \
      --output "$OUTPUT/ablations/$ablation" \
      --model lagf --training-scope joint --seed "$seed" --device "$DEVICE" \
      "$flag"
  done
done

CHECKPOINT="$OUTPUT/lagf/joint/seed-42/best.pt"
for modality in text audio vision; do
  "$PYTHON" -m bimer.cli evaluate \
    --manifest "$MANIFEST" --features "$FEATURES" --checkpoint "$CHECKPOINT" \
    --missing "$modality" --device "$DEVICE" \
    --output "$OUTPUT/robustness/missing-$modality.json"
done

echo "Core experiment matrix complete: $OUTPUT"
echo "For raw-input noise, frame-drop and ASR conditions, follow docs/experiment_protocol.md."
