#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-${LIGHTNING_STUDIO_ROOT:-/root/autodl-tmp/bimer-runtime}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUNTIME_ROOT/checkpoints/joint-full}"
ROBUSTNESS_ROOT="${ROBUSTNESS_ROOT:-$RUNTIME_ROOT/robustness}"
REPORT_ROOT="${REPORT_ROOT:-$RUNTIME_ROOT/reports/robustness-evaluation}"
STANDARD_MANIFEST="${MANIFEST:-$RUNTIME_ROOT/output/bilingual.jsonl}"
WHISPER_MANIFEST="${WHISPER_MANIFEST:-$RUNTIME_ROOT/output/whisper-test.jsonl}"
DRY_RUN="${DRY_RUN:-0}"

conditions=(
  audio_snr_20db
  audio_snr_10db
  video_frame_drop_25pct
  video_frame_drop_50pct
  whisper_text
)
seeds=(42 123 2026)

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
    echo "AUTODL_ROBUSTNESS_EVALUATION_FAILED status=$status" >&2
  fi
}
trap report_failure EXIT

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f "$STANDARD_MANIFEST" ]] || {
    echo "Standard manifest is missing: $STANDARD_MANIFEST" >&2
    exit 20
  }
  [[ -f "$WHISPER_MANIFEST" ]] || {
    echo "Whisper manifest is missing: $WHISPER_MANIFEST" >&2
    exit 21
  }
  if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "A CUDA GPU is required for robustness evaluation" >&2
    exit 22
  fi
fi

run mkdir -p "$REPORT_ROOT"

for condition in "${conditions[@]}"; do
  feature_root="$ROBUSTNESS_ROOT/$condition"
  manifest="$STANDARD_MANIFEST"
  if [[ "$condition" == "whisper_text" ]]; then
    manifest="$WHISPER_MANIFEST"
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -d "$feature_root/meld/test" ]] || {
      echo "MELD features are missing for $condition" >&2
      exit 23
    }
    [[ -d "$feature_root/emotiontalk/test" ]] || {
      echo "EmotionTalk features are missing for $condition" >&2
      exit 24
    }
  fi

  result_paths=()
  for seed in "${seeds[@]}"; do
    checkpoint="$CHECKPOINT_ROOT/seed-$seed/best.pt"
    output="$REPORT_ROOT/$condition/seed-$seed.json"
    result_paths+=("$output")
    if [[ "$DRY_RUN" != "1" && -f "$output" ]]; then
      echo "SKIP $condition seed=$seed existing result"
      continue
    fi
    if [[ "$DRY_RUN" != "1" && ! -f "$checkpoint" ]]; then
      echo "Checkpoint is missing: $checkpoint" >&2
      exit 25
    fi
    run python -m bimer.cli evaluate \
      --manifest "$manifest" \
      --features "$feature_root" \
      --checkpoint "$checkpoint" \
      --output "$output" \
      --condition-name "$condition" \
      --device cuda
  done

  run python -c \
    'import sys; from bimer.experiment import aggregate_seed_results; aggregate_seed_results(sys.argv[2:], sys.argv[1])' \
    "$REPORT_ROOT/$condition/summary.json" \
    "${result_paths[@]}"
done

run touch "$REPORT_ROOT/AUTODL_ROBUSTNESS_EVALUATION_COMPLETE"
echo "AUTODL_ROBUSTNESS_EVALUATION_COMPLETE"
