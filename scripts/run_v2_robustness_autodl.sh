#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-/root/autodl-tmp/bimer-runtime}"
OUTPUT="${BIMER_OUTPUT:-$ROOT/artifacts/experiments/v2/robustness}"
ARCHIVE="${BIMER_ARCHIVE:-$ROOT/artifacts/bimer-v2-robustness.tar.gz}"
BASE_FEATURES="${BASE_FEATURES:-$ROOT/artifacts/features/bilingual-v2-quality}"
ROBUSTNESS_FEATURES="${ROBUSTNESS_ROOT:-$ROOT/artifacts/features/v2-robustness}"
MANIFEST="${BIMER_MANIFEST:-$ROOT/data/processed/v2/all.jsonl}"
WHISPER_MANIFEST="${WHISPER_MANIFEST:-$ROOT/data/processed/v2/whisper-test.jsonl}"
EMOTIONTALK_RAW="${EMOTIONTALK_RAW_ROOT:-$RUNTIME_ROOT/data/raw/emotiontalk}"
STATUS="$OUTPUT/_status"

if [[ -n "${BIMER_PYTHON:-}" ]]; then
  PYTHON_BIN="$BIMER_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="$(command -v python)"
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${BIMER_CPU_THREADS:-1}"
export MKL_NUM_THREADS="${BIMER_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BIMER_CPU_THREADS:-1}"

mkdir -p "$STATUS" "$(dirname "$ARCHIVE")"
rm -f \
  "$STATUS/RUN_COMPLETE" \
  "$STATUS/RUN_FAILED" \
  "$STATUS/DOWNLOAD_READY" \
  "$STATUS/ARCHIVE_FAILED"

# The portable manifest keeps the original /tmp/bimer-data media paths.
# AutoDL clears /tmp on every restart, so restore the ephemeral link before
# any corruption extractor tries to decode EmotionTalk clips.
if [[ -d "$EMOTIONTALK_RAW" ]]; then
  mkdir -p /tmp/bimer-data/raw
  ln -sfn "$EMOTIONTALK_RAW" /tmp/bimer-data/raw/emotiontalk
fi

on_exit() {
  run_status=$?
  archive_status=0
  trap - EXIT

  if [[ "$run_status" -eq 0 ]]; then
    date -u "+%Y-%m-%dT%H:%M:%SZ" > "$STATUS/RUN_COMPLETE"
    rm -f "$STATUS/RUN_FAILED"
  else
    printf '%s\n' "$run_status" > "$STATUS/RUN_FAILED"
  fi

  tar -C "$ROOT" -czf "$ARCHIVE" \
    artifacts/experiments/v2/robustness \
    configs/experiment-v2-selection.json || archive_status=$?
  if [[ "$archive_status" -eq 0 ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    else
      shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
    fi
    touch "$STATUS/DOWNLOAD_READY"
  else
    printf '%s\n' "$archive_status" > "$STATUS/ARCHIVE_FAILED"
  fi

  if [[ "${AUTODL_AUTO_SHUTDOWN:-0}" == "1" ]]; then
    shutdown -h now || sudo shutdown -h now || true
  fi
  if [[ "$run_status" -ne 0 ]]; then
    exit "$run_status"
  fi
  exit "$archive_status"
}
trap on_exit EXIT

common_environment=(
  AUTODL_RUNTIME_ROOT="$RUNTIME_ROOT"
  MANIFEST="$MANIFEST"
  BASE_FEATURES="$BASE_FEATURES"
  ROBUSTNESS_ROOT="$ROBUSTNESS_FEATURES"
)

env "${common_environment[@]}" \
  bash "$ROOT/scripts/run_autodl_audio_robustness.sh"
env "${common_environment[@]}" \
  bash "$ROOT/scripts/run_autodl_video_robustness.sh"
env "${common_environment[@]}" \
  ASR_MANIFEST="$WHISPER_MANIFEST" \
  bash "$ROOT/scripts/run_autodl_whisper_robustness.sh"

"$PYTHON_BIN" "$ROOT/scripts/run_v2_robustness.py" \
  --root "$ROOT" \
  --runtime-root "$RUNTIME_ROOT" \
  --manifest "$MANIFEST" \
  --whisper-manifest "$WHISPER_MANIFEST" \
  --base-features "$BASE_FEATURES" \
  --robustness-features "$ROBUSTNESS_FEATURES" \
  --output "$OUTPUT" \
  --device cuda

echo "V2_ROBUSTNESS_COMPLETE"
