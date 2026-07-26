#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
OUTPUT="${BIMER_OUTPUT:-$ROOT/artifacts/experiments/v2}"
ARCHIVE="${BIMER_ARCHIVE:-$ROOT/artifacts/bimer-v2-formal-ablations.tar.gz}"
CONFIG="${BIMER_CONFIG:-$ROOT/configs/experiment-v2.toml}"
MANIFEST="${BIMER_MANIFEST:-$ROOT/data/processed/v2/all.jsonl}"
QUALITY_FEATURES="${BIMER_QUALITY_FEATURES:-$ROOT/artifacts/features/bilingual-v2-quality}"
CORRUPTION_MANIFEST="${BIMER_CORRUPTION_MANIFEST:-$ROOT/data/processed/v2/corruption-train-10pct-joint.jsonl}"
ASR_MANIFEST="${BIMER_ASR_MANIFEST:-$ROOT/data/processed/v2/corruption-train-10pct-joint-asr.jsonl}"
AUDIO10_FEATURES="${BIMER_AUDIO10_FEATURES:-$ROOT/artifacts/features/v2-corruption-joint-audio10}"
VIDEO50_FEATURES="${BIMER_VIDEO50_FEATURES:-$ROOT/artifacts/features/v2-corruption-joint-video50}"
WHISPER_FEATURES="${BIMER_WHISPER_FEATURES:-$ROOT/artifacts/features/v2-corruption-joint-whisper}"

export OMP_NUM_THREADS="${BIMER_CPU_THREADS:-1}"
export MKL_NUM_THREADS="${BIMER_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BIMER_CPU_THREADS:-1}"
export PYTHONUNBUFFERED=1

if [[ -n "${BIMER_PYTHON:-}" ]]; then
  PYTHON_BIN="$BIMER_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif [[ -x /root/miniconda3/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/bin/python
else
  PYTHON_BIN="$(command -v python)"
fi

mkdir -p "$OUTPUT/_status" "$(dirname "$ARCHIVE")"

on_exit() {
  run_status=$?
  archive_status=0
  trap - EXIT

  if [[ "$run_status" -eq 0 ]]; then
    date -u "+%Y-%m-%dT%H:%M:%SZ" > "$OUTPUT/_status/RUN_COMPLETE"
    rm -f "$OUTPUT/_status/RUN_FAILED"
  else
    printf '%s\n' "$run_status" > "$OUTPUT/_status/RUN_FAILED"
  fi

  tar -C "$ROOT" -czf "$ARCHIVE" \
    artifacts/experiments/v2 \
    configs/experiment-v2.toml \
    configs/experiment-v2-selection.json || archive_status=$?
  if [[ "$archive_status" -eq 0 ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    else
      shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
    fi
    touch "$OUTPUT/_status/DOWNLOAD_READY"
  else
    printf '%s\n' "$archive_status" > "$OUTPUT/_status/ARCHIVE_FAILED"
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

common_args=(
  --config "$CONFIG"
  --manifest "$MANIFEST"
  --quality-features "$QUALITY_FEATURES"
  --output "$OUTPUT"
  --device cuda
  --augmentation-manifest "$CORRUPTION_MANIFEST"
  --augmentation-features "$AUDIO10_FEATURES"
  --augmentation-manifest "$CORRUPTION_MANIFEST"
  --augmentation-features "$VIDEO50_FEATURES"
  --augmentation-manifest "$ASR_MANIFEST"
  --augmentation-features "$WHISPER_FEATURES"
)

cd "$ROOT"
"$PYTHON_BIN" scripts/run_v2_experiments.py --stage formal "${common_args[@]}"
"$PYTHON_BIN" scripts/run_v2_experiments.py --stage ablations "${common_args[@]}"
