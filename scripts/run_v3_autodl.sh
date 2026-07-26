#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
OUTPUT="${BIMER_OUTPUT:-$ROOT/artifacts/experiments/v3}"
LOGS="${BIMER_LOGS:-$OUTPUT/logs}"
ARCHIVE="${BIMER_ARCHIVE:-$ROOT/artifacts/bimer-v3-results.tar.gz}"
PREPROCESS_MAX_SECONDS="${BIMER_PREPROCESS_MAX_SECONDS:-43200}"
TRAIN_MAX_SECONDS="${BIMER_TRAIN_MAX_SECONDS:-28800}"
TOTAL_GPU_MAX_SECONDS="${BIMER_TOTAL_GPU_MAX_SECONDS:-64800}"
STAGE="${BIMER_V3_STAGE:-preprocess}"

mkdir -p "$LOGS" "$OUTPUT/_status" "$ROOT/artifacts/features/v3-validation"
rm -f \
  "$OUTPUT/_status/STAGE_SUCCESS" \
  "$OUTPUT/_status/STAGE_FAILED" \
  "$OUTPUT/_status/DOWNLOAD_READY" \
  "$OUTPUT/_status/LAST_RUN.json"
started_epoch="$(date +%s)"
GPU_LEDGER="$OUTPUT/_status/GPU_SECONDS_USED"
used_before=0
if [[ -f "$GPU_LEDGER" ]]; then
  read -r used_before < "$GPU_LEDGER"
fi
remaining_budget="$((TOTAL_GPU_MAX_SECONDS - used_before))"

archive_and_shutdown() {
  status=$?
  trap - EXIT
  ended_epoch="$(date +%s)"
  elapsed_seconds="$((ended_epoch - started_epoch))"
  printf '%s\n' "$((used_before + elapsed_seconds))" > "$GPU_LEDGER.tmp"
  mv "$GPU_LEDGER.tmp" "$GPU_LEDGER"
  printf '{"exit_code":%d,"stage":"%s","elapsed_seconds":%d}\n' \
    "$status" "$STAGE" "$elapsed_seconds" \
    > "$OUTPUT/_status/LAST_RUN.json"
  tar -C "$ROOT" -czf "$ARCHIVE" \
    artifacts/experiments/v3 \
    artifacts/features/v3-validation 2>>"$LOGS/archive.log" || true
  if [[ -f "$ARCHIVE" ]]; then
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    touch "$OUTPUT/_status/DOWNLOAD_READY"
  fi
  if [[ "$status" -eq 0 ]]; then
    touch "$OUTPUT/_status/STAGE_SUCCESS"
  else
    printf '%s\n' "$status" > "$OUTPUT/_status/STAGE_FAILED"
  fi
  if [[ "${AUTODL_AUTO_SHUTDOWN:-1}" == "1" ]]; then
    shutdown -h now || sudo shutdown -h now || true
  fi
  exit "$status"
}
trap archive_and_shutdown EXIT

if (( remaining_budget <= 0 )); then
  echo "V3 cumulative GPU budget is exhausted" >&2
  exit 124
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$STAGE" != "preprocess" && "$STAGE" != "test" ]] \
  && [[ ! -f "$OUTPUT/_status/OVERFIT_SMOKE_OK" ]]; then
  timeout --signal=TERM --kill-after=30 1800 \
    python3 -m bimer.cli overfit-smoke \
    --manifest "$ROOT/data/processed/v2/all.jsonl" \
    --features "$ROOT/artifacts/features/bilingual-v2-quality" \
    --dataset emotiontalk \
    --split train \
    --sample-count 16 \
    --output "$OUTPUT/_status/overfit-smoke.json" \
    --device cuda
  touch "$OUTPUT/_status/OVERFIT_SMOKE_OK"
fi

case "$STAGE" in
  preprocess)
    stage_limit="$PREPROCESS_MAX_SECONDS"
    if (( remaining_budget < stage_limit )); then stage_limit="$remaining_budget"; fi
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      bash "$ROOT/scripts/prepare_v3_validation_views.sh" \
      2>&1 | tee "$LOGS/preprocess.log"
    ;;
  loss-screen|ranking-screen|formal|test)
    stage_limit="$TRAIN_MAX_SECONDS"
    if (( remaining_budget < stage_limit )); then stage_limit="$remaining_budget"; fi
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      python3 "$ROOT/scripts/run_v3_experiments.py" \
      --stage "$STAGE" "$@" 2>&1 | tee "$LOGS/$STAGE.log"
    ;;
  *)
    echo "Unknown BIMER_V3_STAGE=$STAGE" >&2
    exit 2
    ;;
esac
