#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
export PATH="${BIMER_BIN_DIR:-/root/miniconda3/bin}:$PATH"
OUTPUT="${BIMER_V5_OUTPUT:-$ROOT/artifacts/experiments/v5}"
LOGS="${BIMER_V5_LOGS:-$OUTPUT/logs}"
ARCHIVE="${BIMER_V5_ARCHIVE:-$ROOT/artifacts/bimer-v5-results.tar.gz}"
STAGE="${BIMER_V5_STAGE:-screen}"
SELECTION="${BIMER_V5_SELECTION:-$ROOT/configs/experiment-v5-selection.json}"
DECISION="$OUTPUT/screen-decision.json"
BASELINE="${BIMER_V5_BASELINE:-$ROOT/artifacts/experiments/v3/screen/loss/weighted_ce/quality_lagf/joint/seed-42/results.json}"
TOTAL_GPU_MAX_SECONDS="${BIMER_TOTAL_GPU_MAX_SECONDS:-36000}"
STAGE_MAX_SECONDS="${BIMER_STAGE_MAX_SECONDS:-28800}"

mkdir -p "$LOGS" "$OUTPUT/_status"
rm -f \
  "$OUTPUT/_status/STAGE_SUCCESS" \
  "$OUTPUT/_status/STAGE_FAILED" \
  "$OUTPUT/_status/DOWNLOAD_READY" \
  "$OUTPUT/_status/LAST_RUN.json"

started_epoch="$(date +%s)"
ledger="$OUTPUT/_status/GPU_SECONDS_USED"
used_before=0
if [[ -f "$ledger" ]]; then
  read -r used_before < "$ledger"
fi
remaining="$((TOTAL_GPU_MAX_SECONDS - used_before))"

archive_and_shutdown() {
  status=$?
  trap - EXIT
  ended_epoch="$(date +%s)"
  elapsed="$((ended_epoch - started_epoch))"
  total="$((used_before + elapsed))"
  printf '%s\n' "$total" > "$ledger.tmp"
  mv "$ledger.tmp" "$ledger"
  printf '{"exit_code":%d,"stage":"%s","elapsed_seconds":%d,"gpu_seconds_total":%d}\n' \
    "$status" "$STAGE" "$elapsed" "$total" > "$OUTPUT/_status/LAST_RUN.json"

  archive_inputs=(artifacts/experiments/v5)
  if [[ -f "$SELECTION" ]]; then
    archive_inputs+=("${SELECTION#"$ROOT"/}")
  fi
  tar -C "$ROOT" -czf "$ARCHIVE" "${archive_inputs[@]}" 2>>"$LOGS/archive.log" || true
  if [[ -f "$ARCHIVE" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    else
      shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
    fi
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

if (( remaining <= 0 )); then
  echo "V5 cumulative 10-hour GPU budget is exhausted" >&2
  exit 124
fi
stage_limit="$STAGE_MAX_SECONDS"
if (( remaining < stage_limit )); then
  stage_limit="$remaining"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

case "$STAGE" in
  screen)
    if [[ ! -f "$BASELINE" ]]; then
      echo "Frozen V2 validation baseline is missing: $BASELINE" >&2
      exit 2
    fi
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      python3 "$ROOT/scripts/run_v5_experiments.py" \
      --stage screen 2>&1 | tee "$LOGS/screen.log"
    python3 "$ROOT/scripts/summarize_v5_screen.py" \
      --baseline "$BASELINE" \
      --candidate "beta_005=0.05=$OUTPUT/screen/beta_005/asr_consistent_quality_lagf/joint/seed-42/results.json" \
      --candidate "beta_010=0.10=$OUTPUT/screen/beta_010/asr_consistent_quality_lagf/joint/seed-42/results.json" \
      --output "$DECISION"
    if python3 -c \
      'import json,sys;sys.exit(json.load(open(sys.argv[1]))["decision"]!="pass_v5")' \
      "$DECISION"; then
      python3 "$ROOT/scripts/freeze_v5_selection.py" \
        --decision "$DECISION" \
        --output "$SELECTION"
      touch "$OUTPUT/_status/V5_SCREEN_PASSED"
    else
      touch "$OUTPUT/_status/V5_STOPPED_VALIDATION_FAILURE"
    fi
    ;;
  formal)
    if [[ ! -f "$SELECTION" ]]; then
      echo "Frozen V5 selection is missing: $SELECTION" >&2
      exit 2
    fi
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      python3 "$ROOT/scripts/run_v5_experiments.py" \
      --stage formal \
      --selection "$SELECTION" 2>&1 | tee "$LOGS/formal.log"
    touch "$OUTPUT/_status/V5_FORMAL_COMPLETE"
    ;;
  *)
    echo "Unknown BIMER_V5_STAGE=$STAGE" >&2
    exit 2
    ;;
esac
