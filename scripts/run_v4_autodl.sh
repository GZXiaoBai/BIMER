#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
export PATH="${BIMER_BIN_DIR:-/root/miniconda3/bin}:$PATH"
OUTPUT="${BIMER_OUTPUT:-$ROOT/artifacts/experiments/v4}"
LOGS="${BIMER_LOGS:-$OUTPUT/logs}"
ARCHIVE="${BIMER_ARCHIVE:-$ROOT/artifacts/bimer-v4-results.tar.gz}"
MANIFEST="${BIMER_MANIFEST:-$ROOT/data/processed/v2/all.jsonl}"
FEATURES="${BIMER_FEATURES:-$ROOT/artifacts/features/bilingual-v2-quality}"
ROBUSTNESS_FEATURES="${BIMER_ROBUSTNESS_FEATURES:-$ROOT/artifacts/features/v2-robustness}"
WHISPER_MANIFEST="${BIMER_WHISPER_MANIFEST:-$ROOT/data/processed/v2/whisper-test.jsonl}"
BASE_MODEL="${BIMER_BASE_MODEL:-xlm-roberta-base}"
STAGE="${BIMER_V4_STAGE:-screen}"
STAGE_MAX_SECONDS="${BIMER_STAGE_MAX_SECONDS:-28800}"
TOTAL_GPU_MAX_SECONDS="${BIMER_TOTAL_GPU_MAX_SECONDS:-72000}"
SCREEN_DECISION="$OUTPUT/screen-decision.json"
LORA_DECISION="$OUTPUT/lora-decision.json"
SELECTION="${BIMER_V4_SELECTION:-$ROOT/configs/experiment-v4-selection.json}"
FORMAL_SUMMARY="$OUTPUT/formal-summary.json"
BASELINE="$OUTPUT/screen/v2_no_language/quality_lagf/joint/seed-42/results.json"

mkdir -p "$LOGS" "$OUTPUT/_status"
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
  printf '{"exit_code":%d,"stage":"%s","elapsed_seconds":%d,"gpu_seconds_total":%d}\n' \
    "$status" "$STAGE" "$elapsed_seconds" "$((used_before + elapsed_seconds))" \
    > "$OUTPUT/_status/LAST_RUN.json"

  archive_inputs=(artifacts/experiments/v4)
  if [[ -f "$SELECTION" ]]; then
    archive_inputs+=("${SELECTION#"$ROOT"/}")
  fi
  tar \
    --exclude='artifacts/experiments/v4/lora/*/features' \
    --exclude='artifacts/experiments/v4/lora-robustness' \
    --exclude='artifacts/experiments/v4/lora/*/fusion/*/*/*/best.pt' \
    -C "$ROOT" -czf "$ARCHIVE" "${archive_inputs[@]}" \
    2>>"$LOGS/archive.log" || true
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

if (( remaining_budget <= 0 )); then
  echo "V4 cumulative 20-hour GPU budget is exhausted" >&2
  exit 124
fi
stage_limit="$STAGE_MAX_SECONDS"
if (( remaining_budget < stage_limit )); then
  stage_limit="$remaining_budget"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$STAGE" != "test" && ! -f "$OUTPUT/_status/OVERFIT_SMOKE_OK" ]]; then
  timeout --signal=TERM --kill-after=30 1800 \
    python3 -m bimer.cli overfit-smoke \
    --manifest "$MANIFEST" \
    --features "$FEATURES" \
    --dataset emotiontalk \
    --split train \
    --sample-count 16 \
    --output "$OUTPUT/_status/overfit-smoke.json" \
    --device cuda
  touch "$OUTPUT/_status/OVERFIT_SMOKE_OK"
fi

case "$STAGE" in
  screen)
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      python3 "$ROOT/scripts/run_v4_experiments.py" \
      --stage screen \
      --manifest "$MANIFEST" \
      --features "$FEATURES" \
      --output "$OUTPUT" \
      --device cuda 2>&1 | tee "$LOGS/screen.log"
    python3 "$ROOT/scripts/summarize_v4_screen.py" \
      --baseline "$BASELINE" \
      --candidate "context_only=$OUTPUT/screen/context_only/adaptive_context_prototype/joint/seed-42/results.json" \
      --candidate "prototype_only=$OUTPUT/screen/prototype_only/adaptive_context_prototype/joint/seed-42/results.json" \
      --candidate "combined_mu_050=$OUTPUT/screen/combined_mu_050/adaptive_context_prototype/joint/seed-42/results.json" \
      --candidate "combined_mu_100=$OUTPUT/screen/combined_mu_100/adaptive_context_prototype/joint/seed-42/results.json" \
      --candidate "combined_mu_200=$OUTPUT/screen/combined_mu_200/adaptive_context_prototype/joint/seed-42/results.json" \
      --output "$SCREEN_DECISION"
    if python3 -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["decision"]!="pass_v4a")' "$SCREEN_DECISION"; then
      python3 "$ROOT/scripts/freeze_v4_selection.py" \
        --decision "$SCREEN_DECISION" \
        --output "$SELECTION"
      touch "$OUTPUT/_status/V4A_PASSED"
    else
      touch "$OUTPUT/_status/LORA_REQUIRED"
    fi
    ;;
  lora)
    if [[ ! -f "$SCREEN_DECISION" ]]; then
      echo "V4-A screen decision is missing" >&2
      exit 2
    fi
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      python3 "$ROOT/scripts/run_v4_lora_fallback.py" \
      --decision "$SCREEN_DECISION" \
      --manifest "$MANIFEST" \
      --source-features "$FEATURES" \
      --base-model "$BASE_MODEL" \
      --output "$OUTPUT/lora" \
      --device cuda 2>&1 | tee "$LOGS/lora.log"
    python3 "$ROOT/scripts/summarize_v4_lora.py" \
      --baseline "$BASELINE" \
      --lora-root "$OUTPUT/lora" \
      --structure "$SCREEN_DECISION" \
      --output "$LORA_DECISION"
    if python3 -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["decision"]!="pass_lora")' "$LORA_DECISION"; then
      python3 "$ROOT/scripts/freeze_v4_selection.py" \
        --decision "$LORA_DECISION" \
        --output "$SELECTION"
      timeout --signal=TERM --kill-after=60 3600 \
        python3 "$ROOT/scripts/prepare_v4_lora_robustness_features.py" \
        --selection "$SELECTION" \
        --whisper-manifest "$WHISPER_MANIFEST" \
        --robustness-features "$ROBUSTNESS_FEATURES" \
        --output "$OUTPUT/lora-robustness" \
        --device cuda
      touch "$OUTPUT/_status/LORA_PASSED"
    else
      touch "$OUTPUT/_status/V4_STOPPED_VALIDATION_FAILURE"
    fi
    ;;
  formal)
    if [[ ! -f "$SELECTION" ]]; then
      echo "Frozen V4 selection is missing" >&2
      exit 2
    fi
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      python3 "$ROOT/scripts/run_v4_experiments.py" \
      --stage formal \
      --manifest "$MANIFEST" \
      --output "$OUTPUT" \
      --selection "$SELECTION" \
      --device cuda 2>&1 | tee "$LOGS/formal.log"
    python3 "$ROOT/scripts/summarize_v4_formal.py" \
      --formal-root "$OUTPUT/formal" \
      --baseline "$BASELINE" \
      --output "$FORMAL_SUMMARY" \
      --bootstrap-iterations 2000
    if python3 -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["formal_stable"])' "$FORMAL_SUMMARY"; then
      touch "$OUTPUT/_status/FORMAL_STABLE"
    else
      touch "$OUTPUT/_status/V4_STOPPED_FORMAL_INSTABILITY"
    fi
    ;;
  test)
    timeout --signal=TERM --kill-after=60 "$stage_limit" \
      python3 "$ROOT/scripts/run_v4_exploratory_test.py" \
      --selection "$SELECTION" \
      --formal-summary "$FORMAL_SUMMARY" \
      --formal-root "$OUTPUT/formal" \
      --manifest "$MANIFEST" \
      --features "$FEATURES" \
      --robustness-features "$ROBUSTNESS_FEATURES" \
      --whisper-manifest "$WHISPER_MANIFEST" \
      --lora-robustness-features "$OUTPUT/lora-robustness" \
      --output "$OUTPUT/exploratory-test" \
      --device cuda \
      --bootstrap-iterations 2000 2>&1 | tee "$LOGS/test.log"
    ;;
  *)
    echo "Unknown BIMER_V4_STAGE=$STAGE" >&2
    exit 2
    ;;
esac
