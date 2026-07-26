#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-/root/autodl-tmp/bimer}"
OUTPUT="${BIMER_OUTPUT:-$ROOT/artifacts/experiments/v2}"
ARCHIVE="${BIMER_ARCHIVE:-$ROOT/artifacts/bimer-v2-results.tar.gz}"

on_exit() {
  status=$?
  trap - EXIT
  mkdir -p "$OUTPUT/_status"
  if [[ "$status" -eq 0 ]]; then
    tar -C "$ROOT" -czf "$ARCHIVE" artifacts/experiments/v2
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    touch "$OUTPUT/_status/DOWNLOAD_READY"
  else
    printf '%s\n' "$status" > "$OUTPUT/_status/RUN_FAILED"
  fi
  if [[ "${AUTODL_AUTO_SHUTDOWN:-0}" == "1" ]]; then
    shutdown -h now || sudo shutdown -h now || true
  fi
  exit "$status"
}
trap on_exit EXIT

cd "$ROOT"
python3 scripts/run_v2_experiments.py "$@"
