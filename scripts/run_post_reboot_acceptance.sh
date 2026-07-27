#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT="${BIMER_POST_REBOOT_OUTPUT:-$ROOT/artifacts/acceptance/m2-post-reboot}"
PYTHON="${BIMER_PYTHON:-$ROOT/.venv/bin/python}"

export BIMER_M2_OUTPUT="$OUTPUT"
bash "$ROOT/scripts/run_m2_acceptance.sh" \
  --deployment "$ROOT/configs/deployment-v2.json" \
  --artifact-root "$ROOT" \
  --chinese-video "$ROOT/artifacts/demo/zh-face-cao-dewang-voa-50s.mp4" \
  --english-no-face-video "$ROOT/artifacts/demo/en-noface.mp4" \
  "$@"

"$PYTHON" "$ROOT/scripts/validate_system_swap.py" \
  --resource-report "$OUTPUT/m2-resource-report.json" \
  --output "$OUTPUT/system-swap-acceptance.json"
