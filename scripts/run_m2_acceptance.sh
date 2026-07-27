#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT="${BIMER_M2_OUTPUT:-$ROOT/artifacts/acceptance/m2}"
PYTHON="${BIMER_PYTHON:-$ROOT/.venv/bin/python}"
TIME_LOG="$OUTPUT/time.log"
mkdir -p "$OUTPUT"

swap_before="$(sysctl -n vm.swapusage 2>/dev/null || true)"
/usr/bin/time -l \
  "$PYTHON" "$ROOT/scripts/m2_acceptance.py" "$@" --output "$OUTPUT" \
  2> "$TIME_LOG"
swap_after="$(sysctl -n vm.swapusage 2>/dev/null || true)"

peak_bytes="$(awk '/peak memory footprint/ {print $1}' "$TIME_LOG" | tail -1)"
if [[ -z "$peak_bytes" ]]; then
  peak_bytes="$(awk '/maximum resident set size/ {print $1}' "$TIME_LOG" | tail -1)"
fi
if [[ -z "$peak_bytes" ]]; then
  echo "Unable to read peak memory from /usr/bin/time -l" >&2
  exit 1
fi
if (( peak_bytes > 6979321856 )); then
  echo "Peak memory footprint exceeds 6.5 GiB: $peak_bytes" >&2
  exit 1
fi
process_swaps="$(awk '$2 == "swaps" {print $1}' "$TIME_LOG" | tail -1)"
if [[ -z "$process_swaps" ]]; then
  echo "Unable to read process swap count from /usr/bin/time -l" >&2
  exit 1
fi
if (( process_swaps != 0 )); then
  echo "Acceptance process swapped: $process_swaps" >&2
  exit 1
fi

"$PYTHON" - \
  "$OUTPUT/m2-resource-report.json" \
  "$peak_bytes" \
  "$process_swaps" \
  "$swap_before" \
  "$swap_after" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "peak_memory_footprint_bytes": int(sys.argv[2]),
    "peak_limit_bytes": 6979321856,
    "process_swaps": int(sys.argv[3]),
    "process_swaps_zero": int(sys.argv[3]) == 0,
    "system_swap_before": sys.argv[4],
    "system_swap_after": sys.argv[5],
}, ensure_ascii=False, indent=2) + "\n")
PY
