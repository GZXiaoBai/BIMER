#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BIMER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT="${BIMER_M2_OUTPUT:-$ROOT/artifacts/acceptance/m2}"
TIME_LOG="$OUTPUT/time.log"
mkdir -p "$OUTPUT"

swap_before="$(sysctl -n vm.swapusage 2>/dev/null || true)"
/usr/bin/time -l \
  python3 "$ROOT/scripts/m2_acceptance.py" "$@" --output "$OUTPUT" \
  2> "$TIME_LOG"
swap_after="$(sysctl -n vm.swapusage 2>/dev/null || true)"

peak_bytes="$(awk '/maximum resident set size/ {print $1}' "$TIME_LOG" | tail -1)"
if [[ -z "$peak_bytes" ]]; then
  echo "Unable to read peak resident memory from /usr/bin/time -l" >&2
  exit 1
fi
if (( peak_bytes > 6979321856 )); then
  echo "Peak resident memory exceeds 6.5 GiB: $peak_bytes" >&2
  exit 1
fi
if [[ "$swap_before" != "$swap_after" ]]; then
  echo "Swap usage changed during acceptance run" >&2
  exit 1
fi

python3 - "$OUTPUT/m2-resource-report.json" "$peak_bytes" "$swap_before" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "peak_resident_bytes": int(sys.argv[2]),
    "peak_limit_bytes": 6979321856,
    "swap_before_and_after": sys.argv[3],
    "swap_unchanged": True,
}, ensure_ascii=False, indent=2) + "\n")
PY
