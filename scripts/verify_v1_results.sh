#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/artifacts/registry/v1-preaudit.sha256"
TEMP="$(mktemp)"
trap 'rm -f "$TEMP"' EXIT

awk '$2 ~ /^artifacts\/experiments\// && ($2 ~ /results\.json$/ || $2 ~ /summary\.json$/) {print}' \
  "$MANIFEST" > "$TEMP"
cd "$ROOT"
shasum -a 256 -c "$TEMP"

results="$(awk '$2 ~ /results\.json$/ {count++} END {print count+0}' "$TEMP")"
summaries="$(awk '$2 ~ /summary\.json$/ {count++} END {print count+0}' "$TEMP")"
if [[ "$results" -ne 103 || "$summaries" -ne 21 ]]; then
  echo "unexpected v1 registry counts: results=$results summaries=$summaries" >&2
  exit 1
fi
echo "v1-preaudit verified: 103 results, 21 summaries"
