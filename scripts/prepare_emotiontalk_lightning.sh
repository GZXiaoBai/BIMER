#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it inside Lightning Studio first." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTNING_STUDIO_ROOT="${LIGHTNING_STUDIO_ROOT:-/teamspace/studios/this_studio/bimer}"
DRY_RUN="${DRY_RUN:-0}"

export RAW_ROOT="${RAW_ROOT:-$LIGHTNING_STUDIO_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LIGHTNING_STUDIO_ROOT/output}"
export SKIP_EMOTIONTALK_DOWNLOAD_IF_EXTRACTED=1
export HF_DOWNLOAD_MAX_ATTEMPTS="${HF_DOWNLOAD_MAX_ATTEMPTS:-4}"
export HF_DOWNLOAD_MAX_SECONDS="${HF_DOWNLOAD_MAX_SECONDS:-21600}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-16}"

bash "$SCRIPT_DIR/prepare_emotiontalk_kaggle.sh"

archive="$RAW_ROOT/downloads/emotiontalk/Multimodal.tar"
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY-RUN rm -f %q\n' "$archive"
else
  rm -f "$archive"
fi

echo "Lightning EmotionTalk data ready: $OUTPUT_ROOT/emotiontalk.jsonl"
