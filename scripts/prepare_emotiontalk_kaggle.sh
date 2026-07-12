#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Load it from Kaggle Secrets before running." >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
RAW_ROOT="${RAW_ROOT:-/tmp/bimer-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/bimer-output}"
DOWNLOAD_ROOT="$RAW_ROOT/downloads/emotiontalk"
MEDIA_ROOT="$RAW_ROOT/raw/emotiontalk"
OFFICIAL_REPO="$RAW_ROOT/sources/emotiontalk-official"
MANIFEST="$OUTPUT_ROOT/emotiontalk.jsonl"
OFFICIAL_COMMIT="cb8397e226ce7c1fccee41ea21161a3f98f578e1"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run mkdir -p "$DOWNLOAD_ROOT" "$MEDIA_ROOT" "$(dirname "$OFFICIAL_REPO")" "$(dirname "$MANIFEST")"

if [[ "$DRY_RUN" == "1" || ! -f "$DOWNLOAD_ROOT/Multimodal.tar" ]]; then
  run hf download BAAI/Emotiontalk Multimodal.tar \
    --repo-type dataset \
    --local-dir "$DOWNLOAD_ROOT"
fi

if [[ "$DRY_RUN" == "1" || ! -f "$MEDIA_ROOT/.multimodal-extracted" ]]; then
  run tar -xf "$DOWNLOAD_ROOT/Multimodal.tar" -C "$MEDIA_ROOT"
  run touch "$MEDIA_ROOT/.multimodal-extracted"
fi

if [[ "$DRY_RUN" == "1" || ! -d "$OFFICIAL_REPO/.git" ]]; then
  run git clone https://github.com/NKU-HLT/EmotionTalk.git "$OFFICIAL_REPO"
fi
run git -C "$OFFICIAL_REPO" checkout --detach "$OFFICIAL_COMMIT"

ANNOTATIONS="$OFFICIAL_REPO/EmotionTalk/dataset/mm-process"
run bimer prepare-emotiontalk-official \
  --labels-csv "$ANNOTATIONS/mm.csv" \
  --transcriptions-csv "$ANNOTATIONS/transcription.csv" \
  --media-root "$MEDIA_ROOT" \
  --output "$MANIFEST"
run bimer validate --manifest "$MANIFEST" --official-counts

echo "EmotionTalk manifest ready: $MANIFEST"
