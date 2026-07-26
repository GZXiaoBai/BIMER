#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Load it from Kaggle Secrets before running." >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
HF_DOWNLOAD_MAX_ATTEMPTS="${HF_DOWNLOAD_MAX_ATTEMPTS:-2}"
HF_DOWNLOAD_MAX_SECONDS="${HF_DOWNLOAD_MAX_SECONDS:-14400}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-32}"
HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
HF_DOWNLOAD_PROGRESS_SECONDS="${HF_DOWNLOAD_PROGRESS_SECONDS:-60}"
HF_DOWNLOAD_POLL_SECONDS="${HF_DOWNLOAD_POLL_SECONDS:-5}"
SKIP_EMOTIONTALK_DOWNLOAD_IF_EXTRACTED="${SKIP_EMOTIONTALK_DOWNLOAD_IF_EXTRACTED:-0}"
EMOTIONTALK_ARCHIVE_BYTES="${EMOTIONTALK_ARCHIVE_BYTES:-21294498304}"
HF_DOWNLOAD_REQUIRED_FREE_BYTES="${HF_DOWNLOAD_REQUIRED_FREE_BYTES:-22294498304}"
EMOTIONTALK_EXTRACT_REQUIRED_FREE_BYTES="${EMOTIONTALK_EXTRACT_REQUIRED_FREE_BYTES:-22294498304}"
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

check_disk_capacity() {
  local target="$1" required_bytes="$2" operation="$3" free_bytes
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  free_bytes="$(
    python3 - "$target" <<'PY'
from pathlib import Path
import shutil
import sys

print(shutil.disk_usage(Path(sys.argv[1])).free)
PY
  )"
  if ((free_bytes < required_bytes)); then
    echo "EmotionTalk $operation aborted: insufficient free disk space at $target; free_bytes=$free_bytes required_bytes=$required_bytes." >&2
    return 28
  fi
  echo "EmotionTalk $operation disk check passed: free_bytes=$free_bytes required_bytes=$required_bytes target=$target" >&2
}

partial_download_stats() {
  python3 - "$DOWNLOAD_ROOT" <<'PY'
from pathlib import Path
import sys

cache = Path(sys.argv[1]) / ".cache" / "huggingface" / "download"
stats = [path.stat() for path in cache.glob("*.incomplete") if path.is_file()]
logical = sum(stat.st_size for stat in stats)
allocated = sum(
    getattr(stat, "st_blocks", (stat.st_size + 511) // 512) * 512
    for stat in stats
)
print(logical, allocated)
PY
}

run_download_attempt() {
  local download_pid exit_code
  local last_bytes current_bytes delta_bytes elapsed_seconds bytes_per_second
  local last_allocated_bytes current_allocated_bytes delta_allocated_bytes
  local allocated_bytes_per_second
  local last_report_seconds next_report_seconds

  read -r last_bytes last_allocated_bytes < <(partial_download_stats)
  last_report_seconds="$SECONDS"
  next_report_seconds=$((SECONDS + HF_DOWNLOAD_PROGRESS_SECONDS))

  HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
    HF_HUB_ETAG_TIMEOUT="$HF_HUB_ETAG_TIMEOUT" \
    HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
    HF_XET_NUM_CONCURRENT_RANGE_GETS="$HF_XET_NUM_CONCURRENT_RANGE_GETS" \
    HF_XET_CHUNK_CACHE_SIZE_BYTES="$HF_XET_CHUNK_CACHE_SIZE_BYTES" \
    "$@" &
  download_pid=$!

  while kill -0 "$download_pid" 2>/dev/null; do
    sleep "$HF_DOWNLOAD_POLL_SECONDS"
    if ((SECONDS >= next_report_seconds)); then
      read -r current_bytes current_allocated_bytes < <(partial_download_stats)
      delta_bytes=$((current_bytes - last_bytes))
      delta_allocated_bytes=$((current_allocated_bytes - last_allocated_bytes))
      elapsed_seconds=$((SECONDS - last_report_seconds))
      if ((elapsed_seconds < 1)); then
        elapsed_seconds=1
      fi
      bytes_per_second=$((delta_bytes / elapsed_seconds))
      allocated_bytes_per_second=$((delta_allocated_bytes / elapsed_seconds))
      echo "EmotionTalk download progress: partial_bytes=$current_bytes allocated_bytes=$current_allocated_bytes delta_bytes=$delta_bytes delta_allocated_bytes=$delta_allocated_bytes bytes_per_second=$bytes_per_second allocated_bytes_per_second=$allocated_bytes_per_second xet_concurrency=$HF_XET_NUM_CONCURRENT_RANGE_GETS" >&2
      last_bytes="$current_bytes"
      last_allocated_bytes="$current_allocated_bytes"
      last_report_seconds="$SECONDS"
      next_report_seconds=$((SECONDS + HF_DOWNLOAD_PROGRESS_SECONDS))
    fi
  done

  if wait "$download_pid"; then
    return 0
  else
    exit_code=$?
    return "$exit_code"
  fi
}

download_emotiontalk_media() {
  local attempt exit_code
  local -a command=(
    timeout --signal=TERM --kill-after=30s "$HF_DOWNLOAD_MAX_SECONDS"
    hf download BAAI/Emotiontalk Multimodal.tar
    --repo-type dataset
    --local-dir "$DOWNLOAD_ROOT"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    run "${command[@]}"
    return
  fi

  for ((attempt = 1; attempt <= HF_DOWNLOAD_MAX_ATTEMPTS; attempt++)); do
    if run_download_attempt "${command[@]}"; then
      return
    else
      exit_code=$?
    fi
    if ((attempt < HF_DOWNLOAD_MAX_ATTEMPTS)); then
      echo "EmotionTalk download attempt $attempt/$HF_DOWNLOAD_MAX_ATTEMPTS failed (exit $exit_code); retrying with the cached partial file." >&2
    fi
  done

  echo "EmotionTalk download failed after $HF_DOWNLOAD_MAX_ATTEMPTS bounded attempts." >&2
  return "$exit_code"
}

run mkdir -p "$DOWNLOAD_ROOT" "$MEDIA_ROOT" "$(dirname "$OFFICIAL_REPO")" "$(dirname "$MANIFEST")"

if [[ "$DRY_RUN" == "1" || ( \
  ! -f "$DOWNLOAD_ROOT/Multimodal.tar" \
  && ! ( \
    "$SKIP_EMOTIONTALK_DOWNLOAD_IF_EXTRACTED" == "1" \
    && -f "$MEDIA_ROOT/.multimodal-extracted" \
  ) \
) ]]; then
  check_disk_capacity "$DOWNLOAD_ROOT" "$HF_DOWNLOAD_REQUIRED_FREE_BYTES" "download"
  download_emotiontalk_media
fi

if [[ "$DRY_RUN" == "1" || ! -f "$MEDIA_ROOT/.multimodal-extracted" ]]; then
  check_disk_capacity "$MEDIA_ROOT" "$EMOTIONTALK_EXTRACT_REQUIRED_FREE_BYTES" "extraction"
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
