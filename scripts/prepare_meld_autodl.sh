#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${AUTODL_RUNTIME_ROOT:-${LIGHTNING_STUDIO_ROOT:-/root/autodl-tmp/bimer-runtime}}"
DOWNLOAD_ROOT="${MELD_DOWNLOAD_ROOT:-$RUNTIME_ROOT/downloads/meld}"
RAW_ROOT="${MELD_RAW_ROOT:-$RUNTIME_ROOT/raw/meld}"
MEDIA_ROOT="${MELD_MEDIA_ROOT:-$RAW_ROOT/media}"
ANNOTATION_ROOT="${MELD_ANNOTATION_ROOT:-$RAW_ROOT/annotations}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUNTIME_ROOT/output}"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/meld.jsonl}"
ARCHIVE="$DOWNLOAD_ROOT/MELD.Raw.tar.gz"
EXTRACTED_MARKER="$RAW_ROOT/.meld-raw-extracted"
DRY_RUN="${DRY_RUN:-0}"
MELD_REQUIRED_FREE_BYTES="${MELD_REQUIRED_FREE_BYTES:-36000000000}"
NETWORK_TURBO_PATH="${NETWORK_TURBO_PATH:-/etc/network_turbo}"
MELD_ARCHIVE_BYTES=10878146150
MELD_ARCHIVE_SHA256=a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e
MELD_PARALLEL_URL="${MELD_PARALLEL_URL:-https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz}"
MELD_DOWNLOADER="${MELD_DOWNLOADER:-auto}"
MELD_ANNOTATION_COMMIT=e8cedf27b5d2877e198332c957127e16eb214afe
MELD_ANNOTATION_BASE="https://raw.githubusercontent.com/declare-lab/MELD/$MELD_ANNOTATION_COMMIT/data/MELD"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN'
    LC_ALL=C printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

report_failure() {
  local status=$?
  if ((status != 0)); then
    echo "AUTODL_MELD_PREPARE_FAILED status=$status" >&2
  fi
}
trap report_failure EXIT

check_disk_capacity() {
  local target=$1
  local free_kib free_bytes
  free_kib=$(df -Pk "$target" | awk 'NR == 2 {print $4}')
  if [[ ! "$free_kib" =~ ^[0-9]+$ ]]; then
    echo "Cannot determine free disk space at $target: free_kib=$free_kib" >&2
    exit 19
  fi
  free_bytes=$((free_kib * 1024))
  if ((free_bytes < MELD_REQUIRED_FREE_BYTES)); then
    echo "MELD preparation requires at least $MELD_REQUIRED_FREE_BYTES free bytes; available=$free_bytes at $target" >&2
    exit 20
  fi
}

download_meld_archive() {
  local use_aria2=0
  case "$MELD_DOWNLOADER" in
    auto)
      command -v aria2c >/dev/null 2>&1 && use_aria2=1
      ;;
    aria2)
      command -v aria2c >/dev/null 2>&1 || {
        echo "MELD_DOWNLOADER=aria2 but aria2c is unavailable" >&2
        exit 18
      }
      use_aria2=1
      ;;
    hf)
      ;;
    *)
      echo "Unsupported MELD_DOWNLOADER=$MELD_DOWNLOADER" >&2
      exit 17
      ;;
  esac

  if ((use_aria2)); then
    if [[ "$DRY_RUN" != "1" && -f "$NETWORK_TURBO_PATH" ]]; then
      # AutoDL routes regular Hugging Face HTTPS through this proxy.
      # shellcheck disable=SC1090
      source "$NETWORK_TURBO_PATH"
    fi
    run aria2c \
      --continue=true \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=8M \
      --file-allocation=none \
      --max-tries=10 \
      --retry-wait=5 \
      --connect-timeout=15 \
      --timeout=30 \
      --summary-interval=10 \
      --console-log-level=notice \
      "--checksum=sha-256=$MELD_ARCHIVE_SHA256" \
      "--dir=$DOWNLOAD_ROOT" \
      --out=MELD.Raw.tar.gz \
      "$MELD_PARALLEL_URL"
  else
    if [[ "$DRY_RUN" != "1" && -f "$NETWORK_TURBO_PATH" ]]; then
      # AutoDL routes regular Hugging Face HTTPS through this proxy.
      # shellcheck disable=SC1090
      source "$NETWORK_TURBO_PATH"
    fi
    run hf download declare-lab/MELD MELD.Raw.tar.gz \
      --repo-type dataset --local-dir "$DOWNLOAD_ROOT"
  fi
}

verify_meld_archive() {
  local actual_bytes actual_sha256
  actual_bytes=$(stat -c '%s' "$ARCHIVE")
  if [[ "$actual_bytes" != "$MELD_ARCHIVE_BYTES" ]]; then
    echo "MELD archive size mismatch: expected=$MELD_ARCHIVE_BYTES actual=$actual_bytes" >&2
    exit 21
  fi
  actual_sha256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
  if [[ "$actual_sha256" != "$MELD_ARCHIVE_SHA256" ]]; then
    echo "MELD archive SHA-256 mismatch" >&2
    exit 22
  fi
  echo "MELD archive verified: bytes=$actual_bytes sha256=$actual_sha256"
}

download_meld_annotations() {
  local splits=(train dev test)
  local hashes=(
    d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db
    2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c
    8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888
  )
  local lines=(9990 1110 2611)
  local index split expected_hash expected_lines target temporary actual_hash actual_lines

  if [[ "$DRY_RUN" != "1" && -f "$NETWORK_TURBO_PATH" ]]; then
    # shellcheck disable=SC1090
    source "$NETWORK_TURBO_PATH"
  fi

  for index in 0 1 2; do
    split="${splits[$index]}"
    expected_hash="${hashes[$index]}"
    expected_lines="${lines[$index]}"
    target="$ANNOTATION_ROOT/${split}_sent_emo.csv"
    temporary="$target.part"

    if [[ "$DRY_RUN" == "1" ]]; then
      run curl --fail --location --retry 5 --output "$temporary" \
        "$MELD_ANNOTATION_BASE/${split}_sent_emo.csv"
      run bash -c "printf '%s  %s\\n' '$expected_hash' '$temporary' | sha256sum -c -"
      run mv "$temporary" "$target"
      continue
    fi

    if [[ -f "$target" ]]; then
      actual_hash=$(sha256sum "$target" | awk '{print $1}')
      [[ "$actual_hash" == "$expected_hash" ]] && continue
    fi

    run curl --fail --location --retry 5 --output "$temporary" \
      "$MELD_ANNOTATION_BASE/${split}_sent_emo.csv"
    actual_hash=$(sha256sum "$temporary" | awk '{print $1}')
    if [[ "$actual_hash" != "$expected_hash" ]]; then
      echo "MELD $split annotation SHA-256 mismatch" >&2
      exit 25
    fi
    actual_lines=$(wc -l < "$temporary" | tr -d ' ')
    if [[ "$actual_lines" != "$expected_lines" ]]; then
      echo "MELD $split annotation line count mismatch: expected=$expected_lines actual=$actual_lines" >&2
      exit 26
    fi
    run mv "$temporary" "$target"
  done
}

run mkdir -p "$DOWNLOAD_ROOT" "$RAW_ROOT" "$MEDIA_ROOT" "$ANNOTATION_ROOT" "$OUTPUT_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  download_meld_archive
  download_meld_annotations
  run tar -xzf "$ARCHIVE" -C "$RAW_ROOT"
  raw_dataset_root="$RAW_ROOT/MELD.Raw"
  dataset_root="$ANNOTATION_ROOT"
  for split in train dev test; do
    run mkdir -p "$MEDIA_ROOT/$split/source" "$MEDIA_ROOT/$split/flat"
    run tar -xzf "$raw_dataset_root/$split.tar.gz" -C "$MEDIA_ROOT/$split/source"
    run find "$MEDIA_ROOT/$split/source" -type f -name 'dia*_utt*.mp4' \
      -exec ln -sfn -t "$MEDIA_ROOT/$split/flat" '{}' +
  done
else
  check_disk_capacity "$RUNTIME_ROOT"
  download_meld_annotations
  if [[ ! -f "$EXTRACTED_MARKER" ]]; then
    download_meld_archive
    verify_meld_archive
  fi

  if [[ ! -f "$EXTRACTED_MARKER" ]]; then
    [[ -f "$ARCHIVE" ]] || {
      echo "MELD archive is missing after download: $ARCHIVE" >&2
      exit 21
    }
    run tar -xzf "$ARCHIVE" -C "$RAW_ROOT"
  fi

  train_archive=$(find "$RAW_ROOT" -type f -name train.tar.gz -print -quit)
  [[ -n "$train_archive" ]] || {
    echo "Cannot locate train.tar.gz under $RAW_ROOT" >&2
    exit 22
  }
  raw_dataset_root=$(dirname "$train_archive")
  dataset_root="$ANNOTATION_ROOT"

  if [[ ! -f "$EXTRACTED_MARKER" ]]; then
    for split in train dev test; do
      nested_archive=$(find "$raw_dataset_root" -maxdepth 2 -type f -name "$split.tar.gz" -print -quit)
      [[ -n "$nested_archive" ]] || {
        echo "Cannot locate $split.tar.gz under $dataset_root" >&2
        exit 23
      }
      run mkdir -p "$MEDIA_ROOT/$split/source" "$MEDIA_ROOT/$split/flat"
      run tar -xzf "$nested_archive" -C "$MEDIA_ROOT/$split/source"
      run find "$MEDIA_ROOT/$split/source" -type f -name 'dia*_utt*.mp4' \
        -exec ln -sfn -t "$MEDIA_ROOT/$split/flat" '{}' +
    done

    # Counts in the checksum-pinned raw media archive. The dev archive has
    # four unannotated clips and omits annotated dia110_utt7.mp4; test has
    # five unannotated clips. Raw media counts therefore intentionally differ
    # from the official CSV split counts used by the experiment manifest.
    expected_media_counts=(9989 1112 2615)
    splits=(train dev test)
    for index in 0 1 2; do
      split="${splits[$index]}"
      expected="${expected_media_counts[$index]}"
      actual=$(find "$MEDIA_ROOT/$split/flat" -type l -name 'dia*_utt*.mp4' | wc -l | tr -d ' ')
      if [[ "$actual" != "$expected" ]]; then
        echo "MELD $split media count mismatch: expected=$expected actual=$actual" >&2
        exit 24
      fi
    done
    run touch "$EXTRACTED_MARKER"
  fi
fi

train_csv="$dataset_root/train_sent_emo.csv"
dev_csv="$dataset_root/dev_sent_emo.csv"
test_csv="$dataset_root/test_sent_emo.csv"

run bimer prepare-meld \
  --train-csv "$train_csv" \
  --dev-csv "$dev_csv" \
  --test-csv "$test_csv" \
  --train-media "$MEDIA_ROOT/train/flat" \
  --dev-media "$MEDIA_ROOT/dev/flat" \
  --test-media "$MEDIA_ROOT/test/flat" \
  --output "$MANIFEST"
run bimer validate --manifest "$MANIFEST" --official-counts

if [[ "$DRY_RUN" == "1" ]]; then
  run rm -f "$ARCHIVE" \
    "$raw_dataset_root/train.tar.gz" \
    "$raw_dataset_root/dev.tar.gz" \
    "$raw_dataset_root/test.tar.gz"
else
  run rm -f "$ARCHIVE"
  find "$raw_dataset_root" -maxdepth 2 -type f \
    \( -name train.tar.gz -o -name dev.tar.gz -o -name test.tar.gz \) \
    -delete
fi

echo "AUTODL_MELD_PREPARE_COMPLETE manifest=$MANIFEST"
