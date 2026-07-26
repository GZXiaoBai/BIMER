#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <bimer-v3-results.tar.gz> [extract-directory]" >&2
  exit 2
fi

ARCHIVE="$1"
CHECKSUM="$ARCHIVE.sha256"
DESTINATION="${2:-}"
[[ -f "$ARCHIVE" ]] || { echo "Archive not found: $ARCHIVE" >&2; exit 1; }
[[ -f "$CHECKSUM" ]] || { echo "Checksum not found: $CHECKSUM" >&2; exit 1; }

(
  cd "$(dirname "$ARCHIVE")"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "$(basename "$CHECKSUM")"
  else
    shasum -a 256 -c "$(basename "$CHECKSUM")"
  fi
)
tar -tzf "$ARCHIVE" >/dev/null

if [[ -n "$DESTINATION" ]]; then
  mkdir -p "$DESTINATION"
  tar -xzf "$ARCHIVE" -C "$DESTINATION"
fi

touch "$ARCHIVE.LOCAL_VERIFIED"
echo "V3 archive checksum and tar structure verified"
