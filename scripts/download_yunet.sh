#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-artifacts/models/face_detection_yunet_2023mar.onnx}"
URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
EXPECTED_BYTES=232589
EXPECTED_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
TEMP="${OUTPUT}.partial.onnx"

mkdir -p "$(dirname "$OUTPUT")"
trap 'rm -f "$TEMP"' EXIT

curl --ipv4 --fail --location --retry 3 \
  --connect-timeout 10 --max-time 60 \
  "$URL" \
  --output "$TEMP"

ACTUAL_BYTES="$(wc -c < "$TEMP" | tr -d ' ')"
if [[ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ]]; then
  echo "YuNet size mismatch: expected $EXPECTED_BYTES, got $ACTUAL_BYTES" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA256="$(sha256sum "$TEMP" | awk '{print $1}')"
else
  ACTUAL_SHA256="$(shasum -a 256 "$TEMP" | awk '{print $1}')"
fi
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "YuNet checksum mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256" >&2
  exit 1
fi

"${PYTHON:-python}" - "$TEMP" <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("cv2") is None:
    print(
        "OpenCV is not installed; YuNet size and checksum were verified.",
        file=sys.stderr,
    )
else:
    import cv2

    cv2.FaceDetectorYN.create(sys.argv[1], "", (320, 320), 0.8, 0.3, 5000)
PY

mv "$TEMP" "$OUTPUT"
trap - EXIT
echo "$OUTPUT"
