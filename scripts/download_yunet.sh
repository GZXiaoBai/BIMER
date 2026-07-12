#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-artifacts/models/face_detection_yunet_2023mar.onnx}"
mkdir -p "$(dirname "$OUTPUT")"
curl --fail --location \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  --output "$OUTPUT"
echo "$OUTPUT"

