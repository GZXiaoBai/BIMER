# M2 Acceptance Record

Latest evidence date: 2026-07-27

Machine: MacBook Air M2, 8 GB unified memory

Deployment: `v2_quality_lagf`, canonical seed 42

Mode: offline encoders, sequential low-memory MPS feature extraction, CPU Whisper subprocess

## Completed checks

| Check | Result | Threshold |
|---|---:|---:|
| English no-face video duration | 31.72 s | 30–60 s |
| Cold-cache English analysis | 19.12 s | ≤120 s |
| Peak memory footprint | 3.94 GB (3.67 GiB) | ≤6.5 GiB |
| BIMER process swap operations | 0 | 0 |
| Edited-text reanalysis | 2.95 s | ≤15 s |
| No-face behavior | Vision disabled for all 8 segments | Required |
| JSON/CSV/PNG exports | Generated and downloaded in a real browser | Required |
| Clickable timeline | Seeking from 25.16 s confirmed | Required |
| Browser console errors | 0 | 0 |
| Wrong extension | Rejected before analysis | Required |
| File larger than 500 MB | Rejected before analysis | Required |
| Video without audio stream | Rejected before analysis | Required |

The cold-cache run cleared four feature-cache entries before analysis. Its
edited-text profile was 2.70 s for text, 0.077 s for cached audio, 0.075 s for
cached vision, and 0.034 s for fusion. This confirms that text editing does not
recompute audio or visual features.

Sequential low-memory loading reduced the measured peak memory footprint from
5.33 GB to 3.94 GB (about 26%) while preserving the exported CSV and PNG
SHA-256 values. The English analysis time changed from 17.87 s to 19.12 s and
remained well below the 120 s limit.

The machine was already under substantial system-wide memory pressure before
the run. macOS global swap usage increased by approximately 0.41 GB during the
low-memory run, compared with approximately 1.50 GB in the eager-loading run.
The BIMER process itself reported zero swap operations. A clean-login
system-wide “swap unchanged” check remains pending and is not reported as
passed.

## Runtime isolation

The parent process loaded OpenCV but did not import PyAV. Whisper ran through
`SubprocessWhisperTranscriber`, eliminating the previously observed duplicate
FFmpeg-library warnings from the OpenCV/YuNet process.

## Real-browser workflow

The following actions were completed against the real Gradio application and
the real V2 checkpoint:

1. Upload the English MP4.
2. Run automatic transcription and segmentation.
3. Edit a transcript cell.
4. Run emotion analysis.
5. Inspect the confidence state and modality-quality warning.
6. Click the timeline and verify video seeking.
7. Download JSON, CSV, and PNG outputs.
8. Clear the 24-hour feature cache.

The Chinese face-video half of final acceptance remains pending because an
authorized 30–60 second Chinese sample has not yet been supplied. No synthetic
or unlicensed sample will be substituted for the final claim.

The machine-readable partial report intentionally records `complete: false`
and `passed: false`. Its private evidence hashes are:

- acceptance JSON:
  `78882d9d23ea6552e3c80692ac632afa6b4ec015b6226713ba43e420ff78b62a`
- resource JSON:
  `8ec73d54c88d1875d8180669304707a2cf84e0f6afce5d7c3abbea2a5cb9f780`
- exported CSV:
  `36274e04f76ed1a85a8aabc7ad4d08f13a0caf77ca046d89827078df0d7f0858`
- exported PNG:
  `9114a012f779cb49e2131e422098e659537cedd2b050edd7f553d0649711968b`
