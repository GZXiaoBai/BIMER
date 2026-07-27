# M2 Acceptance Record

Latest evidence date: 2026-07-27

Machine: MacBook Air M2, 8 GB unified memory

Deployment: `v2_quality_lagf`, canonical seed 42

Mode: offline encoders, sequential low-memory MPS feature extraction, CPU Whisper subprocess

## Completed checks

| Check | Result | Threshold |
|---|---:|---:|
| Chinese face video duration | 50.00 s | 30–60 s |
| Cold-cache Chinese analysis | 34.55 s | ≤120 s |
| Chinese content | 8/8 segments in Chinese | Required |
| Chinese face behavior | Vision enabled for all 8 segments | Required |
| English no-face video duration | 31.72 s | 30–60 s |
| Cold-cache English analysis | 18.42 s | ≤120 s |
| Peak memory footprint | 3.84 GB (3.58 GiB) | ≤6.5 GiB |
| BIMER process swap operations | 0 | 0 |
| Edited-text reanalysis | 4.03 s | ≤15 s |
| No-face behavior | Vision disabled for all 8 English segments | Required |
| JSON/CSV/PNG exports | Generated; browser download previously verified | Required |
| Clickable timeline | Seeking from 25.16 s previously confirmed | Required |
| Browser console errors | 0 in real-browser workflow | 0 |
| Wrong extension | Rejected before analysis | Required |
| File larger than 500 MB | Rejected before analysis | Required |
| Video without audio stream | Rejected before analysis | Required |

The final bilingual cold-cache run cleared seven feature-cache entries before
analysis. Its edited-text profile was 3.054 s for text, 0.089 s for cached
audio, 0.085 s for cached vision, and 0.073 s for fusion. This confirms that
text editing does not rerun transcription or recompute audio and visual
features.

The Chinese sample is a 50-second excerpt (seconds 5–55) from the Voice of
America Chinese interview “Ma Jian VOA interview 20181112.” The Wikimedia
Commons source page identifies the VOA-only work as public domain in the United
States. The original and excerpt hashes are recorded in `DATA_AND_LICENSES.md`.
The sample contains continuous Mandarin speech and a visible interview face.

The machine was already under substantial system-wide memory pressure before
the final bilingual run. macOS global swap usage increased from 9,833.06 MB to
10,582.31 MB. The BIMER process itself reported zero swap operations and stayed
below the memory limit. A clean-login system-wide “swap unchanged” check
remains pending and is not reported as passed.

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

The machine-readable final bilingual report records `complete: true` and
`passed: true`. Its private evidence hashes are:

- acceptance JSON:
  `580fdf9c69f90b979489e5d8ea3ff3377752be5fe5aac4563f7a1b2bf97c7d5b`
- resource JSON:
  `e4dac84d7c1c4e99e36c65b038d1b834b66cafacbd833b57eb68079c0292d9b7`
- exported JSON:
  `cf6f46dbd663fb38a994388beba6b86cd148937921838d0d6288c54ad3e7398f`
- exported CSV:
  `355c55fab46d6ee067ff6031c53504d9f5ff18041ee6423b6f70f34fbb394f7b`
- exported PNG:
  `3caa31698c7273262a749f5e5d13fa3b4f45aa919d047cc2d452570221b9097b`
