# M2 Acceptance Record

Date: 2026-07-26

Machine: MacBook Air M2, 8 GB unified memory

Deployment: `v2_quality_lagf`, canonical seed 42

Mode: offline encoders, MPS feature extraction, CPU Whisper subprocess

## Completed checks

| Check | Result | Threshold |
|---|---:|---:|
| English no-face video duration | 31.72 s | 30–60 s |
| First complete CLI analysis | 58.37 s | ≤120 s |
| Peak memory footprint | 5.28 GB | ≤6.5 GB |
| Swap operations | 0 | 0 |
| Edited-text reanalysis | 1.99 s | ≤15 s |
| No-face behavior | Vision disabled for all 8 segments | Required |
| JSON/CSV/PNG exports | Generated and downloaded in a real browser | Required |
| Clickable timeline | Seeking from 25.16 s confirmed | Required |
| Browser console errors | 0 | 0 |
| Wrong extension | Rejected before analysis | Required |
| File larger than 500 MB | Rejected before analysis | Required |
| Video without audio stream | Rejected before analysis | Required |

The edited-text profile was 1.84 s for text, 0.003 s for cached audio,
0.013 s for cached vision, and 0.036 s for fusion. This confirms that text
editing does not recompute the audio and visual encoders.

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
