# M2 Acceptance Record

Latest evidence date: 2026-07-27

Machine: MacBook Air M2, 8 GB unified memory

Deployment: `v2_quality_lagf`, canonical seed 42

Mode: offline encoders, sequential low-memory MPS feature extraction, CPU Whisper subprocess

## Completed checks

| Check | Result | Threshold |
|---|---:|---:|
| Chinese face video duration | 50.00 s | 30–60 s |
| Cold-cache Chinese analysis | 36.51 s | ≤120 s |
| Chinese content | 13/13 segments in Chinese | Required |
| Chinese face behavior | Vision enabled for all 13 segments | Required |
| English no-face video duration | 31.72 s | 30–60 s |
| Cold-cache English analysis | 30.36 s | ≤120 s |
| Peak memory footprint | 3.84 GB (3.58 GiB) | ≤6.5 GiB |
| BIMER process swap operations | 0 | 0 |
| Edited-text reanalysis | 5.28 s | ≤15 s |
| No-face behavior | Vision disabled for all 8 English segments | Required |
| JSON/CSV/PNG exports | Generated; browser download previously verified | Required |
| Clickable timeline | Seeking from 25.16 s previously confirmed | Required |
| Browser console errors | 0 in real-browser workflow | 0 |
| Wrong extension | Rejected before analysis | Required |
| File larger than 500 MB | Rejected before analysis | Required |
| Video without audio stream | Rejected before analysis | Required |

The final bilingual cold-cache run cleared seven feature-cache entries before
analysis. Its edited-text profile was 4.397 s for text, 0.108 s for cached
audio, 0.100 s for cached vision, and 0.057 s for fusion. This confirms that
text editing does not rerun transcription or recompute audio and visual
features.

The Chinese sample is a 50-second excerpt (seconds 70–120) from the Voice of
America Chinese business interview “VOA专访中国玻璃大王曹德旺.” The Wikimedia
Commons source page identifies the VOA-only work as public domain in the United
States and notes that the imported upload has not received an additional
administrator review. The original and excerpt hashes are recorded in
`DATA_AND_LICENSES.md`. The sample contains continuous Mandarin discussion of
manufacturing, labor, and costs, with an interview face visible throughout.

The machine was already under substantial system-wide memory pressure before
the final bilingual run. macOS global swap usage increased from 9,542.00 MB to
10,586.31 MB. The BIMER process itself reported zero swap operations and stayed
below the memory limit. A clean-login system-wide “swap unchanged” check
remains pending and is not reported as passed. The strict post-reboot check is
now automated by `scripts/run_post_reboot_acceptance.sh`: it reruns the complete
bilingual acceptance, requires initial swap usage no greater than 256 MB,
requires zero increase and zero BIMER process swap operations, and writes
`system-swap-acceptance.json`.

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

A 143.84-second real-browser backup recording of the Chinese workflow was also
captured from the final V2 deployment and transcoded to
`output/deliverables/BIMER-中文离线演示.mp4`. It contains no synthetic analyzer
or prerecorded model output.

The machine-readable final bilingual report records `complete: true` and
`passed: true`. Its private evidence hashes are:

- acceptance JSON:
  `462d16c872d2d6228d4e7697a6c7aa9f58f9f9b459344fa0a233b200f40c2a85`
- resource JSON:
  `3bbfab88bda066226c342dc9fec0885cf3ddf2f05d7939fb1aae14e9255f2f1c`
- exported JSON:
  `f909f8771f772c0d5d8a1627371a0a504ddeaff9333e5d73af34b2e6c5e406e9`
- exported CSV:
  `04c63fcb6043b782f0eedd474e67ada5d24c97366ce597a9a910a7695fa5df93`
- exported PNG:
  `37a157d7e00f39d1907ee4d71713943a55364b0f4190209123bba7aac8abd94e`
