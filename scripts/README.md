# Script support index

The stable user interface is the `bimer` CLI. Scripts exist for workflows that
need orchestration, packaging, or historical cloud recovery. File paths are
retained so published experiment evidence remains reproducible.

## 正式支持

- `check_public_tree.py`: validate the public repository boundary.
- `generate_public_results.py`: regenerate aggregate public result tables.
- `m2_acceptance.py`, `run_m2_acceptance.sh`,
  `run_post_reboot_acceptance.sh`: frozen deployment acceptance.
- `lock_external_video_plan.py`, `prepare_external_annotation_handoff.py`,
  `evaluate_external_videos.py`: locked external evaluation workflow.
- `build_private_defense_package.py`: create the private offline package.
- `run_full_suite.sh`: reproduce the documented baseline suite when licensed
  data and features are available.
- `download_yunet.sh`: fetch the pinned YuNet asset.

These scripts must stay compatible with the deployment manifest, public result
schema, and current tests.

## 研究归档

- `run_v2_*`, `summarize_v2_*`: confirmatory V2 experiment evidence.
- `run_v3_*`, `summarize_v3_*`: stopped V3 ranking/loss study.
- `run_v4_*`, `summarize_v4_*`, `prepare_v4_*`: validation-only V4 LoRA and
  adaptive-context study.
- report builders and paired-bootstrap utilities: preserve figures and
  aggregate evidence used by the thesis.

Archived research scripts are maintained for evidence reproduction, not as the
recommended first entry for a new user. Their outputs must never be mixed
across protocol versions.

## 云端历史

Scripts named for Kaggle, AutoDL, or Lightning document earlier data extraction
and recovery sessions. They may contain provider-specific paths and machine
assumptions. Do not run them against a new account without reviewing every
path, storage limit, GPU count, completion marker, and shutdown option.

New cloud work should use a versioned protocol wrapper, a bounded time budget,
atomic status files, SHA-256 packaging, and automatic shutdown.
