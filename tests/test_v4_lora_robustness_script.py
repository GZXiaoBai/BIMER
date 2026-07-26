import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_v4_lora_robustness_features.py"


def test_lora_robustness_preparation_dry_run_records_three_views(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "state": "frozen",
                "version": "v4",
                "candidate_config": {
                    "feature_root": "/adapted/standard",
                    "adapter_path": "/adapter",
                    "adapter_base_model": "xlm-roberta-base",
                    "adapter_sha256": "abc",
                },
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selection",
            str(selection),
            "--whisper-manifest",
            "whisper.jsonl",
            "--robustness-features",
            "robust",
            "--output",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["adapter_sha256"] == "abc"
    assert payload["views"] == [
        "audio_snr_10db",
        "video_frame_drop_50pct",
        "whisper_text",
    ]
