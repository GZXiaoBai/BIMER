import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_system_checkpoint_selection_uses_validation_only(tmp_path):
    formal = tmp_path / "formal"
    for seed, score in ((42, 0.60), (123, 0.62), (2026, 0.61)):
        root = formal / "quality_lagf" / "joint" / f"seed-{seed}"
        root.mkdir(parents=True)
        (root / "best.pt").write_bytes(f"seed-{seed}".encode())
        (root / "results.json").write_text(
            json.dumps(
                {
                    "config": {"protocol_stage": "v3_formal"},
                    "test": {},
                    "evaluation_datasets": [],
                    "validation": {
                        "meld": {"weighted_f1": score},
                        "emotiontalk": {"weighted_f1": score},
                    },
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "selected.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "select_v3_system_checkpoint.py"),
            "--formal-root",
            str(formal),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )

    selected = json.loads(output.read_text())
    assert selected["seed"] == 123
    assert selected["test_set_used"] is False
    assert len(selected["checkpoint_sha256"]) == 64
