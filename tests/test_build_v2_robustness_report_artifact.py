from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_v2_robustness_results.py"
REPORT_SCRIPT = ROOT / "scripts" / "build_v2_robustness_report_artifact.py"


def test_report_builder_uses_canonical_snapshot_and_declares_decision(tmp_path):
    from test_v2_robustness_summary import _fixture

    results, selection = _fixture(tmp_path)
    output_dir = tmp_path / "analysis"
    summary = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--results-root",
            str(results),
            "--selection-config",
            str(selection),
            "--output-dir",
            str(output_dir),
            "--bootstrap-iterations",
            "50",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert summary.returncode == 0, summary.stderr

    output = output_dir / "artifact.json"
    report = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--input-dir",
            str(output_dir),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert report.returncode == 0, report.stderr
    artifact = json.loads(output.read_text())
    assert artifact["manifest"]["title"] == "新版鲁棒性实验与最终模型选择"
    assert artifact["manifest"]["blocks"][0]["body"].startswith(
        "# 新版鲁棒性实验与最终模型选择"
    )
    assert artifact["manifest"]["charts"][0]["encodings"]["color"]["field"] == (
        "model_zh"
    )
    assert artifact["manifest"]["tables"][0]["defaultSort"]["direction"] == "asc"
    assert isinstance(artifact["snapshot"]["datasets"]["comparison_table"], list)
    assert len(artifact["snapshot"]["datasets"]["comparison_table"]) == 12
    assert (output_dir / "robustness-decision-report.md").is_file()
