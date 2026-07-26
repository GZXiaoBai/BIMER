import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v4_exploratory_test.py"


def _selection(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "state": "frozen",
                "version": "v4",
                "selected_candidate": "combined_mu_010",
                "candidate_config": {
                    "model": "adaptive_context_prototype",
                    "prototype_loss_weight": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_exploratory_test_dry_run_plans_all_three_seeds_once(tmp_path):
    summary = tmp_path / "formal.json"
    summary.write_text(
        json.dumps(
            {
                "formal_stable": True,
                "seeds": [42, 123, 2026],
                "test_set_used": False,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selection",
            str(_selection(tmp_path / "selection.json")),
            "--formal-summary",
            str(summary),
            "--formal-root",
            str(tmp_path / "formal"),
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--output",
            str(tmp_path / "test"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("EVALUATE seed=") == 21
    assert "CONDITION standard" in result.stdout
    assert "CONDITION missing_text_audio" in result.stdout
    assert "OFFICIAL_TEST_WILL_BE_CONSUMED_ON_EXECUTION" in result.stdout
    assert not (tmp_path / "test" / "TEST_EVALUATED").exists()


def test_exploratory_test_rejects_unstable_formal_result(tmp_path):
    summary = tmp_path / "formal.json"
    summary.write_text(
        json.dumps(
            {
                "formal_stable": False,
                "seeds": [42, 123, 2026],
                "test_set_used": False,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selection",
            str(_selection(tmp_path / "selection.json")),
            "--formal-summary",
            str(summary),
            "--formal-root",
            str(tmp_path / "formal"),
            "--manifest",
            "manifest.jsonl",
            "--features",
            "features",
            "--output",
            str(tmp_path / "test"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not stable" in result.stderr
