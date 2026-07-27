from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bimer.system_acceptance import evaluate_system_swap, parse_swap_megabytes

ROOT = Path(__file__).resolve().parents[1]


def test_parse_swap_megabytes_reads_macos_swapusage():
    assert (
        parse_swap_megabytes("total = 2048.00M  used = 12.50M  free = 2035.50M  (encrypted)")
        == 12.5
    )


def test_system_swap_acceptance_requires_clean_start_and_no_increase():
    result = evaluate_system_swap(
        "total = 2048.00M  used = 0.00M  free = 2048.00M  (encrypted)",
        "total = 2048.00M  used = 0.00M  free = 2048.00M  (encrypted)",
    )

    assert result == {
        "before_used_mb": 0.0,
        "after_used_mb": 0.0,
        "increase_mb": 0.0,
        "clean_start": True,
        "unchanged": True,
        "passed": True,
    }


@pytest.mark.parametrize(
    ("before", "after", "reason"),
    [
        (300.0, 300.0, "dirty start"),
        (0.0, 1.0, "swap increase"),
    ],
)
def test_system_swap_acceptance_rejects_dirty_or_increasing_swap(
    before: float,
    after: float,
    reason: str,
):
    result = evaluate_system_swap(
        f"total = 2048.00M  used = {before:.2f}M  free = 0.00M  (encrypted)",
        f"total = 2048.00M  used = {after:.2f}M  free = 0.00M  (encrypted)",
    )

    assert not result["passed"], reason


def test_validate_system_swap_script_writes_machine_readable_evidence(tmp_path: Path):
    resource_report = tmp_path / "m2-resource-report.json"
    resource_report.write_text(
        json.dumps(
            {
                "process_swaps": 0,
                "system_swap_before": (
                    "total = 2048.00M  used = 0.00M  free = 2048.00M  (encrypted)"
                ),
                "system_swap_after": (
                    "total = 2048.00M  used = 0.00M  free = 2048.00M  (encrypted)"
                ),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "system-swap-acceptance.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_system_swap.py"),
            "--resource-report",
            str(resource_report),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
