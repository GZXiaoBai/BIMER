from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v5_experiments.py"


def _commands(*arguments: str) -> list[list[str]]:
    result = subprocess.run(
        [sys.executable, str(RUNNER), *arguments, "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        shlex.split(line.removeprefix("RUN "))
        for line in result.stdout.splitlines()
        if line.startswith("RUN ")
    ]


def test_v5_screen_runs_only_two_seed42_validation_candidates() -> None:
    commands = _commands("--stage", "screen")

    assert len(commands) == 2
    assert {
        float(command[command.index("--asr-consistency-weight") + 1]) for command in commands
    } == {0.05, 0.10}
    assert all(
        command[command.index("--model") + 1] == "asr_consistent_quality_lagf"
        and command[command.index("--seed") + 1] == "42"
        and "--v5-screen" in command
        and "--skip-test" in command
        and "--no-language" in command
        and "--augmentation-modality" in command
        and command[command.index("--augmentation-modality") + 1] == "text"
        for command in commands
    )


def test_v5_formal_uses_only_frozen_candidate_and_three_seeds(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "state": "frozen",
                "version": "v5",
                "selected_candidate": "beta_005",
                "candidate_config": {"asr_consistency_weight": 0.05},
            }
        ),
        encoding="utf-8",
    )

    commands = _commands(
        "--stage",
        "formal",
        "--selection",
        str(selection),
    )

    assert len(commands) == 3
    assert {command[command.index("--seed") + 1] for command in commands} == {
        "42",
        "123",
        "2026",
    }
    assert all(
        "--v5-formal" in command
        and "--skip-test" in command
        and command[command.index("--asr-consistency-weight") + 1] == "0.05"
        for command in commands
    )
