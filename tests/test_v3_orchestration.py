import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v3_experiments.py"


def _commands(*arguments):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments, "--dry-run"],
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


def test_v3_loss_screen_is_seed42_validation_only():
    commands = _commands("--stage", "loss-screen")

    assert len(commands) == 3
    assert all("--seed" in command and command[command.index("--seed") + 1] == "42" for command in commands)
    assert all("--skip-test" in command and "--v3-screen" in command for command in commands)
    assert {
        command[command.index("--classification-loss") + 1] for command in commands
    } == {"weighted_ce", "balanced_softmax", "focal"}
    assert all(command.count("--augmentation-modality") == 3 for command in commands)


def test_v3_ranking_screen_uses_three_fixed_lambdas():
    commands = _commands(
        "--stage",
        "ranking-screen",
        "--classification-loss",
        "balanced_softmax",
    )

    assert len(commands) == 3
    assert {
        float(command[command.index("--gate-ranking-weight") + 1])
        for command in commands
    } == {0.05, 0.10, 0.20}
    assert all("--skip-test" in command and "--v3-screen" in command for command in commands)


def test_v3_formal_still_skips_test_and_uses_frozen_selection(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "state": "frozen",
                "version": "v3",
                "classification_loss": "balanced_softmax",
                "gate_ranking_weight": 0.05,
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

    assert len(commands) == 6
    assert all(
        "--skip-test" in command
        and "--v3-screen" not in command
        and "--v3-formal" in command
        for command in commands
    )
    assert {command[command.index("--seed") + 1] for command in commands} == {
        "42",
        "123",
        "2026",
    }
