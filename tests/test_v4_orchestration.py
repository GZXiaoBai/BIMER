import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v4_experiments.py"


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


def test_v4_screen_runs_six_seed42_validation_only_candidates():
    commands = _commands("--stage", "screen")

    assert len(commands) == 6
    assert all(
        command[command.index("--seed") + 1] == "42"
        and "--skip-test" in command
        and "--v4-screen" in command
        and "--no-language" in command
        for command in commands
    )
    assert sum(command[command.index("--model") + 1] == "quality_lagf" for command in commands) == 1
    assert {
        float(command[command.index("--prototype-loss-weight") + 1])
        for command in commands
        if command[command.index("--model") + 1] == "adaptive_context_prototype"
    } == {0.0, 0.05, 0.10, 0.20}


def test_v4_formal_runs_selected_model_and_three_ablations_for_three_seeds(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "state": "frozen",
                "version": "v4",
                "selected_candidate": "combined_mu_010",
                "candidate_config": {
                    "prototype_loss_weight": 0.10,
                    "use_adaptive_context_gate": True,
                },
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

    assert len(commands) == 12
    assert {command[command.index("--seed") + 1] for command in commands} == {
        "42",
        "123",
        "2026",
    }
    assert all("--skip-test" in command and "--v4-formal" in command for command in commands)
    assert sum("--no-adaptive-context-gate" in command for command in commands) == 6
    assert (
        sum(
            float(command[command.index("--prototype-loss-weight") + 1]) == 0.0
            for command in commands
        )
        == 6
    )


def test_v4_formal_uses_frozen_lora_feature_root(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "state": "frozen",
                "version": "v4",
                "selected_candidate": "lr_100",
                "candidate_config": {
                    "prototype_loss_weight": 0.10,
                    "use_adaptive_context_gate": True,
                    "feature_root": "/frozen/adapted/features",
                },
            }
        ),
        encoding="utf-8",
    )

    commands = _commands("--stage", "formal", "--selection", str(selection))

    assert all(
        command[command.index("--features") + 1] == "/frozen/adapted/features"
        for command in commands
    )
