import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v4_lora_fallback.py"


def _decision(path: Path, *, decision: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "decision": decision,
                "best_candidate": "combined_mu_010",
                "selected": "combined_mu_010" if decision == "pass_v4a" else None,
                "candidate_configs": {
                    "combined_mu_010": {
                        "model": "adaptive_context_prototype",
                        "prototype_loss_weight": 0.10,
                        "use_adaptive_context_gate": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(tmp_path: Path, decision: str):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--decision",
            str(_decision(tmp_path / "decision.json", decision=decision)),
            "--manifest",
            "manifest.jsonl",
            "--source-features",
            "source",
            "--base-model",
            "xlm-roberta-base",
            "--output",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [
        shlex.split(line.removeprefix("RUN "))
        for line in result.stdout.splitlines()
        if line.startswith("RUN ")
    ]
    return result.stdout, commands


def test_lora_fallback_runs_only_after_v4a_screen_failure(tmp_path):
    stdout, commands = _run(tmp_path, "trigger_lora")

    assert "TRIGGER_LORA" in stdout
    assert len(commands) == 6
    training = [command for command in commands if "train_v4_text_lora.py" in command[1]]
    extraction = [
        command for command in commands if "extract_v4_lora_text_features.py" in command[1]
    ]
    fusion = [command for command in commands if command[1:4] == ["-m", "bimer.cli", "train"]]
    assert {float(command[command.index("--learning-rate") + 1]) for command in training} == {
        1e-4,
        2e-4,
    }
    assert len(extraction) == 2
    assert len(fusion) == 2
    assert all("--skip-test" in command and "--v4-screen" in command for command in fusion)


def test_lora_fallback_is_skipped_when_v4a_passes(tmp_path):
    stdout, commands = _run(tmp_path, "pass_v4a")

    assert "SKIP_LORA" in stdout
    assert commands == []
