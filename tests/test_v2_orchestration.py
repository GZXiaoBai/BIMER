import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_selection_matches_experiment_config():
    with (ROOT / "configs" / "experiment-v2.toml").open("rb") as stream:
        config = tomllib.load(stream)
    selection = json.loads(
        (ROOT / "configs" / "experiment-v2-selection.json").read_text(encoding="utf-8")
    )

    assert selection["status"] == "frozen"
    assert selection["selection_scope"] == "validation_only"
    assert selection["test_set_used_for_selection"] is False
    assert config["formal_learning_rate"] == selection["formal_experiments"]["learning_rate"]
    assert config["seeds"] == selection["formal_experiments"]["seeds"]
    assert config["formal"]["variants"] == selection["formal_experiments"]["variants"]
    assert list(config["ablations"]) == selection["ablations"]


def test_validation_screen_dry_run_never_evaluates_test():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_v2_experiments.py"),
            "--stage",
            "audio-screen",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in result.stdout.splitlines() if line.startswith("RUN ")]
    assert len(commands) == 6
    assert all("--skip-test" in command for command in commands)
    assert all("--seed 42" in command for command in commands)


def test_fusion_screen_can_select_one_variant():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_v2_experiments.py"),
            "--stage",
            "fusion-screen",
            "--variant",
            "early_context",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in result.stdout.splitlines() if line.startswith("RUN ")]
    assert len(commands) == 3
    assert all("--model early_context" in command for command in commands)
    assert all("--skip-test" in command for command in commands)


def test_formal_dry_run_uses_frozen_quality_selection():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_v2_experiments.py"),
            "--stage",
            "formal",
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
    assert len(commands) == 12
    assert all(command[command.index("--learning-rate") + 1] == "0.0001" for command in commands)
    assert all("--skip-test" not in command for command in commands)
    assert sum(command[command.index("--model") + 1] == "early_mlp" for command in commands) == 3
    assert (
        sum(command[command.index("--model") + 1] == "early_context" for command in commands) == 3
    )
    assert sum(command[command.index("--model") + 1] == "quality_lagf" for command in commands) == 3
    no_gate_commands = [
        command for command in commands if command[command.index("--model") + 1] == "lagf"
    ]
    assert len(no_gate_commands) == 3
    assert all("--no-gates" in command for command in no_gate_commands)
    for seed in ("42", "123", "2026"):
        assert sum(command[command.index("--seed") + 1] == seed for command in commands) == 4


def test_ablation_dry_run_uses_frozen_learning_rate_and_all_seeds():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_v2_experiments.py"),
            "--stage",
            "ablations",
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
    assert len(commands) == 18
    assert all(command[command.index("--learning-rate") + 1] == "0.0001" for command in commands)
    assert all(command[command.index("--model") + 1] == "quality_lagf" for command in commands)
    for seed in ("42", "123", "2026"):
        assert sum(command[command.index("--seed") + 1] == seed for command in commands) == 6


def test_formal_and_ablation_autodl_runner_executes_both_stages_and_packages(
    tmp_path,
):
    root = tmp_path / "project"
    output = root / "artifacts" / "experiments" / "v2"
    archive = root / "artifacts" / "formal-ablations.tar.gz"
    fake_bin = tmp_path / "bin"
    call_log = tmp_path / "python-calls.log"
    fake_bin.mkdir()
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "experiment-v2.toml").write_text(
        "formal_learning_rate = 0.0001\n",
        encoding="utf-8",
    )
    (root / "configs" / "experiment-v2-selection.json").write_text(
        '{"status":"frozen"}\n',
        encoding="utf-8",
    )
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "BIMER_ROOT": str(root),
        "BIMER_OUTPUT": str(output),
        "BIMER_ARCHIVE": str(archive),
        "AUTODL_AUTO_SHUTDOWN": "0",
    }

    result = subprocess.run(
        [str(ROOT / "scripts" / "run_v2_formal_ablations_autodl.sh")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--stage formal" in calls[0]
    assert "--stage ablations" in calls[1]
    assert all("--quality-features" in call for call in calls)
    assert all(call.count("--augmentation-manifest") == 3 for call in calls)
    assert all(call.count("--augmentation-features") == 3 for call in calls)
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()
    assert (output / "_status" / "DOWNLOAD_READY").is_file()


def test_autodl_wrapper_archives_and_hashes_before_optional_shutdown():
    text = (ROOT / "scripts" / "run_v2_autodl.sh").read_text(encoding="utf-8")

    assert "AUTODL_AUTO_SHUTDOWN" in text
    assert "sha256sum" in text
    assert "shutdown -h now" in text
    assert text.index("sha256sum") < text.index("shutdown -h now")
    assert "trap on_exit EXIT" in text


def test_quality_view_script_builds_all_three_real_corruptions():
    text = (ROOT / "scripts" / "prepare_v2_quality_views.sh").read_text(encoding="utf-8")

    assert "attach-quality" in text
    assert "--audio-snr 10" in text
    assert "--frame-drop 0.5" in text
    assert "asr-manifest" in text
    assert "--only-modality text" in text


def test_emotiontalk_quality_runner_is_dataset_scoped_and_auto_shutdown_safe():
    text = (ROOT / "scripts" / "run_v2_quality_emotiontalk_autodl.sh").read_text(encoding="utf-8")

    assert "--dataset emotiontalk" in text
    assert "output/emotiontalk.jsonl" in text
    assert "EmotionTalk media preflight" in text
    assert '--workers "$WORKERS"' in text
    assert "QUALITY_RANGE" in text
    assert '--start-shard "$start_shard"' in text
    assert "--audio-snr 10" in text
    assert "--frame-drop 0.5" in text
    assert "asr-manifest" in text
    assert "DOWNLOAD_READY" in text
    assert text.index("sha256sum") < text.index("shutdown -h now")


def test_meld_quality_and_screen_runner_preserves_existing_views_and_is_validation_only():
    text = (ROOT / "scripts" / "run_v2_meld_quality_and_screen_autodl.sh").read_text(
        encoding="utf-8"
    )

    assert "--dataset meld" in text
    assert 'find -L "$BASE/meld/$split"' in text
    assert "QUALITY_RANGE dataset=meld" in text
    assert "v2-corruption-joint-clean" in text
    assert "v2-corruption-joint-audio10" in text
    assert "v2-corruption-joint-video50" in text
    assert "v2-corruption-joint-whisper" in text
    assert "v2-corruption-clean" not in text.replace("v2-corruption-joint-clean", "")
    assert "--audio-snr 10" in text
    assert "--frame-drop 0.5" in text
    assert "asr-manifest" in text
    assert "--stage fusion-screen" in text
    assert "--variant lagf" in text
    assert "--variant lagf_no_gates" in text
    assert "--variant quality_lagf" in text
    assert "--augmentation-manifest" in text
    assert "--augmentation-features" in text
    assert "AUTODL_AUTO_SHUTDOWN" in text
    assert "DOWNLOAD_READY" in text
    assert text.index("sha256sum") < text.index("shutdown -h now")
