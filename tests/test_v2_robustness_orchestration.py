import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "scripts" / "run_v2_robustness.py"
AUTODL_WRAPPER = ROOT / "scripts" / "run_v2_robustness_autodl.sh"


def _dry_run(tmp_path: Path) -> list[list[str]]:
    result = subprocess.run(
        [
            sys.executable,
            str(ORCHESTRATOR),
            "--root",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return [
        shlex.split(line.removeprefix("RUN "))
        for line in result.stdout.splitlines()
        if line.startswith("RUN ")
    ]


def test_v2_robustness_matrix_covers_two_models_three_seeds_and_twelve_conditions(
    tmp_path,
):
    commands = _dry_run(tmp_path)

    assert len(commands) == 72
    outputs = [command[command.index("--output") + 1] for command in commands]
    for model in ("quality_lagf", "no_gates"):
        assert sum(f"/{model}/" in output for output in outputs) == 36
    for seed in ("42", "123", "2026"):
        assert sum(f"seed-{seed}.json" in output for output in outputs) == 24
    expected_conditions = {
        "standard",
        "audio_snr_20db",
        "audio_snr_10db",
        "video_frame_drop_25pct",
        "video_frame_drop_50pct",
        "whisper_text",
        "missing-text",
        "missing-audio",
        "missing-vision",
        "missing-audio-vision",
        "missing-text-vision",
        "missing-text-audio",
    }
    assert {
        Path(output).parent.name for output in outputs
    } == expected_conditions


def test_v2_robustness_uses_controlled_no_gate_checkpoints_and_valid_missing_flags(
    tmp_path,
):
    commands = _dry_run(tmp_path)

    full = next(
        command
        for command in commands
        if "/quality_lagf/standard/" in command[command.index("--output") + 1]
        and command[command.index("--output") + 1].endswith("seed-42.json")
    )
    no_gates = next(
        command
        for command in commands
        if "/no_gates/standard/" in command[command.index("--output") + 1]
        and command[command.index("--output") + 1].endswith("seed-42.json")
    )
    assert (
        "/formal/quality_lagf/quality_lagf/joint/seed-42/best.pt"
        in full[full.index("--checkpoint") + 1]
    )
    assert (
        "/ablations/no_gates/quality_lagf/joint/seed-42/best.pt"
        in no_gates[no_gates.index("--checkpoint") + 1]
    )

    missing = next(
        command
        for command in commands
        if "/quality_lagf/missing-text-audio/"
        in command[command.index("--output") + 1]
        and command[command.index("--output") + 1].endswith("seed-42.json")
    )
    assert "--condition-name" not in missing
    missing_values = [
        missing[index + 1]
        for index, value in enumerate(missing)
        if value == "--missing"
    ]
    assert missing_values == ["text", "audio"]

    corrupted = next(
        command
        for command in commands
        if "/quality_lagf/video_frame_drop_50pct/"
        in command[command.index("--output") + 1]
        and command[command.index("--output") + 1].endswith("seed-42.json")
    )
    assert corrupted[corrupted.index("--condition-name") + 1] == (
        "video_frame_drop_50pct"
    )

    whisper = next(
        command
        for command in commands
        if "/quality_lagf/whisper_text/"
        in command[command.index("--output") + 1]
        and command[command.index("--output") + 1].endswith("seed-42.json")
    )
    assert whisper[whisper.index("--manifest") + 1] == str(
        tmp_path / "data" / "processed" / "v2" / "whisper-test.jsonl"
    )


def test_v2_robustness_reruns_incomplete_existing_output(tmp_path):
    output = (
        tmp_path
        / "artifacts"
        / "experiments"
        / "v2"
        / "robustness"
        / "quality_lagf"
        / "standard"
        / "seed-42.json"
    )
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")

    commands = _dry_run(tmp_path)

    assert any(
        command[command.index("--output") + 1] == str(output)
        for command in commands
    )


def test_autodl_wrapper_prepares_views_and_packages_before_optional_shutdown(
    tmp_path,
):
    root = tmp_path / "project"
    runtime = tmp_path / "runtime"
    archive = root / "artifacts" / "v2-robustness.tar.gz"
    call_log = tmp_path / "calls.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "artifacts" / "experiments" / "v2" / "robustness").mkdir(
        parents=True
    )
    (root / "configs" / "experiment-v2-selection.json").write_text(
        '{"status":"frozen"}\n',
        encoding="utf-8",
    )
    for name in (
        "run_autodl_audio_robustness.sh",
        "run_autodl_video_robustness.sh",
        "run_autodl_whisper_robustness.sh",
    ):
        script = root / "scripts" / name
        script.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$0" >> "$CALL_LOG"\n',
            encoding="utf-8",
        )
        script.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        '#!/bin/sh\nprintf "python %s\\n" "$*" >> "$CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "BIMER_ROOT": str(root),
        "AUTODL_RUNTIME_ROOT": str(runtime),
        "BIMER_ARCHIVE": str(archive),
        "BIMER_PYTHON": str(fake_python),
        "AUTODL_AUTO_SHUTDOWN": "0",
    }

    result = subprocess.run(
        [str(AUTODL_WRAPPER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8")
    assert "run_autodl_audio_robustness.sh" in calls
    assert "run_autodl_video_robustness.sh" in calls
    assert "run_autodl_whisper_robustness.sh" in calls
    assert "scripts/run_v2_robustness.py" in calls
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()
    assert (
        root
        / "artifacts"
        / "experiments"
        / "v2"
        / "robustness"
        / "_status"
        / "DOWNLOAD_READY"
    ).is_file()
