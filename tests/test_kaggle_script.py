import os
import subprocess
from pathlib import Path


def test_emotiontalk_kaggle_script_dry_run_is_minimal_and_secret_safe(tmp_path):
    project_root = Path(__file__).parents[1]
    secret = "hf_test_secret_must_not_be_printed"
    raw_root = tmp_path / "raw"
    (raw_root / "downloads" / "emotiontalk").mkdir(parents=True)
    (raw_root / "downloads" / "emotiontalk" / "Multimodal.tar").touch()
    (raw_root / "raw" / "emotiontalk").mkdir(parents=True)
    (raw_root / "raw" / "emotiontalk" / ".multimodal-extracted").touch()
    result = subprocess.run(
        ["bash", str(project_root / "scripts" / "prepare_emotiontalk_kaggle.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "HF_TOKEN": secret,
            "RAW_ROOT": str(raw_root),
            "OUTPUT_ROOT": str(tmp_path / "output"),
        },
    )

    assert "Multimodal.tar" in result.stdout
    assert "Audio.tar" not in result.stdout
    assert "prepare-emotiontalk-official" in result.stdout
    assert "validate --manifest" in result.stdout
    assert secret not in result.stdout


def test_emotiontalk_kaggle_script_keeps_large_raw_files_out_of_working():
    project_root = Path(__file__).parents[1]
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RAW_ROOT", "OUTPUT_ROOT"}
    }
    result = subprocess.run(
        ["bash", str(project_root / "scripts" / "prepare_emotiontalk_kaggle.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={**clean_environment, "DRY_RUN": "1", "HF_TOKEN": "hf_test"},
    )

    assert "/tmp/bimer-data/downloads/emotiontalk/Multimodal.tar" in result.stdout
    assert "/tmp/bimer-data/raw/emotiontalk" in result.stdout
    assert "/kaggle/working/bimer-output/emotiontalk.jsonl" in result.stdout
    assert "/kaggle/working/bimer-data/downloads" not in result.stdout


def test_kaggle_guide_documents_parallel_pipeline_and_rollback():
    project_root = Path(__file__).parents[1]
    guide = (project_root / "docs" / "kaggle.md").read_text(encoding="utf-8")

    assert "--mode parallel" in guide
    assert "--text-audio-device cuda:0" in guide
    assert "--vision-device cuda:1" in guide
    assert "watch -n 2 nvidia-smi" in guide
    assert "--mode serial" in guide
    assert "staging" in guide
