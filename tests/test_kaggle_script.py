import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


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
    assert "14400" in result.stdout
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


def test_emotiontalk_download_retries_transient_failure_with_bounded_attempts(
    tmp_path,
):
    project_root = Path(__file__).parents[1]
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    attempts_path = tmp_path / "hf-attempts.txt"
    xet_settings_path = tmp_path / "hf-xet-settings.txt"
    timeout_log = tmp_path / "timeout.log"

    (raw_root / "raw" / "emotiontalk").mkdir(parents=True)
    (raw_root / "raw" / "emotiontalk" / ".multimodal-extracted").touch()
    (raw_root / "sources" / "emotiontalk-official" / ".git").mkdir(
        parents=True
    )

    _write_executable(
        fake_bin / "hf",
        """#!/usr/bin/env bash
set -euo pipefail
attempts=0
if [[ -f "$HF_FAKE_ATTEMPTS_PATH" ]]; then
  attempts="$(cat "$HF_FAKE_ATTEMPTS_PATH")"
fi
attempts=$((attempts + 1))
printf '%s' "$attempts" > "$HF_FAKE_ATTEMPTS_PATH"
printf '%s,%s,%s,%s,%s' \
  "${HF_HUB_DISABLE_XET:-}" \
  "${HF_XET_HIGH_PERFORMANCE:-}" \
  "${HF_XET_NUM_CONCURRENT_RANGE_GETS:-}" \
  "${HF_XET_CHUNK_CACHE_SIZE_BYTES:-}" \
  "${HF_HUB_DOWNLOAD_TIMEOUT:-}" > "$HF_FAKE_XET_SETTINGS_PATH"
if [[ "$attempts" -eq 1 ]]; then
  exit 17
fi
mkdir -p "$RAW_ROOT/downloads/emotiontalk"
touch "$RAW_ROOT/downloads/emotiontalk/Multimodal.tar"
""",
    )
    _write_executable(
        fake_bin / "timeout",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$HF_FAKE_TIMEOUT_LOG"
while [[ "${1:-}" == --* ]]; do
  shift
done
shift
exec "$@"
""",
    )
    for command in ("git", "bimer"):
        _write_executable(
            fake_bin / command,
            "#!/usr/bin/env bash\nexit 0\n",
        )

    result = subprocess.run(
        ["bash", str(project_root / "scripts" / "prepare_emotiontalk_kaggle.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HF_TOKEN": "hf_test",
            "HF_FAKE_ATTEMPTS_PATH": str(attempts_path),
            "HF_FAKE_XET_SETTINGS_PATH": str(xet_settings_path),
            "HF_FAKE_TIMEOUT_LOG": str(timeout_log),
            "HF_DOWNLOAD_MAX_ATTEMPTS": "3",
            "HF_DOWNLOAD_MAX_SECONDS": "5",
            "HF_DOWNLOAD_REQUIRED_FREE_BYTES": "1",
            "RAW_ROOT": str(raw_root),
            "OUTPUT_ROOT": str(output_root),
        },
    )

    assert result.returncode == 0, result.stderr
    assert attempts_path.read_text(encoding="utf-8") == "2"
    assert xet_settings_path.read_text(encoding="utf-8") == "0,1,32,0,60"
    timeout_lines = timeout_log.read_text(encoding="utf-8").splitlines()
    assert len(timeout_lines) == 2
    assert all("hf download BAAI/Emotiontalk Multimodal.tar" in line for line in timeout_lines)
    assert "attempt 1/3 failed" in result.stderr


def test_emotiontalk_download_reports_partial_file_throughput(tmp_path):
    project_root = Path(__file__).parents[1]
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    (raw_root / "raw" / "emotiontalk").mkdir(parents=True)
    (raw_root / "raw" / "emotiontalk" / ".multimodal-extracted").touch()
    (raw_root / "sources" / "emotiontalk-official" / ".git").mkdir(
        parents=True
    )

    _write_executable(
        fake_bin / "hf",
        r"""#!/usr/bin/env bash
set -euo pipefail
cache="$RAW_ROOT/downloads/emotiontalk/.cache/huggingface/download"
mkdir -p "$cache"
incomplete="$cache/test.incomplete"
dd if=/dev/zero of="$incomplete" bs=1048576 count=1 2>/dev/null
sleep 2
mv "$incomplete" "$RAW_ROOT/downloads/emotiontalk/Multimodal.tar"
""",
    )
    _write_executable(
        fake_bin / "timeout",
        """#!/usr/bin/env bash
set -euo pipefail
while [[ "${1:-}" == --* ]]; do
  shift
done
shift
exec "$@"
""",
    )
    for command in ("git", "bimer"):
        _write_executable(
            fake_bin / command,
            "#!/usr/bin/env bash\nexit 0\n",
        )

    result = subprocess.run(
        ["bash", str(project_root / "scripts" / "prepare_emotiontalk_kaggle.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HF_TOKEN": "hf_test",
            "HF_DOWNLOAD_MAX_SECONDS": "10",
            "HF_DOWNLOAD_PROGRESS_SECONDS": "1",
            "HF_DOWNLOAD_POLL_SECONDS": "0.1",
            "HF_DOWNLOAD_REQUIRED_FREE_BYTES": "1",
            "RAW_ROOT": str(raw_root),
            "OUTPUT_ROOT": str(output_root),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "EmotionTalk download progress:" in result.stderr
    assert "partial_bytes=1048576" in result.stderr
    assert "allocated_bytes=" in result.stderr
    assert "bytes_per_second=" in result.stderr
    assert "allocated_bytes_per_second=" in result.stderr


def test_emotiontalk_download_fails_before_network_when_disk_is_too_small(
    tmp_path,
):
    project_root = Path(__file__).parents[1]
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hf_called = tmp_path / "hf-called"

    _write_executable(
        fake_bin / "hf",
        f"#!/usr/bin/env bash\ntouch {hf_called}\nexit 0\n",
    )

    result = subprocess.run(
        ["bash", str(project_root / "scripts" / "prepare_emotiontalk_kaggle.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HF_TOKEN": "hf_test",
            "HF_DOWNLOAD_REQUIRED_FREE_BYTES": str(10**18),
            "RAW_ROOT": str(raw_root),
            "OUTPUT_ROOT": str(output_root),
        },
    )

    assert result.returncode != 0
    assert "insufficient free disk space" in result.stderr
    assert "required_bytes=" in result.stderr
    assert not hf_called.exists()


def test_model_cache_retries_hf_downloads_then_verifies_offline(tmp_path):
    project_root = Path(__file__).parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    model_root = tmp_path / "models"
    attempts_root = tmp_path / "attempts"
    attempts_root.mkdir()
    hf_environment_log = tmp_path / "hf-environment.log"
    python_environment_log = tmp_path / "python-environment.log"
    yunet_path = model_root / "yunet.onnx"
    yunet_path.parent.mkdir(parents=True)
    yunet_path.touch()

    _write_executable(
        fake_bin / "hf",
        r"""#!/usr/bin/env bash
set -euo pipefail
repo="$2"
key="${repo//\//_}"
attempt_file="$HF_FAKE_ATTEMPTS_ROOT/$key"
attempt=0
if [[ -f "$attempt_file" ]]; then
  attempt="$(cat "$attempt_file")"
fi
attempt=$((attempt + 1))
printf '%s' "$attempt" > "$attempt_file"
printf '%s,%s,%s\n' \
  "$repo" \
  "${HF_HUB_DISABLE_XET:-}" \
  "${HF_HOME:-}" >> "$HF_FAKE_ENVIRONMENT_LOG"
if [[ "$attempt" -eq 1 ]]; then
  exit 17
fi
mkdir -p "$HF_HOME/hub"
""",
    )
    _write_executable(
        fake_bin / "timeout",
        """#!/usr/bin/env bash
set -euo pipefail
while [[ "${1:-}" == --* ]]; do
  shift
done
shift
exec "$@"
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
printf '%s,%s\n' \
  "${HF_HUB_OFFLINE:-}" \
  "${TRANSFORMERS_OFFLINE:-}" >> "$PYTHON_FAKE_ENVIRONMENT_LOG"
if [[ "${HF_HUB_OFFLINE:-}" == "1" ]]; then
  touch "$MODEL_CACHE_READY_PATH"
else
  mkdir -p "$TORCH_HOME/hub/checkpoints"
  touch "$TORCH_HOME/hub/checkpoints/r3d_18-b3b3357e.pth"
fi
""",
    )

    result = subprocess.run(
        ["bash", str(project_root / "scripts" / "prepare_model_cache_kaggle.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PYTHON": str(fake_bin / "python"),
            "MODEL_CACHE_ROOT": str(model_root),
            "MODEL_CACHE_READY_PATH": str(model_root / "ready.json"),
            "YUNET_MODEL_PATH": str(yunet_path),
            "HF_FAKE_ATTEMPTS_ROOT": str(attempts_root),
            "HF_FAKE_ENVIRONMENT_LOG": str(hf_environment_log),
            "PYTHON_FAKE_ENVIRONMENT_LOG": str(python_environment_log),
            "MODEL_DOWNLOAD_MAX_ATTEMPTS": "3",
            "MODEL_DOWNLOAD_MAX_SECONDS": "5",
            "MODEL_DOWNLOAD_RETRY_DELAY_SECONDS": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert (attempts_root / "xlm-roberta-base").read_text() == "2"
    assert (attempts_root / "facebook_wav2vec2-xls-r-300m").read_text() == "2"
    hf_lines = hf_environment_log.read_text(encoding="utf-8").splitlines()
    assert len(hf_lines) == 4
    assert all(",1," in line for line in hf_lines)
    assert python_environment_log.read_text(encoding="utf-8").splitlines() == [
        ",",
        "1,1",
    ]
    assert (model_root / "ready.json").is_file()
    assert "attempt 1/3 failed" in result.stderr


def test_kaggle_guide_documents_parallel_pipeline_and_rollback():
    project_root = Path(__file__).parents[1]
    guide = (project_root / "docs" / "kaggle.md").read_text(encoding="utf-8")

    assert "--mode parallel" in guide
    assert "--text-audio-device cuda:0" in guide
    assert "--vision-device cuda:1" in guide
    assert "watch -n 2 nvidia-smi" in guide
    assert "--mode serial" in guide
    assert "staging" in guide


def test_kaggle_guide_documents_cross_session_train_ranges():
    project_root = Path(__file__).parents[1]
    guide = (project_root / "docs" / "kaggle.md").read_text(encoding="utf-8")

    assert "--start-shard 0" in guide
    assert "--end-shard 120" in guide
    assert "--start-shard 840" in guide
    assert "--end-shard 964" in guide
    assert "verify-features" in guide
    assert "dirs_exist_ok=True" in guide
    assert "range-00000-00120.json" in guide


def test_kaggle_guide_pins_transformers_and_uses_durable_feature_storage():
    project_root = Path(__file__).parents[1]
    guide = (project_root / "docs" / "kaggle.md").read_text(encoding="utf-8")

    assert "transformers==4.49.0" in guide
    assert "Quick Save 不会保存" in guide
    assert "私有 Kaggle Dataset" in guide


def test_yunet_download_uses_kaggle_safe_url_and_verifies_checksum():
    project_root = Path(__file__).parents[1]
    script = (project_root / "scripts" / "download_yunet.sh").read_text(
        encoding="utf-8"
    )

    assert "media.githubusercontent.com/media/opencv/opencv_zoo" in script
    assert "curl --ipv4" in script
    assert "232589" in script
    assert "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4" in script
    assert "FaceDetectorYN.create" in script
    assert 'TEMP="${OUTPUT}.partial.onnx"' in script
