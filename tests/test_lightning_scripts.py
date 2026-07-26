from __future__ import annotations

import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).parents[1]


def _run(script: str, *, studio_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "HF_TOKEN": "hf_lightning_test_secret",
            "LIGHTNING_STUDIO_ROOT": str(studio_root),
            "MELD_DOWNLOADER": "hf",
        },
    )


def test_lightning_prepare_uses_persistent_storage_and_removes_archive(tmp_path):
    result = _run("prepare_emotiontalk_lightning.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert str(tmp_path / "data") in result.stdout
    assert str(tmp_path / "output" / "emotiontalk.jsonl") in result.stdout
    assert "Multimodal.tar" in result.stdout
    assert "rm -f" in result.stdout
    assert "hf_lightning_test_secret" not in result.stdout


def test_lightning_fifth_segment_uses_one_gpu_and_first_pending_range(tmp_path):
    result = _run("run_emotiontalk_lightning.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--start-shard 480" in result.stdout
    assert "--end-shard 500" in result.stdout
    assert "--text-audio-device cuda:0" in result.stdout
    assert "--vision-device cuda:0" in result.stdout
    assert "cuda:1" not in result.stdout
    assert "HF_HUB_OFFLINE=1" in result.stdout
    assert "TRANSFORMERS_OFFLINE=1" in result.stdout


def test_lightning_fifth_segment_skips_completed_ranges(tmp_path):
    ranges = tmp_path / "features-emotiontalk-train-v4" / "ranges"
    ranges.mkdir(parents=True)
    (ranges / "range-00480-00500.json").write_text(
        '{"is_valid": true}\n', encoding="utf-8"
    )

    result = _run("run_emotiontalk_lightning.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--start-shard 500" in result.stdout
    assert "--end-shard 520" in result.stdout


def test_lightning_fifth_segment_reports_completion(tmp_path):
    ranges = tmp_path / "features-emotiontalk-train-v4" / "ranges"
    ranges.mkdir(parents=True)
    for start in range(480, 600, 20):
        (ranges / f"range-{start:05d}-{start + 20:05d}.json").write_text(
            '{"is_valid": true}\n', encoding="utf-8"
        )

    result = _run("run_emotiontalk_lightning.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "All Lightning fifth-segment ranges are complete" in result.stdout


def test_lightning_setup_installs_without_replacing_preinstalled_torch():
    source = (PROJECT_ROOT / "scripts" / "setup_lightning_ai.sh").read_text(
        encoding="utf-8"
    )

    assert "pip install --no-deps --editable" in source
    assert 'transformers==4.49.0' in source
    assert "pip install torch" not in source
    assert "pip install torchvision" not in source


def test_lightning_setup_uses_transformers_compatible_huggingface_hub():
    source = (PROJECT_ROOT / "scripts" / "setup_lightning_ai.sh").read_text(
        encoding="utf-8"
    )

    assert "huggingface_hub[hf_xet]>=0.30,<1.0" in source
    assert "huggingface_hub[hf_xet]==1.23.0" not in source


def test_autodl_pipeline_queues_smoke_all_ranges_and_final_verification(tmp_path):
    result = _run("run_autodl_fifth_segment.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--start-shard 480" in result.stdout
    assert "--end-shard 482" in result.stdout
    assert result.stdout.count("run_emotiontalk_lightning.sh") == 6
    assert "--end-shard 600" in result.stdout
    assert "verify-features" in result.stdout
    assert "AUTODL_PIPELINE_COMPLETE" in result.stdout


def test_autodl_remaining_train_queues_all_ranges_and_clamps_final_range(tmp_path):
    result = _run("run_autodl_remaining_train.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "WAIT-FOR range-00480-00600.json" in result.stdout
    assert "--start-shard 600" in result.stdout
    assert "--end-shard 620" in result.stdout
    assert "--start-shard 960" in result.stdout
    assert "--end-shard 964" in result.stdout
    assert "--end-shard 980" not in result.stdout
    assert result.stdout.count("extract-features") == 19
    assert "HF_HUB_OFFLINE=1" in result.stdout
    assert "TRANSFORMERS_OFFLINE=1" in result.stdout
    assert "AUTODL_REMAINING_TRAIN_COMPLETE" in result.stdout


def test_autodl_eval_splits_queue_official_validation_and_test_ranges(tmp_path):
    result = _run("run_autodl_emotiontalk_eval_splits.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--split validation" in result.stdout
    assert "--start-shard 0" in result.stdout
    assert "--end-shard 120" in result.stdout
    assert "--split test" in result.stdout
    assert "--end-shard 121" in result.stdout
    assert "--end-shard 140" not in result.stdout
    assert result.stdout.count("extract-features") == 13
    assert "HF_HUB_OFFLINE=1" in result.stdout
    assert "TRANSFORMERS_OFFLINE=1" in result.stdout
    assert "feature-stats" in result.stdout
    assert "AUTODL_EMOTIONTALK_EVAL_COMPLETE" in result.stdout


def test_autodl_eval_splits_skip_completed_small_ranges(tmp_path):
    ranges = tmp_path / "features-emotiontalk-validation-v4" / "ranges"
    ranges.mkdir(parents=True)
    (ranges / "range-00000-00020.json").write_text(
        '{"is_valid": true}\n', encoding="utf-8"
    )

    result = _run("run_autodl_emotiontalk_eval_splits.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "SKIP validation completed range [0,20)" in result.stdout
    assert result.stdout.count("extract-features") == 12


def test_autodl_meld_prepare_downloads_raw_data_and_builds_official_manifest(tmp_path):
    result = _run("prepare_meld_autodl.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "hf download declare-lab/MELD MELD.Raw.tar.gz" in result.stdout
    assert "MELD.Features.Models.tar.gz" not in result.stdout
    assert "prepare-meld" in result.stdout
    assert "train_sent_emo.csv" in result.stdout
    assert "dev_sent_emo.csv" in result.stdout
    assert "test_sent_emo.csv" in result.stdout
    assert "validate" in result.stdout
    assert "--official-counts" in result.stdout
    assert "hf_lightning_test_secret" not in result.stdout


def test_autodl_meld_prepare_downloads_pinned_official_annotations(tmp_path):
    result = _run("prepare_meld_autodl.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    commit = "e8cedf27b5d2877e198332c957127e16eb214afe"
    base = f"https://raw.githubusercontent.com/declare-lab/MELD/{commit}/data/MELD"
    for split in ("train", "dev", "test"):
        assert f"{base}/{split}_sent_emo.csv" in result.stdout
    assert "d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db" in result.stdout
    assert "2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c" in result.stdout
    assert "8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888" in result.stdout


def test_autodl_meld_prepare_batches_flat_symlinks_with_find_placeholder_last(tmp_path):
    result = _run("prepare_meld_autodl.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    link_commands = [
        line for line in result.stdout.splitlines() if "-exec ln" in line
    ]
    assert len(link_commands) == 3
    for command in link_commands:
        assert "ln -sfn -t" in command
        assert command.endswith("\\{\\} +")


def test_autodl_meld_prepare_uses_verified_raw_media_counts():
    source = (PROJECT_ROOT / "scripts" / "prepare_meld_autodl.sh").read_text(
        encoding="utf-8"
    )

    # The checksum-pinned official raw archive contains four unannotated dev
    # clips and omits annotated dia110_utt7.mp4.
    assert "expected_media_counts=(9989 1112 2615)" in source


def test_autodl_meld_prepare_disk_check_uses_integer_bytes(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_hf = fake_bin / "hf"
    fake_hf.write_text("#!/usr/bin/env bash\nexit 77\n", encoding="utf-8")
    fake_hf.chmod(0o755)
    fake_awk = fake_bin / "awk"
    fake_awk.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *\"* 1024\"* ]]; then\n"
        "  printf '8.18968e+10\\n'\n"
        "else\n"
        "  /usr/bin/awk \"$@\"\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_awk.chmod(0o755)

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "prepare_meld_autodl.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "AUTODL_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "MELD_REQUIRED_FREE_BYTES": "1",
            "MELD_DOWNLOADER": "hf",
        },
    )

    assert result.returncode == 77
    assert "syntax error" not in result.stderr


def test_autodl_meld_prepare_enables_autodl_academic_network(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"${https_proxy:-}\" == 'http://turbo.test:1234' ]] || exit 78\n"
        "exit 77\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    network_turbo = tmp_path / "network_turbo"
    network_turbo.write_text(
        "export http_proxy='http://turbo.test:1234'\n"
        "export https_proxy='http://turbo.test:1234'\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "prepare_meld_autodl.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "AUTODL_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "MELD_REQUIRED_FREE_BYTES": "1",
            "NETWORK_TURBO_PATH": str(network_turbo),
            "MELD_DOWNLOADER": "hf",
        },
    )

    assert result.returncode == 77, result.stderr


def test_autodl_meld_prepare_prefers_parallel_download_with_official_checksum(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_aria2 = fake_bin / "aria2c"
    fake_aria2.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_aria2.chmod(0o755)

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "prepare_meld_autodl.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "DRY_RUN": "1",
            "AUTODL_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "MELD_DOWNLOADER": "aria2",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "aria2c" in result.stdout
    assert "--max-connection-per-server=16" in result.stdout
    assert "--split=16" in result.stdout
    assert (
        "--checksum=sha-256=a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e"
        in result.stdout
    )
    assert "huggingface.co/datasets/declare-lab/MELD" in result.stdout
    assert "hf download" not in result.stdout


def test_autodl_meld_features_queue_all_official_splits_and_clamp_ranges(tmp_path):
    result = _run("run_autodl_meld_features.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--dataset meld" in result.stdout
    assert "--split train" in result.stdout
    assert "--end-shard 625" in result.stdout
    assert "--end-shard 640" not in result.stdout
    assert "--split dev" in result.stdout
    assert "--end-shard 70" in result.stdout
    assert "--split test" in result.stdout
    assert "--end-shard 164" in result.stdout
    assert result.stdout.count("extract-features") == 45
    assert "HF_HUB_OFFLINE=1" in result.stdout
    assert "TRANSFORMERS_OFFLINE=1" in result.stdout
    assert "AUTODL_MELD_FEATURES_COMPLETE" in result.stdout


def test_autodl_audio_robustness_recomputes_only_audio_for_two_snr_levels(
    tmp_path,
):
    result = _run("run_autodl_audio_robustness.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("extract-features") == 2
    assert result.stdout.count("--only-modality audio") == 2
    assert "--audio-snr 20" in result.stdout
    assert "--audio-snr 10" in result.stdout
    assert "--condition-name audio_snr_20db" in result.stdout
    assert "--condition-name audio_snr_10db" in result.stdout
    assert "--base-features" in result.stdout
    assert "--split test" in result.stdout
    assert "OMP_NUM_THREADS=1" in result.stdout
    assert "AUTODL_AUDIO_ROBUSTNESS_COMPLETE" in result.stdout


def test_autodl_video_robustness_recomputes_only_vision_for_two_drop_levels(
    tmp_path,
):
    result = _run("run_autodl_video_robustness.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("extract-features") == 2
    assert result.stdout.count("--only-modality vision") == 2
    assert "--frame-drop 0.25" in result.stdout
    assert "--frame-drop 0.50" in result.stdout
    assert "--condition-name video_frame_drop_25pct" in result.stdout
    assert "--condition-name video_frame_drop_50pct" in result.stdout
    assert "--base-features" in result.stdout
    assert "--split test" in result.stdout
    assert "OMP_NUM_THREADS=1" in result.stdout
    assert "AUTODL_VIDEO_ROBUSTNESS_COMPLETE" in result.stdout


def test_autodl_whisper_robustness_replaces_only_text_and_logs_failures(
    tmp_path,
):
    result = _run("run_autodl_whisper_robustness.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("asr-manifest") == 1
    assert "--keep-original-on-error" in result.stdout
    assert "--error-log" in result.stdout
    assert "--device cuda" in result.stdout
    assert result.stdout.count("extract-features") == 1
    assert "--only-modality text" in result.stdout
    assert "--condition-name whisper_text" in result.stdout
    assert "--base-features" in result.stdout
    assert "--split test" in result.stdout
    assert "OMP_NUM_THREADS=1" in result.stdout
    assert "AUTODL_WHISPER_ROBUSTNESS_COMPLETE" in result.stdout


def test_autodl_robustness_evaluation_runs_all_conditions_and_seeds(tmp_path):
    result = _run("run_autodl_robustness_evaluation.sh", studio_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count(" evaluate ") == 15
    for condition in (
        "audio_snr_20db",
        "audio_snr_10db",
        "video_frame_drop_25pct",
        "video_frame_drop_50pct",
        "whisper_text",
    ):
        assert f"--condition-name {condition}" in result.stdout
    for seed in (42, 123, 2026):
        assert f"seed-{seed}/best.pt" in result.stdout
    assert "whisper-test.jsonl" in result.stdout
    assert "aggregate_seed_results" in result.stdout
    assert result.stdout.count("/summary.json") == 5
    assert "AUTODL_ROBUSTNESS_EVALUATION_COMPLETE" in result.stdout
