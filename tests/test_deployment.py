from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from bimer.deployment import DeploymentManifest, verify_deployment
from bimer.labels import EMOTION_LABELS


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_version": "v2_quality_lagf",
        "architecture": "QualityAwareLanguageGatedFusion",
        "seed": 42,
        "labels": list(EMOTION_LABELS),
        "checkpoint": {
            "path": "private/checkpoint.pt",
            "sha256": _sha256(b"checkpoint"),
        },
        "yunet": {
            "path": "private/yunet.onnx",
            "sha256": _sha256(b"yunet"),
        },
        "encoders": {
            "text": {
                "identifier": "xlm-roberta-base",
                "revision": "text-revision",
                "local_path": "private/text",
            },
            "audio": {
                "identifier": "facebook/wav2vec2-xls-r-300m",
                "revision": "audio-revision",
                "local_path": "private/audio",
            },
            "vision": {
                "identifier": "torchvision/r3d_18",
                "revision": "b3b3357e",
                "local_path": "private/r3d.pt",
            },
            "asr": {
                "identifier": "Systran/faster-whisper-small",
                "revision": "asr-revision",
                "local_path": "private/whisper",
            },
        },
        "calibration": None,
        "runtime": {
            "window_size": 32,
            "window_overlap": 8,
            "cache_directory": "artifacts/runtime-cache",
            "minimum_free_bytes": 1,
        },
        "provenance": {},
    }


def _write_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_rejects_absolute_artifact_paths(tmp_path: Path) -> None:
    payload = _manifest_payload()
    payload["checkpoint"] = {
        "path": str(tmp_path / "checkpoint.pt"),
        "sha256": _sha256(b"checkpoint"),
    }

    with pytest.raises(ValueError, match="relative"):
        DeploymentManifest.load(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"labels": ["neutral"]}, "label order"),
        ({"encoders": []}, "encoders must be an object"),
    ],
)
def test_manifest_rejects_incompatible_public_contracts(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    payload = _manifest_payload()
    payload.update(change)

    with pytest.raises(ValueError, match=message):
        DeploymentManifest.load(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        ({"window_size": 0}, "window_size"),
        ({"window_size": 32, "window_overlap": 32}, "window_overlap"),
        ({"minimum_free_bytes": -1}, "minimum_free_bytes"),
    ],
)
def test_manifest_rejects_invalid_runtime_settings(
    tmp_path: Path,
    runtime: dict[str, object],
    message: str,
) -> None:
    payload = _manifest_payload()
    payload["runtime"] = runtime

    with pytest.raises(ValueError, match=message):
        DeploymentManifest.load(_write_manifest(tmp_path, payload))


def test_manifest_rejects_invalid_hash_and_incomplete_provenance(tmp_path: Path) -> None:
    payload = _manifest_payload()
    payload["checkpoint"] = {"path": "checkpoint.pt", "sha256": "not-a-digest"}
    with pytest.raises(ValueError, match="SHA-256"):
        DeploymentManifest.load(_write_manifest(tmp_path, payload))

    payload = _manifest_payload()
    payload["provenance"] = {"selection_config_sha256": _sha256(b"selection")}
    with pytest.raises(ValueError, match="no matching"):
        DeploymentManifest.load(_write_manifest(tmp_path, payload))


def test_offline_verification_checks_hashes_assets_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _manifest_payload()
    manifest = DeploymentManifest.load(_write_manifest(tmp_path, payload))
    private = tmp_path / "private"
    private.mkdir()
    (private / "checkpoint.pt").write_bytes(b"checkpoint")
    (private / "yunet.onnx").write_bytes(b"yunet")
    (private / "text").mkdir()
    (private / "audio").mkdir()
    (private / "whisper").mkdir()
    (private / "r3d.pt").write_bytes(b"r3d")
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        executable = bin_directory / name
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_directory))

    report = verify_deployment(
        manifest,
        artifact_root=tmp_path,
        offline=True,
    )

    assert report.ok
    assert report.checks["checkpoint_sha256"] == "ok"
    assert report.checks["offline_encoder_text"] == "ok"
    assert report.checks["ffmpeg"] == "ok"

    (private / "checkpoint.pt").write_bytes(b"changed")
    broken = verify_deployment(
        manifest,
        artifact_root=tmp_path,
        offline=True,
    )
    assert not broken.ok
    assert "checkpoint SHA-256 mismatch" in broken.errors


def test_offline_verification_reports_missing_encoder_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _manifest_payload()
    manifest = DeploymentManifest.load(_write_manifest(tmp_path, payload))
    private = tmp_path / "private"
    private.mkdir()
    (private / "checkpoint.pt").write_bytes(b"checkpoint")
    (private / "yunet.onnx").write_bytes(b"yunet")
    (private / "audio").mkdir()
    (private / "whisper").mkdir()
    (private / "r3d.pt").write_bytes(b"r3d")
    monkeypatch.setenv("PATH", os.defpath)

    report = verify_deployment(
        manifest,
        artifact_root=tmp_path,
        offline=True,
    )

    assert not report.ok
    assert "offline text encoder asset is missing" in report.errors


def test_verification_checks_provenance_files_declared_in_manifest(
    tmp_path: Path,
) -> None:
    payload = _manifest_payload()
    payload["provenance"] = {
        "selection_config": "private/selection.json",
        "selection_config_sha256": _sha256(b"selection"),
    }
    manifest = DeploymentManifest.load(_write_manifest(tmp_path, payload))
    private = tmp_path / "private"
    private.mkdir()
    (private / "checkpoint.pt").write_bytes(b"checkpoint")
    (private / "yunet.onnx").write_bytes(b"yunet")
    (private / "selection.json").write_bytes(b"selection")

    report = verify_deployment(
        manifest,
        artifact_root=tmp_path,
        offline=False,
    )

    assert report.ok
    assert report.checks["provenance_selection_config_sha256"] == "ok"

    (private / "selection.json").write_bytes(b"changed")
    broken = verify_deployment(
        manifest,
        artifact_root=tmp_path,
        offline=False,
    )
    assert not broken.ok
    assert "selection_config provenance SHA-256 mismatch" in broken.errors


def test_verification_reports_missing_files_tools_and_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _manifest_payload()
    payload["calibration"] = {
        "path": "private/calibration.json",
        "sha256": _sha256(b"calibration"),
    }
    payload["provenance"] = {
        "selection_config": "private/selection.json",
        "selection_config_sha256": _sha256(b"selection"),
    }
    payload["runtime"] = {"minimum_free_bytes": 2**63}
    manifest = DeploymentManifest.load(_write_manifest(tmp_path, payload))
    monkeypatch.setattr("bimer.deployment.shutil.which", lambda _name: None)

    report = verify_deployment(manifest, artifact_root=tmp_path, offline=False)

    assert not report.ok
    assert report.checks["checkpoint_file"] == "missing"
    assert report.checks["calibration_file"] == "missing"
    assert report.checks["provenance_selection_config_sha256"] == "missing"
    assert report.checks["ffmpeg"] == "missing"
    assert report.checks["disk_space"] == "insufficient"
    assert report.to_dict()["ok"] is False
