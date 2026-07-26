from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import bimer.runtime as runtime_module
from bimer.labels import EMOTION_LABELS
from bimer.runtime import (
    DeploymentNotReadyError,
    build_legacy_runtime,
    build_runtime,
    runtime_devices,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_runtime_devices_keep_whisper_on_cpu_for_mps() -> None:
    assert runtime_devices(torch.device("mps")) == ("mps", "cpu")
    assert runtime_devices(torch.device("cuda")) == ("cuda", "cuda")
    assert runtime_devices(torch.device("cpu")) == ("cpu", "cpu")


def test_build_runtime_rejects_incomplete_deployment_before_model_loading(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "model_version": "v2_quality_lagf",
        "architecture": "QualityAwareLanguageGatedFusion",
        "seed": 42,
        "labels": list(EMOTION_LABELS),
        "checkpoint": {
            "path": "missing.pt",
            "sha256": _digest(b"checkpoint"),
        },
        "yunet": {
            "path": "missing.onnx",
            "sha256": _digest(b"yunet"),
        },
        "encoders": {
            name: {
                "identifier": name,
                "revision": "revision",
                "local_path": f"models/{name}",
            }
            for name in ("text", "audio", "vision", "asr")
        },
        "calibration": None,
        "runtime": {
            "cache_directory": "cache",
            "minimum_free_bytes": 0,
        },
        "provenance": {},
    }
    deployment = tmp_path / "deployment.json"
    deployment.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DeploymentNotReadyError, match="checkpoint file is missing"):
        build_runtime(
            deployment,
            artifact_root=tmp_path,
            device_name="cpu",
            offline=False,
        )


def test_runtime_assembly_passes_the_resolved_model_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def load_state_dict(self, _state) -> None:
            pass

    class FakeExtractor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    checkpoint = {
        "metadata": {
            "model_config": {"model": "quality_lagf"},
            "experiment": {"protocol_stage": "standard"},
        },
        "model_state_dict": {},
    }
    monkeypatch.setattr(runtime_module.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(runtime_module, "build_model", lambda **_kwargs: FakeModel())
    monkeypatch.setattr(runtime_module, "TextFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "AudioFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "VisionFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "YuNetFaceCropper", FakeExtractor)
    monkeypatch.setattr(runtime_module, "RuntimeFeatureCache", FakeExtractor)
    monkeypatch.setattr(runtime_module, "PretrainedFeaturePipeline", FakeExtractor)
    monkeypatch.setattr(runtime_module, "FasterWhisperTranscriber", FakeExtractor)
    monkeypatch.setattr(
        runtime_module,
        "DialogueAnalyzer",
        lambda **kwargs: kwargs,
    )

    assembled = build_legacy_runtime(
        checkpoint_path=tmp_path / "checkpoint.pt",
        yunet_path=tmp_path / "yunet.onnx",
        device_name="cpu",
        model_version="auto",
    )

    assert assembled["model_version"] == "v2_quality_lagf"


def test_runtime_assembly_resolves_ranked_v3_and_loads_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def load_state_dict(self, _state) -> None:
            pass

    class FakeExtractor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    checkpoint = {
        "metadata": {
            "model_config": {"model": "quality_lagf"},
            "experiment": {
                "protocol_stage": "v3_formal",
                "gate_ranking_weight": 0.1,
            },
        },
        "model_state_dict": {},
    }
    monkeypatch.setattr(runtime_module.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(runtime_module, "build_model", lambda **_kwargs: FakeModel())
    monkeypatch.setattr(runtime_module, "TextFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "AudioFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "VisionFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "YuNetFaceCropper", FakeExtractor)
    monkeypatch.setattr(runtime_module, "RuntimeFeatureCache", FakeExtractor)
    monkeypatch.setattr(runtime_module, "PretrainedFeaturePipeline", FakeExtractor)
    monkeypatch.setattr(runtime_module, "FasterWhisperTranscriber", FakeExtractor)
    monkeypatch.setattr(runtime_module, "DialogueAnalyzer", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        runtime_module.CalibrationProfile,
        "load",
        lambda path: {"loaded_from": path},
    )

    assembled = build_legacy_runtime(
        checkpoint_path=tmp_path / "checkpoint.pt",
        yunet_path=tmp_path / "yunet.onnx",
        device_name="cpu",
        calibration_path=tmp_path / "calibration.json",
        model_version="auto",
    )

    assert assembled["model_version"] == "v3_ranked"
    assert assembled["calibration_profile"] == {"loaded_from": tmp_path / "calibration.json"}


def test_runtime_assembly_falls_back_to_cpu_when_mps_extractor_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def load_state_dict(self, _state) -> None:
            pass

    devices: list[str] = []

    class FakeExtractor:
        def __init__(self, *_args, device: str = "cpu", **_kwargs) -> None:
            devices.append(device)

    class FailingPipeline:
        attempts = 0

        def __init__(self, **_kwargs) -> None:
            self.__class__.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("unsupported MPS operation")

    checkpoint = {
        "metadata": {"model_config": {"model": "quality_lagf"}},
        "model_state_dict": {},
    }
    monkeypatch.setattr(runtime_module, "resolve_device", lambda _name: torch.device("mps"))
    monkeypatch.setattr(runtime_module.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(runtime_module, "build_model", lambda **_kwargs: FakeModel())
    monkeypatch.setattr(runtime_module, "TextFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "AudioFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "VisionFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(runtime_module, "YuNetFaceCropper", FakeExtractor)
    monkeypatch.setattr(runtime_module, "RuntimeFeatureCache", FakeExtractor)
    monkeypatch.setattr(runtime_module, "PretrainedFeaturePipeline", FailingPipeline)
    monkeypatch.setattr(runtime_module, "FasterWhisperTranscriber", FakeExtractor)
    monkeypatch.setattr(runtime_module, "DialogueAnalyzer", lambda **kwargs: kwargs)

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assembled = build_legacy_runtime(
            checkpoint_path=tmp_path / "checkpoint.pt",
            yunet_path=tmp_path / "yunet.onnx",
            device_name="mps",
        )

    assert FailingPipeline.attempts == 2
    assert devices.count("mps") == 3
    assert devices.count("cpu") >= 4
    assert assembled["device"] == torch.device("mps")


def test_runtime_assembly_rejects_checkpoint_without_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.torch,
        "load",
        lambda *_args, **_kwargs: {"metadata": {}, "model_state_dict": {}},
    )

    with pytest.raises(ValueError, match="model_config"):
        build_legacy_runtime(
            checkpoint_path=tmp_path / "checkpoint.pt",
            yunet_path=tmp_path / "yunet.onnx",
            device_name="cpu",
        )
