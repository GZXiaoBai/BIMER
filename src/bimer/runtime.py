from __future__ import annotations

import warnings
from pathlib import Path

import torch

from .asr_subprocess import SubprocessWhisperTranscriber
from .calibration import CalibrationProfile
from .deployment import DeploymentManifest, verify_deployment
from .experiment import resolve_device
from .feature_extractors import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    VisionFeatureExtractor,
    YuNetFaceCropper,
)
from .inference import (
    DialogueAnalyzer,
    PretrainedFeaturePipeline,
)
from .model_factory import build_model
from .runtime_cache import RuntimeFeatureCache


class DeploymentNotReadyError(RuntimeError):
    pass


def runtime_devices(device: torch.device) -> tuple[str, str]:
    if device.type == "cuda":
        return "cuda", "cuda"
    if device.type == "mps":
        return "mps", "cpu"
    return "cpu", "cpu"


def build_runtime(
    deployment: DeploymentManifest | Path | str,
    *,
    artifact_root: Path | str = ".",
    device_name: str = "auto",
    offline: bool = True,
) -> DialogueAnalyzer:
    manifest = (
        deployment
        if isinstance(deployment, DeploymentManifest)
        else DeploymentManifest.load(deployment)
    )
    root = Path(artifact_root).resolve()
    verification = verify_deployment(
        manifest,
        artifact_root=root,
        offline=offline,
    )
    if not verification.ok:
        raise DeploymentNotReadyError("; ".join(verification.errors))

    text_reference = manifest.encoders["text"]
    audio_reference = manifest.encoders["audio"]
    vision_reference = manifest.encoders["vision"]
    asr_reference = manifest.encoders["asr"]
    return _assemble_runtime(
        checkpoint_path=root / manifest.checkpoint.path,
        yunet_path=root / manifest.yunet.path,
        device_name=device_name,
        text_model=str(root / text_reference.local_path if offline else text_reference.identifier),
        audio_model=str(
            root / audio_reference.local_path if offline else audio_reference.identifier
        ),
        whisper_model=str(root / asr_reference.local_path if offline else asr_reference.identifier),
        vision_weights_path=(root / vision_reference.local_path if offline else None),
        calibration_path=(
            root / manifest.calibration.path if manifest.calibration is not None else None
        ),
        cache_directory=root / manifest.runtime.cache_directory,
        model_version=manifest.model_version,
        asr_timeout_seconds=manifest.runtime.asr_timeout_seconds,
        encoder_versions={
            name: f"{reference.identifier}@{reference.revision}"
            for name, reference in manifest.encoders.items()
            if name != "asr"
        },
    )


def build_legacy_runtime(
    *,
    checkpoint_path: Path | str,
    yunet_path: Path | str,
    device_name: str = "auto",
    text_model: str = "xlm-roberta-base",
    audio_model: str = "facebook/wav2vec2-xls-r-300m",
    whisper_model: str = "small",
    calibration_path: Path | str | None = None,
    cache_directory: Path | str = "artifacts/runtime-cache",
    model_version: str = "auto",
) -> DialogueAnalyzer:
    return _assemble_runtime(
        checkpoint_path=Path(checkpoint_path),
        yunet_path=Path(yunet_path),
        device_name=device_name,
        text_model=text_model,
        audio_model=audio_model,
        whisper_model=whisper_model,
        vision_weights_path=None,
        calibration_path=(Path(calibration_path) if calibration_path is not None else None),
        cache_directory=Path(cache_directory),
        model_version=model_version,
        asr_timeout_seconds=600,
        encoder_versions={
            "text": text_model,
            "audio": audio_model,
            "vision": "r3d18-yunet-v2",
        },
    )


def _assemble_runtime(
    *,
    checkpoint_path: Path,
    yunet_path: Path,
    device_name: str,
    text_model: str,
    audio_model: str,
    whisper_model: str,
    vision_weights_path: Path | None,
    calibration_path: Path | None,
    cache_directory: Path,
    model_version: str,
    asr_timeout_seconds: int,
    encoder_versions: dict[str, str],
) -> DialogueAnalyzer:
    device = resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint.get("metadata", {}).get("model_config")
    if not model_config:
        raise ValueError("checkpoint does not contain model_config metadata")
    model = build_model(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    extractor_device, whisper_device = runtime_devices(device)

    if model_version == "auto":
        experiment = checkpoint.get("metadata", {}).get("experiment", {})
        if experiment.get("protocol_stage") == "v3_formal":
            model_version = (
                "v3_ranked"
                if float(experiment.get("gate_ranking_weight", 0.0)) > 0
                else "v3_loss_only"
            )
        else:
            model_version = "v2_quality_lagf"
    face_cropper = YuNetFaceCropper(yunet_path)
    cache = RuntimeFeatureCache(cache_directory)

    try:
        pipeline = PretrainedFeaturePipeline(
            text_extractor=TextFeatureExtractor(
                text_model,
                device=extractor_device,
            ),
            audio_extractor=AudioFeatureExtractor(
                audio_model,
                device=extractor_device,
            ),
            vision_extractor=VisionFeatureExtractor(
                device=extractor_device,
                weights_path=vision_weights_path,
            ),
            face_cropper=face_cropper,
            cache=cache,
            encoder_versions=encoder_versions,
        )
    except (RuntimeError, NotImplementedError) as exc:
        if extractor_device != "mps":
            raise
        warnings.warn(
            f"MPS feature extraction unavailable; falling back to CPU: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        pipeline = PretrainedFeaturePipeline(
            text_extractor=TextFeatureExtractor(text_model, device="cpu"),
            audio_extractor=AudioFeatureExtractor(audio_model, device="cpu"),
            vision_extractor=VisionFeatureExtractor(
                device="cpu",
                weights_path=vision_weights_path,
            ),
            face_cropper=face_cropper,
            cache=cache,
            encoder_versions=encoder_versions,
        )

    calibration_profile = (
        CalibrationProfile.load(calibration_path) if calibration_path is not None else None
    )
    return DialogueAnalyzer(
        transcriber=SubprocessWhisperTranscriber(
            whisper_model,
            device=whisper_device,
            timeout_seconds=asr_timeout_seconds,
        ),
        feature_pipeline=pipeline,
        model=model,
        device=device,
        calibration_profile=calibration_profile,
        model_version=model_version,
    )
