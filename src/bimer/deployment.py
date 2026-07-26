from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import torch

from .integrity import sha256_file
from .labels import EMOTION_LABELS

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ENCODER_NAMES = ("text", "audio", "vision", "asr")


def _relative_path(value: object, *, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"{field} must be a relative path")
    if ".." in path.parts:
        raise ValueError(f"{field} must not escape the artifact root")
    return path


def _sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: Path
    sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        field: str,
    ) -> ArtifactReference:
        return cls(
            path=_relative_path(value["path"], field=f"{field}.path"),
            sha256=_sha256(value["sha256"], field=f"{field}.sha256"),
        )


@dataclass(frozen=True, slots=True)
class EncoderReference:
    identifier: str
    revision: str
    local_path: Path

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        field: str,
    ) -> EncoderReference:
        identifier = str(value["identifier"]).strip()
        revision = str(value["revision"]).strip()
        if not identifier or not revision:
            raise ValueError(f"{field} identifier and revision must not be empty")
        return cls(
            identifier=identifier,
            revision=revision,
            local_path=_relative_path(
                value["local_path"],
                field=f"{field}.local_path",
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    window_size: int = 32
    window_overlap: int = 8
    cache_directory: Path = Path("artifacts/runtime-cache")
    minimum_free_bytes: int = 2 * 1024**3

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RuntimeSettings:
        settings = cls(
            window_size=int(cast(int, value.get("window_size", 32))),
            window_overlap=int(cast(int, value.get("window_overlap", 8))),
            cache_directory=_relative_path(
                value.get("cache_directory", "artifacts/runtime-cache"),
                field="runtime.cache_directory",
            ),
            minimum_free_bytes=int(cast(int, value.get("minimum_free_bytes", 2 * 1024**3))),
        )
        if settings.window_size <= 0:
            raise ValueError("runtime.window_size must be positive")
        if not 0 <= settings.window_overlap < settings.window_size:
            raise ValueError("runtime.window_overlap must be smaller than window_size")
        if settings.minimum_free_bytes < 0:
            raise ValueError("runtime.minimum_free_bytes must be non-negative")
        return settings


@dataclass(frozen=True, slots=True)
class DeploymentManifest:
    schema_version: int
    model_version: str
    architecture: str
    seed: int
    labels: tuple[str, ...]
    checkpoint: ArtifactReference
    yunet: ArtifactReference
    encoders: Mapping[str, EncoderReference]
    calibration: ArtifactReference | None
    runtime: RuntimeSettings
    provenance: Mapping[str, object]
    source_path: Path

    @classmethod
    def load(cls, path: Path | str) -> DeploymentManifest:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported deployment schema_version")
        labels = tuple(str(label) for label in payload.get("labels", ()))
        if labels != tuple(EMOTION_LABELS):
            raise ValueError("deployment labels must match the public label order")
        encoder_payload = payload.get("encoders")
        if not isinstance(encoder_payload, dict):
            raise ValueError("deployment encoders must be an object")
        if set(encoder_payload) != set(_ENCODER_NAMES):
            raise ValueError("deployment encoders must contain text, audio, vision, and asr")
        encoders = {
            name: EncoderReference.from_mapping(
                encoder_payload[name],
                field=f"encoders.{name}",
            )
            for name in _ENCODER_NAMES
        }
        calibration_payload = payload.get("calibration")
        provenance = dict(payload.get("provenance", {}))
        for key, digest in provenance.items():
            if not key.endswith("_sha256"):
                continue
            path_key = key.removesuffix("_sha256")
            if path_key not in provenance:
                raise ValueError(f"provenance.{key} has no matching provenance.{path_key}")
            _relative_path(
                provenance[path_key],
                field=f"provenance.{path_key}",
            )
            _sha256(digest, field=f"provenance.{key}")
        return cls(
            schema_version=1,
            model_version=str(payload["model_version"]),
            architecture=str(payload["architecture"]),
            seed=int(payload["seed"]),
            labels=labels,
            checkpoint=ArtifactReference.from_mapping(
                payload["checkpoint"],
                field="checkpoint",
            ),
            yunet=ArtifactReference.from_mapping(
                payload["yunet"],
                field="yunet",
            ),
            encoders=encoders,
            calibration=(
                ArtifactReference.from_mapping(
                    calibration_payload,
                    field="calibration",
                )
                if isinstance(calibration_payload, dict)
                else None
            ),
            runtime=RuntimeSettings.from_mapping(payload.get("runtime", {})),
            provenance=provenance,
            source_path=source,
        )


@dataclass(frozen=True, slots=True)
class DeploymentVerification:
    checks: Mapping[str, str]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": dict(self.checks),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def verify_deployment(
    manifest: DeploymentManifest,
    *,
    artifact_root: Path | str,
    offline: bool,
) -> DeploymentVerification:
    root = Path(artifact_root).resolve()
    checks: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for name, reference in (
        ("checkpoint", manifest.checkpoint),
        ("yunet", manifest.yunet),
    ):
        path = root / reference.path
        if not path.is_file():
            checks[f"{name}_file"] = "missing"
            errors.append(f"{name} file is missing")
            continue
        checks[f"{name}_file"] = "ok"
        if sha256_file(path) != reference.sha256:
            checks[f"{name}_sha256"] = "mismatch"
            errors.append(f"{name} SHA-256 mismatch")
        else:
            checks[f"{name}_sha256"] = "ok"

    if manifest.calibration is not None:
        path = root / manifest.calibration.path
        if not path.is_file():
            checks["calibration_file"] = "missing"
            errors.append("calibration file is missing")
        elif sha256_file(path) != manifest.calibration.sha256:
            checks["calibration_sha256"] = "mismatch"
            errors.append("calibration SHA-256 mismatch")
        else:
            checks["calibration_file"] = "ok"
            checks["calibration_sha256"] = "ok"

    for key, value in manifest.provenance.items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        relative = _relative_path(
            manifest.provenance[path_key],
            field=f"provenance.{path_key}",
        )
        path = root / relative
        check_key = f"provenance_{key}"
        if not path.is_file():
            checks[check_key] = "missing"
            errors.append(f"{path_key} provenance file is missing")
        elif sha256_file(path) != str(value):
            checks[check_key] = "mismatch"
            errors.append(f"{path_key} provenance SHA-256 mismatch")
        else:
            checks[check_key] = "ok"

    if offline:
        for name, encoder in manifest.encoders.items():
            path = root / encoder.local_path
            key = f"offline_encoder_{name}"
            if not path.exists():
                checks[key] = "missing"
                errors.append(f"offline {name} encoder asset is missing")
            else:
                checks[key] = "ok"

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable):
            checks[executable] = "ok"
        else:
            checks[executable] = "missing"
            errors.append(f"{executable} executable is missing")

    free_bytes = shutil.disk_usage(root).free
    if free_bytes < manifest.runtime.minimum_free_bytes:
        checks["disk_space"] = "insufficient"
        errors.append("available disk space is below the deployment minimum")
    else:
        checks["disk_space"] = "ok"

    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    checks["mps"] = "available" if mps_available else "unavailable"
    if not mps_available:
        warnings.append("MPS is unavailable; runtime will fall back to CPU")

    return DeploymentVerification(
        checks=checks,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
