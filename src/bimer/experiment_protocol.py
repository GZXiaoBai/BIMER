from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


class ExperimentProtocolConfig(Protocol):
    @property
    def protocol_stage(self) -> str: ...

    @property
    def seed(self) -> int: ...

    @property
    def evaluate_test(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    stage: str
    version: str
    seed: int
    evaluate_test: bool

    @classmethod
    def from_config(cls, config: ExperimentProtocolConfig) -> ProtocolSpec:
        stage = str(config.protocol_stage)
        allowed = {
            "standard",
            "v3_screen",
            "v3_formal",
            "v4_screen",
            "v4_formal",
            "v5_screen",
            "v5_formal",
        }
        if stage not in allowed:
            raise ValueError("unknown experiment protocol_stage")
        is_screen = stage.endswith("_screen")
        is_formal = stage.endswith("_formal")
        if is_screen and int(config.seed) != 42:
            raise ValueError(f"{stage} is restricted to seed 42")
        if (is_screen or is_formal) and bool(config.evaluate_test):
            raise ValueError(f"{stage} must use --skip-test")
        version = stage.split("_", 1)[0] if stage != "standard" else "v2"
        return cls(
            stage=stage,
            version=version,
            seed=int(config.seed),
            evaluate_test=bool(config.evaluate_test),
        )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


class ExperimentProtocolRunner:
    """Own protocol validation, resumable status, and failure evidence."""

    def __init__(
        self,
        spec: ProtocolSpec,
        *,
        status_path: Path | str,
        result_path: Path | str | None = None,
    ) -> None:
        self.spec = spec
        self.status_path = Path(status_path)
        self.result_path = Path(result_path) if result_path is not None else None

    def run(self, operation: Callable[[], T]) -> T | Path:
        if self.result_path is not None and self.result_path.is_file():
            if self.status_path.is_file():
                status = json.loads(self.status_path.read_text(encoding="utf-8"))
                if status.get("status") == "completed":
                    return self.result_path
        base = {
            "protocol_stage": self.spec.stage,
            "version": self.spec.version,
            "seed": self.spec.seed,
            "evaluate_test": self.spec.evaluate_test,
        }
        _atomic_json(self.status_path, {**base, "status": "running"})
        try:
            result = operation()
        except Exception as exc:
            _atomic_json(
                self.status_path,
                {
                    **base,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        _atomic_json(
            self.status_path,
            {
                **base,
                "status": "completed",
                "result": str(self.result_path or result),
            },
        )
        return result


def run_guarded_exploratory_test(
    *,
    version: str,
    selection_path: Path | str,
    marker_path: Path | str,
    evaluator: Callable[[], T],
) -> T:
    selection_source = Path(selection_path)
    selection = json.loads(selection_source.read_text(encoding="utf-8"))
    if selection.get("state") != "frozen" or selection.get("version") != version:
        raise RuntimeError(f"{version.upper()} selection configuration is not frozen")
    marker = Path(marker_path)
    if marker.exists():
        raise RuntimeError(f"{version.upper()} official test has already been evaluated")
    marker.parent.mkdir(parents=True, exist_ok=True)
    claim = marker.with_name(f"{marker.name}.RUNNING")
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"{version.upper()} official test evaluation is already running"
        ) from exc
    os.close(descriptor)
    try:
        result = evaluator()
        _atomic_json(
            marker,
            {
                "status": "evaluated",
                "version": version,
                "selection": str(selection_source.resolve()),
            },
        )
        return result
    except Exception as exc:
        _atomic_json(
            marker,
            {
                "status": "failed",
                "version": version,
                "selection": str(selection_source.resolve()),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        claim.unlink(missing_ok=True)
