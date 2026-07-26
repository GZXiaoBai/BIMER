from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Sha256Entry:
    path: str
    sha256: str


@dataclass(frozen=True)
class VerificationResult:
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside root: {path}") from exc


def _iter_files(inputs: Iterable[Path]) -> Iterable[Path]:
    for path in inputs:
        if path.is_dir():
            yield from (candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(path)


def build_sha256_manifest(*, root: Path, inputs: Sequence[Path]) -> list[Sha256Entry]:
    by_path: dict[str, Path] = {}
    for path in _iter_files(inputs):
        relative = _portable_relative_path(path, root)
        by_path[relative] = path
    return [
        Sha256Entry(path=relative, sha256=sha256_file(by_path[relative]))
        for relative in sorted(by_path)
    ]


def write_sha256_manifest(
    *,
    destination: Path,
    root: Path,
    inputs: Sequence[Path],
    header: Sequence[str] = (),
) -> None:
    entries = build_sha256_manifest(root=root, inputs=inputs)
    lines = [*(f"# {line}" for line in header)]
    lines.extend(f"{entry.sha256}  {entry.path}" for entry in entries)
    payload = "\n".join(lines) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def verify_sha256_manifest(*, manifest: Path, root: Path) -> VerificationResult:
    missing: list[str] = []
    mismatched: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", maxsplit=1)
        target = root / relative
        if not target.is_file():
            missing.append(relative)
        elif sha256_file(target) != expected:
            mismatched.append(relative)
    return VerificationResult(
        missing=tuple(missing),
        mismatched=tuple(mismatched),
    )
