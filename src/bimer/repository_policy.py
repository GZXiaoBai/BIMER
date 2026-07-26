from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY-----"),
)


@dataclass(frozen=True, slots=True)
class RepositoryViolation:
    code: str
    path: Path
    detail: str


def _forbidden_path(path: Path) -> bool:
    parts = path.parts
    if not parts:
        return False
    if parts[0] in {"artifacts", ".tools", ".superpowers"}:
        return True
    if parts[0] == "data":
        return len(parts) < 2 or parts[1] != "templates"
    return False


def inspect_public_tree(
    *,
    root: Path,
    tracked_paths: Iterable[Path],
    maximum_bytes: int = 10 * 1024**2,
) -> tuple[RepositoryViolation, ...]:
    violations: list[RepositoryViolation] = []
    for relative in sorted(set(tracked_paths), key=lambda value: value.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            violations.append(
                RepositoryViolation(
                    code="invalid-path",
                    path=relative,
                    detail="tracked path must be relative to the repository",
                )
            )
            continue
        if _forbidden_path(relative):
            violations.append(
                RepositoryViolation(
                    code="forbidden-path",
                    path=relative,
                    detail="licensed or private artifacts must not be tracked",
                )
            )
            continue
        path = root / relative
        if not path.is_file():
            violations.append(
                RepositoryViolation(
                    code="missing-file",
                    path=relative,
                    detail="tracked file is missing from the working tree",
                )
            )
            continue
        size = path.stat().st_size
        if size > maximum_bytes:
            violations.append(
                RepositoryViolation(
                    code="oversized-file",
                    path=relative,
                    detail=f"{size} bytes exceeds the {maximum_bytes}-byte limit",
                )
            )
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            violations.append(
                RepositoryViolation(
                    code="secret-pattern",
                    path=relative,
                    detail="file contains a credential-like value",
                )
            )
    return tuple(violations)
