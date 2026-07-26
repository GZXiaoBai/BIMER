from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping

import numpy as np


class RuntimeFeatureCache:
    def __init__(
        self,
        root: Path | str,
        *,
        max_bytes: int = 2 * 1024**3,
        ttl_seconds: float = 24 * 60 * 60,
    ) -> None:
        if max_bytes <= 0 or ttl_seconds <= 0:
            raise ValueError("cache limits must be positive")
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.ttl_seconds = float(ttl_seconds)
        self.root.mkdir(parents=True, exist_ok=True)
        self._evict()

    @staticmethod
    def file_sha256(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def key(namespace: str, payload: Mapping[str, object]) -> str:
        serialized = json.dumps(
            {"namespace": namespace, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.root / f"{key}.npz"

    def load(self, key: str) -> dict[str, np.ndarray] | None:
        path = self._path(key)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                result = {name: archive[name].copy() for name in archive.files}
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        now = time.time()
        os.utime(path, (now, now))
        return result

    def store(self, key: str, arrays: Mapping[str, np.ndarray]) -> Path:
        if not arrays:
            raise ValueError("at least one array is required")
        target = self._path(key)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=self.root,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    **{name: np.asarray(value) for name, value in arrays.items()},
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        self._evict(protect=target)
        return target

    def clear(self) -> int:
        paths = list(self.root.glob("*.npz"))
        for path in paths:
            path.unlink(missing_ok=True)
        return len(paths)

    def _evict(self, *, protect: Path | None = None) -> None:
        now = time.time()
        paths = list(self.root.glob("*.npz"))
        for path in paths:
            try:
                expired = now - path.stat().st_mtime > self.ttl_seconds
            except FileNotFoundError:
                continue
            if expired and path != protect:
                path.unlink(missing_ok=True)
        paths = [path for path in self.root.glob("*.npz") if path.exists()]
        total = sum(path.stat().st_size for path in paths)
        for path in sorted(paths, key=lambda value: value.stat().st_mtime):
            if total <= self.max_bytes:
                break
            if path == protect and len(paths) > 1:
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
