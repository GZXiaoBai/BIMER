from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureShard:
    sample_ids: np.ndarray
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    modality_mask: np.ndarray

    def __post_init__(self) -> None:
        validate_feature_shard(self)


def validate_feature_shard(
    shard: FeatureShard,
    expected_dims: Mapping[str, int] | None = None,
) -> None:
    sample_ids = np.asarray(shard.sample_ids).astype(str)
    if sample_ids.ndim != 1:
        raise ValueError("sample_ids must be a vector")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    for name in ("text", "audio", "vision"):
        values = np.asarray(getattr(shard, name))
        if values.ndim != 2:
            raise ValueError(f"{name} must be a matrix")
        if values.shape[0] != len(sample_ids):
            raise ValueError("all feature arrays must have the same row count")
        if expected_dims is not None and values.shape[1] != expected_dims[name]:
            raise ValueError(f"{name} must have width {expected_dims[name]}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} features must be finite")
    if np.asarray(shard.modality_mask).shape != (len(sample_ids), 3):
        raise ValueError("modality_mask must have shape [rows, 3]")


class FeatureStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, dataset: str, split: str, shard_index: int) -> Path:
        return self.root / dataset / split / f"features-{shard_index:05d}.npz"

    def write(
        self,
        dataset: str,
        split: str,
        shard_index: int,
        shard: FeatureShard,
    ) -> Path:
        path = self.path(dataset, split, shard_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    sample_ids=shard.sample_ids.astype(str),
                    text=shard.text.astype(np.float32),
                    audio=shard.audio.astype(np.float32),
                    vision=shard.vision.astype(np.float32),
                    modality_mask=shard.modality_mask.astype(np.bool_),
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def read(self, path: Path | str) -> FeatureShard:
        with np.load(Path(path), allow_pickle=False) as payload:
            return FeatureShard(
                sample_ids=payload["sample_ids"],
                text=payload["text"],
                audio=payload["audio"],
                vision=payload["vision"],
                modality_mask=payload["modality_mask"],
            )

    def paths(self, dataset: str, split: str) -> list[Path]:
        return sorted((self.root / dataset / split).glob("features-*.npz"))

    def read_all(self, dataset: str, split: str) -> list[FeatureShard]:
        return [self.read(path) for path in self.paths(dataset, split)]
