from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureShard:
    sample_ids: np.ndarray
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    modality_mask: np.ndarray

    def __post_init__(self) -> None:
        row_counts = {
            len(self.sample_ids),
            self.text.shape[0],
            self.audio.shape[0],
            self.vision.shape[0],
            self.modality_mask.shape[0],
        }
        if len(row_counts) != 1:
            raise ValueError("all feature arrays must have the same row count")
        if self.modality_mask.shape != (len(self.sample_ids), 3):
            raise ValueError("modality_mask must have shape [rows, 3]")


class FeatureStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write(
        self,
        dataset: str,
        split: str,
        shard_index: int,
        shard: FeatureShard,
    ) -> Path:
        directory = self.root / dataset / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"features-{shard_index:05d}.npz"
        np.savez_compressed(
            path,
            sample_ids=shard.sample_ids.astype(str),
            text=shard.text.astype(np.float32),
            audio=shard.audio.astype(np.float32),
            vision=shard.vision.astype(np.float32),
            modality_mask=shard.modality_mask.astype(np.bool_),
        )
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
