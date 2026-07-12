from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .feature_store import FeatureShard, FeatureStore, validate_feature_shard


MODALITY_DIMS = {"text": 768, "audio": 1024, "vision": 512}
STAGING_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ModalityShard:
    sample_ids: np.ndarray
    features: np.ndarray
    available: np.ndarray

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids)
        features = np.asarray(self.features)
        available = np.asarray(self.available)
        if sample_ids.ndim != 1:
            raise ValueError("sample_ids must be a vector")
        if features.ndim != 2:
            raise ValueError("features must be a matrix")
        if features.shape[0] != len(sample_ids):
            raise ValueError("features must have one row per sample")
        if available.shape != (len(sample_ids),):
            raise ValueError("available must have shape [rows]")


class ModalityStore:
    def __init__(
        self,
        root: Path | str,
        modality: str,
        output_dim: int,
    ) -> None:
        if modality not in MODALITY_DIMS:
            raise ValueError(f"unknown modality: {modality}")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.root = Path(root)
        self.modality = modality
        self.output_dim = output_dim

    def path(self, dataset: str, split: str, shard_index: int) -> Path:
        return (
            self.root
            / "staging"
            / dataset
            / split
            / self.modality
            / f"features-{shard_index:05d}.npz"
        )

    def validate(
        self,
        shard: ModalityShard,
        expected_ids: np.ndarray | None = None,
    ) -> None:
        sample_ids = np.asarray(shard.sample_ids).astype(str)
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        if expected_ids is not None and not np.array_equal(
            sample_ids, np.asarray(expected_ids).astype(str)
        ):
            raise ValueError("staging shard has unexpected sample IDs")
        if shard.features.shape != (len(sample_ids), self.output_dim):
            raise ValueError(
                f"features must have shape [rows, {self.output_dim}]"
            )
        if shard.available.shape != (len(sample_ids),):
            raise ValueError("available must have shape [rows]")
        if not np.isfinite(shard.features).all():
            raise ValueError("features must be finite")

    def write(
        self,
        dataset: str,
        split: str,
        shard_index: int,
        shard: ModalityShard,
    ) -> Path:
        self.validate(shard)
        path = self.path(dataset, split, shard_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    schema_version=np.asarray(
                        [STAGING_SCHEMA_VERSION], dtype=np.int16
                    ),
                    modality=np.asarray([self.modality]),
                    output_dim=np.asarray([self.output_dim], dtype=np.int32),
                    sample_ids=np.asarray(shard.sample_ids).astype(str),
                    features=np.asarray(shard.features, dtype=np.float32),
                    available=np.asarray(shard.available, dtype=np.bool_),
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def read(self, path: Path | str) -> ModalityShard:
        with np.load(Path(path), allow_pickle=False) as payload:
            version = int(payload["schema_version"][0])
            modality = str(payload["modality"][0])
            output_dim = int(payload["output_dim"][0])
            if version != STAGING_SCHEMA_VERSION:
                raise ValueError(f"unsupported staging schema version: {version}")
            if modality != self.modality:
                raise ValueError(
                    f"staging modality is {modality}, expected {self.modality}"
                )
            if output_dim != self.output_dim:
                raise ValueError(
                    f"staging output_dim is {output_dim}, expected {self.output_dim}"
                )
            shard = ModalityShard(
                sample_ids=payload["sample_ids"],
                features=payload["features"],
                available=payload["available"],
            )
        self.validate(shard)
        return shard

    def read_verified(
        self,
        dataset: str,
        split: str,
        shard_index: int,
        expected_ids: np.ndarray,
    ) -> ModalityShard:
        shard = self.read(self.path(dataset, split, shard_index))
        self.validate(shard, expected_ids)
        return shard


def verified_final_shard(
    store: FeatureStore,
    dataset: str,
    split: str,
    shard_index: int,
    expected_sample_ids: np.ndarray,
) -> Path | None:
    path = store.path(dataset, split, shard_index)
    if not path.is_file():
        return None
    shard = store.read(path)
    validate_feature_shard(shard, MODALITY_DIMS)
    if not np.array_equal(
        shard.sample_ids.astype(str),
        np.asarray(expected_sample_ids).astype(str),
    ):
        raise ValueError(f"existing shard {path} has unexpected sample IDs")
    return path


def merge_staged_shard(
    *,
    staging_root: Path | str,
    final_store: FeatureStore,
    dataset: str,
    split: str,
    shard_index: int,
    expected_sample_ids: np.ndarray,
) -> Path:
    expected_ids = np.asarray(expected_sample_ids).astype(str)
    existing = verified_final_shard(
        final_store,
        dataset,
        split,
        shard_index,
        expected_ids,
    )
    if existing is not None:
        return existing

    staged = {
        modality: ModalityStore(staging_root, modality, width).read_verified(
            dataset,
            split,
            shard_index,
            expected_ids,
        )
        for modality, width in MODALITY_DIMS.items()
    }
    features: dict[str, np.ndarray] = {}
    for modality, shard in staged.items():
        values = np.asarray(shard.features, dtype=np.float32).copy()
        values[~np.asarray(shard.available, dtype=np.bool_)] = 0.0
        features[modality] = values
    modality_mask = np.stack(
        [
            np.asarray(staged[modality].available, dtype=np.bool_)
            for modality in MODALITY_DIMS
        ],
        axis=1,
    )
    merged = FeatureShard(
        sample_ids=expected_ids,
        text=features["text"],
        audio=features["audio"],
        vision=features["vision"],
        modality_mask=modality_mask,
    )
    validate_feature_shard(merged, MODALITY_DIMS)
    return final_store.write(dataset, split, shard_index, merged)
