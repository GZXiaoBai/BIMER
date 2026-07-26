from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Sampler

from .feature_store import FeatureShard, FeatureStore


@dataclass(frozen=True, slots=True)
class LoraTextAdaptationConfig:
    learning_rate: float
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    max_epochs: int = 5
    max_length: int = 128
    contrastive_weight: float = 0.1
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.learning_rate not in {1e-4, 2e-4}:
            raise ValueError("learning_rate must be 1e-4 or 2e-4")
        if (
            self.rank != 8
            or self.alpha != 16
            or self.dropout != 0.1
            or self.max_epochs != 5
            or self.max_length != 128
            or self.contrastive_weight != 0.1
            or self.temperature != 0.07
        ):
            raise ValueError("LoRA adaptation settings must match the frozen V4 protocol")


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, *, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [samples, dimension]")
        if labels.shape != (embeddings.shape[0],):
            raise ValueError("labels must have shape [samples]")
        normalized = F.normalize(embeddings, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity = similarity / self.temperature
        self_mask = torch.eye(
            embeddings.shape[0],
            dtype=torch.bool,
            device=embeddings.device,
        )
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
        valid = positive_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0
        denominator_logits = similarity.masked_fill(self_mask, float("-inf"))
        log_probabilities = similarity - torch.logsumexp(
            denominator_logits,
            dim=1,
            keepdim=True,
        )
        positive_counts = positive_mask.sum(dim=1).clamp_min(1)
        losses = -(log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_counts)
        return losses[valid].mean()


class BalancedTextSampler(Sampler[int]):
    """Deterministically alternate MELD and EmotionTalk text rows."""

    def __init__(self, datasets: Sequence[str], *, seed: int = 42) -> None:
        if not datasets:
            raise ValueError("datasets must not be empty")
        invalid = set(datasets) - {"meld", "emotiontalk"}
        if invalid:
            raise ValueError("datasets must contain only meld or emotiontalk")
        self.datasets = tuple(datasets)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        groups = {
            dataset: [index for index, value in enumerate(self.datasets) if value == dataset]
            for dataset in ("meld", "emotiontalk")
        }
        generator = random.Random(self.seed + self.epoch)
        for indices in groups.values():
            generator.shuffle(indices)
        if not groups["meld"] or not groups["emotiontalk"]:
            indices = groups["meld"] or groups["emotiontalk"]
            return iter(indices)
        target = max(len(groups["meld"]), len(groups["emotiontalk"]))
        balanced = []
        for offset in range(target):
            balanced.append(groups["meld"][offset % len(groups["meld"])])
            balanced.append(groups["emotiontalk"][offset % len(groups["emotiontalk"])])
        return iter(balanced)

    def __len__(self) -> int:
        counts = [self.datasets.count(dataset) for dataset in ("meld", "emotiontalk")]
        return 2 * max(counts) if all(counts) else max(counts)


def replace_text_features(
    shard: FeatureShard,
    replacements: Mapping[str, np.ndarray],
    *,
    expected_dim: int = 768,
) -> FeatureShard:
    if expected_dim <= 0:
        raise ValueError("expected_dim must be positive")
    rows = []
    for sample_id in np.asarray(shard.sample_ids).astype(str).tolist():
        if sample_id not in replacements:
            raise ValueError(f"missing adapted text feature for {sample_id}")
        values = np.asarray(replacements[sample_id], dtype=np.float32)
        if values.shape != (expected_dim,):
            raise ValueError(f"adapted text feature width must be {expected_dim} for {sample_id}")
        if not np.isfinite(values).all():
            raise ValueError(f"adapted text feature must be finite for {sample_id}")
        rows.append(values)
    return FeatureShard(
        sample_ids=np.asarray(shard.sample_ids).copy(),
        text=np.stack(rows).astype(np.float32),
        audio=np.asarray(shard.audio).copy(),
        vision=np.asarray(shard.vision).copy(),
        modality_mask=np.asarray(shard.modality_mask).copy(),
        modality_quality=np.asarray(shard.modality_quality).copy(),
    )


def rewrite_text_feature_store(
    source: FeatureStore,
    destination: FeatureStore,
    *,
    replacements: Mapping[str, np.ndarray],
    partitions: Sequence[tuple[str, str]],
    expected_dim: int = 768,
) -> list[Path]:
    written = []
    for dataset, split in partitions:
        for path in source.paths(dataset, split):
            try:
                shard_index = int(path.stem.rsplit("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid feature shard name {path.name}") from exc
            updated = replace_text_features(
                source.read(path),
                replacements,
                expected_dim=expected_dim,
            )
            written.append(destination.write(dataset, split, shard_index, updated))
    return written


def compose_feature_stores(
    base: FeatureStore,
    destination: FeatureStore,
    *,
    replacements: Mapping[str, FeatureStore],
    partitions: Sequence[tuple[str, str]],
) -> list[Path]:
    """Replace selected modalities while preserving the base store's row identity."""

    modality_indices = {"text": 0, "audio": 1, "vision": 2}
    if not replacements:
        raise ValueError("at least one modality replacement is required")
    invalid = set(replacements) - set(modality_indices)
    if invalid:
        raise ValueError(f"unknown replacement modalities: {sorted(invalid)}")
    written: list[Path] = []
    for dataset, split in partitions:
        base_paths = base.paths(dataset, split)
        if not base_paths:
            raise ValueError(f"base feature store has no {dataset}/{split} shards")
        for base_path in base_paths:
            try:
                shard_index = int(base_path.stem.rsplit("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid feature shard name {base_path.name}") from exc
            base_shard = base.read(base_path)
            features = {
                name: np.asarray(getattr(base_shard, name)).copy() for name in modality_indices
            }
            mask = np.asarray(base_shard.modality_mask).copy()
            quality = np.asarray(base_shard.modality_quality).copy()
            for modality, store in replacements.items():
                replacement_path = store.path(dataset, split, shard_index)
                if not replacement_path.is_file():
                    raise FileNotFoundError(
                        f"replacement feature shard is missing: {replacement_path}"
                    )
                replacement = store.read(replacement_path)
                if not np.array_equal(
                    np.asarray(replacement.sample_ids).astype(str),
                    np.asarray(base_shard.sample_ids).astype(str),
                ):
                    raise ValueError(
                        f"replacement {modality} sample IDs do not align for {dataset}/{split}"
                    )
                index = modality_indices[modality]
                values = np.asarray(getattr(replacement, modality)).copy()
                available = np.asarray(replacement.modality_mask)[:, index].astype(bool)
                values[~available] = 0.0
                features[modality] = values
                mask[:, index] = available
                quality[:, index] = np.asarray(replacement.modality_quality)[:, index]
                quality[~available, index] = 0.0
            written.append(
                destination.write(
                    dataset,
                    split,
                    shard_index,
                    FeatureShard(
                        sample_ids=np.asarray(base_shard.sample_ids).copy(),
                        text=features["text"],
                        audio=features["audio"],
                        vision=features["vision"],
                        modality_mask=mask,
                        modality_quality=quality,
                    ),
                )
            )
    return written
