from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .feature_store import FeatureShard


@dataclass(frozen=True, slots=True)
class ModalityStatistics:
    mean: np.ndarray
    std: np.ndarray
    count: int


def compute_input_statistics(
    shards: Iterable[FeatureShard],
    *,
    minimum_std: float = 1e-6,
) -> dict[str, ModalityStatistics]:
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    sums: dict[str, np.ndarray] = {}
    squared_sums: dict[str, np.ndarray] = {}
    counts = {name: 0 for name in ("text", "audio", "vision")}
    modality_index = {"text": 0, "audio": 1, "vision": 2}
    for shard in shards:
        for name, index in modality_index.items():
            values = np.asarray(getattr(shard, name), dtype=np.float64)
            active = np.asarray(shard.modality_mask[:, index], dtype=np.bool_)
            selected = values[active]
            if not selected.size:
                continue
            if name not in sums:
                sums[name] = np.zeros(values.shape[1], dtype=np.float64)
                squared_sums[name] = np.zeros(values.shape[1], dtype=np.float64)
            sums[name] += selected.sum(axis=0)
            squared_sums[name] += np.square(selected).sum(axis=0)
            counts[name] += int(selected.shape[0])
    statistics: dict[str, ModalityStatistics] = {}
    for name in modality_index:
        if counts[name] == 0:
            raise ValueError(f"no available {name} rows for normalization")
        mean = sums[name] / counts[name]
        variance = squared_sums[name] / counts[name] - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < minimum_std, 1.0, std)
        statistics[name] = ModalityStatistics(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            count=counts[name],
        )
    return statistics


class InputNormalizer(nn.Module):
    def __init__(self, input_dims: Sequence[int]) -> None:
        super().__init__()
        if len(input_dims) != 3:
            raise ValueError("input_dims must contain text, audio, and vision widths")
        for name, dimension in zip(("text", "audio", "vision"), input_dims, strict=True):
            self.register_buffer(f"{name}_mean", torch.zeros(int(dimension)))
            self.register_buffer(f"{name}_std", torch.ones(int(dimension)))

    def set_statistics(self, statistics: Mapping[str, ModalityStatistics]) -> None:
        with torch.no_grad():
            for name in ("text", "audio", "vision"):
                values = statistics[name]
                mean = torch.as_tensor(values.mean, dtype=getattr(self, f"{name}_mean").dtype)
                std = torch.as_tensor(values.std, dtype=getattr(self, f"{name}_std").dtype)
                if mean.shape != getattr(self, f"{name}_mean").shape:
                    raise ValueError(f"{name} normalization width does not match the model")
                getattr(self, f"{name}_mean").copy_(mean)
                getattr(self, f"{name}_std").copy_(std)

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            (text - self.text_mean) / self.text_std,
            (audio - self.audio_mean) / self.audio_std,
            (vision - self.vision_mean) / self.vision_std,
        )


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, input_dims: Sequence[int]) -> None:
        super().__init__()
        self.model = model
        self.normalizer = InputNormalizer(input_dims)

    def set_statistics(self, statistics: Mapping[str, ModalityStatistics]) -> None:
        self.normalizer.set_statistics(statistics)

    def forward(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        vision_features: Tensor,
        **inputs: Tensor,
    ):
        text, audio, vision = self.normalizer(
            text_features,
            audio_features,
            vision_features,
        )
        return self.model(
            text_features=text,
            audio_features=audio,
            vision_features=vision,
            **inputs,
        )
