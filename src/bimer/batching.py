from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DialogueExample:
    dataset: str
    sample_ids: tuple[str, ...]
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    modality_mask: np.ndarray
    labels: np.ndarray
    language_id: int
    modality_quality: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.sample_ids)
        if not all(
            array.shape[0] == rows
            for array in (
                self.text,
                self.audio,
                self.vision,
                self.modality_mask,
                self.labels,
            )
        ):
            raise ValueError("dialogue feature arrays must share a row count")
        if self.modality_mask.shape != (rows, 3):
            raise ValueError("modality_mask must have shape [utterances, 3]")
        if self.modality_quality is None:
            object.__setattr__(
                self,
                "modality_quality",
                np.repeat(
                    self.modality_mask.astype(np.float32)[..., None],
                    4,
                    axis=-1,
                ),
            )
        if np.asarray(self.modality_quality).shape != (rows, 3, 4):
            raise ValueError("modality_quality must have shape [utterances, 3, 4]")
        if self.language_id not in {0, 1}:
            raise ValueError("language_id must be 0 for English or 1 for Chinese")


@dataclass(slots=True)
class MultimodalBatch:
    text_features: Tensor
    audio_features: Tensor
    vision_features: Tensor
    modality_mask: Tensor
    modality_quality: Tensor
    attention_mask: Tensor
    language_ids: Tensor
    labels: Tensor
    sample_ids: tuple[tuple[str, ...], ...]

    def model_inputs(self) -> dict[str, Tensor]:
        return {
            "text_features": self.text_features,
            "audio_features": self.audio_features,
            "vision_features": self.vision_features,
            "modality_mask": self.modality_mask,
            "modality_quality": self.modality_quality,
            "attention_mask": self.attention_mask,
            "language_ids": self.language_ids,
        }

    def to(self, device: torch.device) -> MultimodalBatch:
        return MultimodalBatch(
            text_features=self.text_features.to(device),
            audio_features=self.audio_features.to(device),
            vision_features=self.vision_features.to(device),
            modality_mask=self.modality_mask.to(device),
            modality_quality=self.modality_quality.to(device),
            attention_mask=self.attention_mask.to(device),
            language_ids=self.language_ids.to(device),
            labels=self.labels.to(device),
            sample_ids=self.sample_ids,
        )


def collate_dialogues(examples: Sequence[DialogueExample]) -> MultimodalBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    max_length = max(len(example.sample_ids) for example in examples)

    def zeros(dimension: int) -> Tensor:
        return torch.zeros(
            batch_size,
            max_length,
            dimension,
            dtype=torch.float32,
        )

    text = zeros(examples[0].text.shape[1])
    audio = zeros(examples[0].audio.shape[1])
    vision = zeros(examples[0].vision.shape[1])
    modality_mask = torch.zeros(
        batch_size,
        max_length,
        3,
        dtype=torch.bool,
    )
    modality_quality = torch.zeros(
        batch_size,
        max_length,
        3,
        4,
        dtype=torch.float32,
    )
    attention_mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.bool,
    )
    labels = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )
    languages = torch.empty(batch_size, dtype=torch.long)
    sample_ids: list[tuple[str, ...]] = []

    for index, example in enumerate(examples):
        length = len(example.sample_ids)
        text[index, :length] = torch.from_numpy(example.text)
        audio[index, :length] = torch.from_numpy(example.audio)
        vision[index, :length] = torch.from_numpy(example.vision)
        modality_mask[index, :length] = torch.from_numpy(example.modality_mask)
        modality_quality[index, :length] = torch.from_numpy(
            np.asarray(example.modality_quality, dtype=np.float32)
        )
        attention_mask[index, :length] = True
        labels[index, :length] = torch.from_numpy(example.labels)
        languages[index] = example.language_id
        sample_ids.append(example.sample_ids)

    return MultimodalBatch(
        text_features=text,
        audio_features=audio,
        vision_features=vision,
        modality_mask=modality_mask,
        modality_quality=modality_quality,
        attention_mask=attention_mask,
        language_ids=languages,
        labels=labels,
        sample_ids=tuple(sample_ids),
    )
