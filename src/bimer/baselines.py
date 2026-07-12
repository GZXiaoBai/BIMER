from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model import FusionOutput


def _uniform_gates(modality_mask: Tensor, attention_mask: Tensor) -> Tensor:
    active = modality_mask.bool() & attention_mask.bool().unsqueeze(-1)
    gates = active.to(dtype=torch.float32)
    return gates / gates.sum(dim=-1, keepdim=True).clamp_min(1.0)


class MajorityClassifier(nn.Module):
    def __init__(self, majority_class: int, *, num_classes: int = 7) -> None:
        super().__init__()
        if not 0 <= majority_class < num_classes:
            raise ValueError("majority_class is outside the class range")
        self.majority_class = majority_class
        self.num_classes = num_classes

    def forward(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        vision_features: Tensor,
        modality_mask: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
    ) -> FusionOutput:
        del audio_features, vision_features, language_ids
        logits = torch.zeros(
            *text_features.shape[:2],
            self.num_classes,
            device=text_features.device,
            dtype=text_features.dtype,
        )
        logits[..., self.majority_class] = 1.0
        logits = logits * attention_mask.unsqueeze(-1)
        return FusionOutput(logits=logits, gates=_uniform_gates(modality_mask, attention_mask))


class UnimodalClassifier(nn.Module):
    def __init__(
        self,
        modality: str,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if modality not in {"text", "audio", "vision"}:
            raise ValueError("modality must be text, audio, or vision")
        self.modality = modality
        self.modality_index = {"text": 0, "audio": 1, "vision": 2}[modality]
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        vision_features: Tensor,
        modality_mask: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
    ) -> FusionOutput:
        del language_ids
        features = {
            "text": text_features,
            "audio": audio_features,
            "vision": vision_features,
        }[self.modality]
        active = modality_mask[..., self.modality_index] & attention_mask
        logits = self.classifier(features) * active.unsqueeze(-1)
        gates = torch.zeros(*modality_mask.shape, device=features.device, dtype=features.dtype)
        gates[..., self.modality_index] = active.to(dtype=features.dtype)
        return FusionOutput(logits=logits, gates=gates)


class EarlyFusionMLP(nn.Module):
    def __init__(
        self,
        input_dims: Sequence[int] = (768, 1024, 512),
        *,
        hidden_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(sum(input_dims), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        vision_features: Tensor,
        modality_mask: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
    ) -> FusionOutput:
        del language_ids
        features = torch.cat(
            (
                text_features * modality_mask[..., 0:1],
                audio_features * modality_mask[..., 1:2],
                vision_features * modality_mask[..., 2:3],
            ),
            dim=-1,
        )
        logits = self.classifier(features) * attention_mask.unsqueeze(-1)
        return FusionOutput(logits=logits, gates=_uniform_gates(modality_mask, attention_mask))


class EarlyFusionContext(nn.Module):
    def __init__(
        self,
        input_dims: Sequence[int] = (768, 1024, 512),
        *,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(sum(input_dims), hidden_dim * 2),
            nn.GELU(),
        )
        self.context = nn.GRU(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        vision_features: Tensor,
        modality_mask: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
    ) -> FusionOutput:
        del language_ids
        concatenated = torch.cat(
            (
                text_features * modality_mask[..., 0:1],
                audio_features * modality_mask[..., 1:2],
                vision_features * modality_mask[..., 2:3],
            ),
            dim=-1,
        )
        projected = self.projection(concatenated) * attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1).clamp_min(1).to(dtype=torch.long).cpu()
        packed = pack_padded_sequence(projected, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.context(packed)
        contextualized, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=projected.shape[1],
        )
        logits = self.classifier(contextualized) * attention_mask.unsqueeze(-1)
        return FusionOutput(logits=logits, gates=_uniform_gates(modality_mask, attention_mask))
