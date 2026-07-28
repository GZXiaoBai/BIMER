from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import FusionOutput, QualityAwareLanguageGatedFusion


class ASRConsistentTextAdapter(nn.Module):
    """Quality-conditioned residual adapter for paired human/ASR text features."""

    def __init__(
        self,
        *,
        text_dim: int = 768,
        quality_dim: int = 4,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if text_dim <= 0 or quality_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.text_dim = text_dim
        self.quality_dim = quality_dim
        self.input_projection = nn.Linear(text_dim + quality_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(bottleneck_dim, text_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, text_features: Tensor, text_quality: Tensor) -> Tensor:
        if text_features.ndim != 3 or text_features.shape[-1] != self.text_dim:
            raise ValueError(f"text_features must have shape [batch, sequence, {self.text_dim}]")
        if text_quality.shape != (*text_features.shape[:2], self.quality_dim):
            raise ValueError(f"text_quality must have shape [batch, sequence, {self.quality_dim}]")
        residual = self.output_projection(
            self.dropout(
                self.activation(
                    self.input_projection(
                        torch.cat(
                            (text_features, text_quality.to(dtype=text_features.dtype)),
                            dim=-1,
                        )
                    )
                )
            )
        )
        return text_features + residual


class ASRConsistentQualityFusion(QualityAwareLanguageGatedFusion):
    """V2 quality fusion with an identity-initialized text consistency adapter."""

    def __init__(
        self,
        *,
        text_dim: int = 768,
        text_quality_dim: int = 4,
        text_adapter_bottleneck_dim: int = 128,
        text_adapter_dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_dim=text_dim, **kwargs)
        self.text_adapter = ASRConsistentTextAdapter(
            text_dim=text_dim,
            quality_dim=text_quality_dim,
            bottleneck_dim=text_adapter_bottleneck_dim,
            dropout=text_adapter_dropout,
        )
        self.text_quality_dim = text_quality_dim

    def forward(
        self,
        *,
        text_features: Tensor,
        audio_features: Tensor,
        vision_features: Tensor,
        modality_mask: Tensor,
        modality_quality: Tensor | None = None,
        attention_mask: Tensor,
        language_ids: Tensor,
    ) -> FusionOutput:
        if modality_quality is None:
            text_quality = torch.zeros(
                *text_features.shape[:2],
                self.text_quality_dim,
                dtype=text_features.dtype,
                device=text_features.device,
            )
        else:
            if modality_quality.shape[:3] != (*text_features.shape[:2], 3):
                raise ValueError(
                    "modality_quality must have shape [batch, sequence, 3, quality_dim]"
                )
            text_quality = modality_quality[:, :, 0]
        adapted_text = self.text_adapter(text_features, text_quality)
        return super().forward(
            text_features=adapted_text,
            audio_features=audio_features,
            vision_features=vision_features,
            modality_mask=modality_mask,
            modality_quality=modality_quality,
            attention_mask=attention_mask,
            language_ids=language_ids,
        )


def jensen_shannon_consistency_loss(
    clean_logits: Tensor,
    corrupted_logits: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Mean symmetric JS divergence over valid paired utterances."""

    if clean_logits.shape != corrupted_logits.shape:
        raise ValueError("clean and corrupted logits must have the same shape")
    if clean_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, classes]")
    if attention_mask.shape != clean_logits.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    valid = attention_mask.bool()
    if not bool(valid.any()):
        return clean_logits.sum() * 0.0 + corrupted_logits.sum() * 0.0

    clean_log = F.log_softmax(clean_logits, dim=-1)
    corrupted_log = F.log_softmax(corrupted_logits, dim=-1)
    clean_probability = clean_log.exp()
    corrupted_probability = corrupted_log.exp()
    midpoint_log = torch.logsumexp(
        torch.stack((clean_log, corrupted_log), dim=0),
        dim=0,
    ) - torch.log(torch.as_tensor(2.0, dtype=clean_logits.dtype, device=clean_logits.device))
    divergence = 0.5 * (
        (clean_probability * (clean_log - midpoint_log)).sum(dim=-1)
        + (corrupted_probability * (corrupted_log - midpoint_log)).sum(dim=-1)
    )
    return divergence[valid].mean()
