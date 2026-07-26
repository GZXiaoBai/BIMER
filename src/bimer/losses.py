from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PrototypeContrastiveLoss(nn.Module):
    """Pull active utterance representations toward their class prototype."""

    def __init__(self, *, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        representations: Tensor,
        prototypes: Tensor,
        labels: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if representations.ndim != 3:
            raise ValueError("representations must have shape [batch, sequence, dimension]")
        if prototypes.ndim != 2 or prototypes.shape[1] != representations.shape[-1]:
            raise ValueError("prototypes must have shape [classes, dimension]")
        if labels.shape != representations.shape[:2]:
            raise ValueError("labels must match batch and sequence dimensions")
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must match labels")
        active = attention_mask.reshape(-1).bool()
        if not active.any():
            raise ValueError("attention_mask contains no active utterances")
        active_representations = F.normalize(
            representations.reshape(-1, representations.shape[-1])[active],
            dim=-1,
        )
        normalized_prototypes = F.normalize(prototypes, dim=-1)
        logits = active_representations @ normalized_prototypes.transpose(0, 1)
        logits = logits / self.temperature
        return F.cross_entropy(logits, labels.reshape(-1)[active])


def sqrt_inverse_class_weights(labels: Tensor, *, num_classes: int) -> Tensor:
    flat = labels.reshape(-1).to(dtype=torch.long)
    counts = torch.bincount(flat, minlength=num_classes).to(dtype=torch.float32)
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].rsqrt()
    if present.any():
        weights[present] = weights[present] / weights[present].mean()
    return weights


def masked_weighted_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    active = attention_mask.reshape(-1).bool()
    if not active.any():
        raise ValueError("attention_mask contains no active utterances")
    flat_logits = logits.reshape(-1, logits.shape[-1])[active]
    flat_labels = labels.reshape(-1)[active]
    return F.cross_entropy(flat_logits, flat_labels, weight=class_weights)


def masked_classification_loss(
    logits: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
    *,
    loss_name: str = "weighted_ce",
    class_weights: Tensor | None = None,
    class_counts: Tensor | None = None,
    focal_gamma: float = 2.0,
) -> Tensor:
    active = attention_mask.reshape(-1).bool()
    if not active.any():
        raise ValueError("attention_mask contains no active utterances")
    flat_logits = logits.reshape(-1, logits.shape[-1])[active]
    flat_labels = labels.reshape(-1)[active]
    if loss_name == "weighted_ce":
        return F.cross_entropy(flat_logits, flat_labels, weight=class_weights)
    if loss_name == "balanced_softmax":
        if class_counts is None:
            raise ValueError("class_counts are required for balanced_softmax")
        counts = class_counts.to(device=flat_logits.device, dtype=flat_logits.dtype)
        if counts.shape != (flat_logits.shape[-1],):
            raise ValueError("class_counts must match the number of classes")
        if torch.any(counts <= 0):
            raise ValueError("class_counts must be positive")
        return F.cross_entropy(flat_logits + counts.log(), flat_labels)
    if loss_name == "focal":
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        cross_entropy = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
        )
        probability = torch.softmax(flat_logits, dim=-1)
        target_probability = probability.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
        losses = ((1.0 - target_probability) ** focal_gamma) * cross_entropy
        if class_weights is None:
            return losses.mean()
        weights = class_weights.to(
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        )[flat_labels]
        return (losses * weights).sum() / weights.sum().clamp_min(
            torch.finfo(flat_logits.dtype).eps
        )
    raise ValueError("loss_name must be weighted_ce, balanced_softmax, or focal")
