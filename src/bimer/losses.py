from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


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

