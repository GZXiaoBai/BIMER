from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True, slots=True)
class FusionOutput:
    logits: Tensor
    gates: Tensor


def apply_modality_dropout(mask: Tensor, probability: float) -> Tensor:
    """Drop available modalities independently while retaining at least one."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if probability == 0.0:
        return mask.clone()

    available = mask.to(dtype=torch.bool)
    keep = torch.rand(available.shape, device=available.device) >= probability
    dropped = available & keep
    flattened_available = available.reshape(-1, available.shape[-1])
    flattened_dropped = dropped.reshape(-1, dropped.shape[-1])
    for row_index in range(flattened_available.shape[0]):
        candidates = torch.nonzero(flattened_available[row_index], as_tuple=False).flatten()
        if candidates.numel() and not flattened_dropped[row_index].any():
            selected = candidates[
                torch.randint(candidates.numel(), (1,), device=mask.device).item()
            ]
            flattened_dropped[row_index, selected] = True
    return dropped


class LanguageAwareGatedFusion(nn.Module):
    """Language-aware reliability-gated fusion with dialogue context."""

    def __init__(
        self,
        *,
        text_dim: int = 768,
        audio_dim: int = 1024,
        vision_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ffn_dim: int = 512,
        context_hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.2,
        modality_dropout: float = 0.2,
        use_language_embedding: bool = True,
        use_reliability_gates: bool = True,
        use_context: bool = True,
    ) -> None:
        super().__init__()
        self.modality_dropout = modality_dropout
        self.use_language_embedding = use_language_embedding
        self.use_reliability_gates = use_reliability_gates
        self.use_context = use_context

        self.text_projection = nn.Sequential(nn.Linear(text_dim, d_model), nn.LayerNorm(d_model))
        self.audio_projection = nn.Sequential(nn.Linear(audio_dim, d_model), nn.LayerNorm(d_model))
        self.vision_projection = nn.Sequential(nn.Linear(vision_dim, d_model), nn.LayerNorm(d_model))
        self.modality_embedding = nn.Embedding(3, d_model)
        self.language_embedding = nn.Embedding(2, d_model)
        self.gate_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                )
                for _ in range(3)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.context = nn.GRU(
            input_size=d_model,
            hidden_size=context_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        classifier_dim = context_hidden_dim * 2 if use_context else d_model
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
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
        batch_size, sequence_length, _ = text_features.shape
        tokens = torch.stack(
            (
                self.text_projection(text_features),
                self.audio_projection(audio_features),
                self.vision_projection(vision_features),
            ),
            dim=2,
        )
        modality_ids = torch.arange(3, device=tokens.device)
        tokens = tokens + self.modality_embedding(modality_ids).view(1, 1, 3, -1)
        if self.use_language_embedding:
            language = self.language_embedding(language_ids).view(batch_size, 1, 1, -1)
            tokens = tokens + language

        active_mask = modality_mask.bool() & attention_mask.bool().unsqueeze(-1)
        if self.training and self.modality_dropout > 0:
            active_mask = apply_modality_dropout(active_mask, self.modality_dropout)

        flat_tokens = tokens.reshape(batch_size * sequence_length, 3, -1)
        flat_mask = active_mask.reshape(batch_size * sequence_length, 3)
        utterance_is_valid = flat_mask.any(dim=-1)
        safe_mask = flat_mask.clone()
        safe_mask[~utterance_is_valid, 0] = True

        transformed = self.cross_modal_transformer(
            flat_tokens,
            src_key_padding_mask=~safe_mask,
        )
        if self.use_reliability_gates:
            gate_logits = torch.cat(
                [network(flat_tokens[:, index]) for index, network in enumerate(self.gate_networks)],
                dim=-1,
            )
            gate_logits = gate_logits.masked_fill(~safe_mask, torch.finfo(gate_logits.dtype).min)
            gates = torch.softmax(gate_logits, dim=-1)
        else:
            gates = safe_mask.to(dtype=flat_tokens.dtype)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1.0)
        gates = gates * utterance_is_valid.unsqueeze(-1)

        fused = (transformed * gates.unsqueeze(-1)).sum(dim=1)
        fused = fused.reshape(batch_size, sequence_length, -1)
        fused = fused * attention_mask.unsqueeze(-1)

        if self.use_context:
            lengths = attention_mask.sum(dim=1).clamp_min(1).to(dtype=torch.long).cpu()
            packed = pack_padded_sequence(
                fused,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_context, _ = self.context(packed)
            contextualized, _ = pad_packed_sequence(
                packed_context,
                batch_first=True,
                total_length=sequence_length,
            )
        else:
            contextualized = fused

        logits = self.classifier(contextualized)
        logits = logits * attention_mask.unsqueeze(-1)
        return FusionOutput(
            logits=logits,
            gates=gates.reshape(batch_size, sequence_length, 3),
        )
