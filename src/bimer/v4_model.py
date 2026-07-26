from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model import FusionOutput, QualityAwareLanguageGatedFusion


class AdaptiveContextMixer(nn.Module):
    """Mix local utterance evidence with dialogue context through a learned gate."""

    def __init__(
        self,
        *,
        d_model: int,
        context_dim: int,
        gate_override: float | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or context_dim <= 0:
            raise ValueError("representation dimensions must be positive")
        if gate_override is not None and not 0.0 <= gate_override <= 1.0:
            raise ValueError("gate_override must be between 0 and 1")
        self.gate_override = gate_override
        self.context_projection = nn.Linear(context_dim, d_model)
        self.gate_network = nn.Sequential(
            nn.Linear(d_model + context_dim, max(1, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(1, d_model // 2), 1),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        local: Tensor,
        context: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if local.shape[:2] != context.shape[:2] or local.shape[:2] != attention_mask.shape:
            raise ValueError("local, context, and attention_mask dimensions must align")
        if self.gate_override is None:
            gates = torch.sigmoid(self.gate_network(torch.cat((local, context), dim=-1)))
            gates = gates.squeeze(-1)
        else:
            gates = torch.full(
                attention_mask.shape,
                self.gate_override,
                dtype=local.dtype,
                device=local.device,
            )
        gates = gates * attention_mask.to(dtype=local.dtype)
        mixed = self.output_norm(local + gates.unsqueeze(-1) * self.context_projection(context))
        mixed = mixed * attention_mask.unsqueeze(-1)
        return mixed, gates


class AdaptiveContextPrototypeFusion(QualityAwareLanguageGatedFusion):
    """V4 fusion model with adaptive context use and shared emotion prototypes."""

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
        use_language_embedding: bool = False,
        use_reliability_gates: bool = True,
        use_quality_input: bool = True,
        use_adaptive_context_gate: bool = True,
        context_gate_override: float | None = None,
        prototype_temperature: float = 0.07,
    ) -> None:
        if prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if not use_adaptive_context_gate and context_gate_override is None:
            context_gate_override = 1.0
        super().__init__(
            text_dim=text_dim,
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            d_model=d_model,
            num_heads=num_heads,
            transformer_layers=transformer_layers,
            transformer_ffn_dim=transformer_ffn_dim,
            context_hidden_dim=context_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_language_embedding=use_language_embedding,
            use_reliability_gates=use_reliability_gates,
            use_context=False,
            use_quality_input=use_quality_input,
        )
        self.classifier = nn.Identity()
        context_dim = context_hidden_dim * 2
        self.context_mixer = AdaptiveContextMixer(
            d_model=d_model,
            context_dim=context_dim,
            gate_override=context_gate_override,
        )
        self.output_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, d_model))
        nn.init.xavier_uniform_(self.prototypes)
        self.prototype_temperature = prototype_temperature

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
        local_output = super().forward(
            text_features=text_features,
            audio_features=audio_features,
            vision_features=vision_features,
            modality_mask=modality_mask,
            modality_quality=modality_quality,
            attention_mask=attention_mask,
            language_ids=language_ids,
        )
        local = local_output.logits
        lengths = attention_mask.sum(dim=1).clamp_min(1).to(dtype=torch.long).cpu()
        packed = pack_padded_sequence(
            local,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _ = self.context(packed)
        context, _ = pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=local.shape[1],
        )
        representations, context_gates = self.context_mixer(
            local,
            context,
            attention_mask,
        )
        logits = self.output_classifier(representations) * attention_mask.unsqueeze(-1)
        active = attention_mask.unsqueeze(-1)
        local_representations = self.context_mixer.output_norm(local) * active
        fixed_context_representations = (
            self.context_mixer.output_norm(local + self.context_mixer.context_projection(context))
            * active
        )
        local_logits = self.output_classifier(local_representations) * active
        fixed_context_logits = self.output_classifier(fixed_context_representations) * active
        normalized_representations = F.normalize(representations, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        prototype_logits = (
            normalized_representations @ normalized_prototypes.transpose(0, 1)
        ) / self.prototype_temperature
        prototype_logits = prototype_logits * attention_mask.unsqueeze(-1)
        return FusionOutput(
            logits=logits,
            gates=local_output.gates,
            context_gates=context_gates,
            representations=representations,
            prototype_logits=prototype_logits,
            local_logits=local_logits,
            fixed_context_logits=fixed_context_logits,
        )
