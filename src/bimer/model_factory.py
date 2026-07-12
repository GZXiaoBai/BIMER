from __future__ import annotations

from torch import nn

from .baselines import (
    EarlyFusionContext,
    EarlyFusionMLP,
    MajorityClassifier,
    UnimodalClassifier,
)
from .model import LanguageAwareGatedFusion


def build_model(
    name: str,
    *,
    text_dim: int = 768,
    audio_dim: int = 1024,
    vision_dim: int = 512,
    hidden_dim: int = 256,
    num_classes: int = 7,
    dropout: float = 0.2,
    modality_dropout: float = 0.2,
    use_language_embedding: bool = True,
    use_reliability_gates: bool = True,
    use_context: bool = True,
    majority_class: int = 0,
) -> nn.Module:
    if name == "majority":
        return MajorityClassifier(majority_class, num_classes=num_classes)
    if name == "text":
        return UnimodalClassifier(
            "text", input_dim=text_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    if name == "audio":
        return UnimodalClassifier(
            "audio", input_dim=audio_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    if name == "vision":
        return UnimodalClassifier(
            "vision", input_dim=vision_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    if name == "early_mlp":
        return EarlyFusionMLP(
            (text_dim, audio_dim, vision_dim),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    if name == "early_context":
        return EarlyFusionContext(
            (text_dim, audio_dim, vision_dim),
            hidden_dim=max(1, hidden_dim // 2),
            num_classes=num_classes,
            dropout=dropout,
        )
    if name == "lagf":
        return LanguageAwareGatedFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            d_model=hidden_dim,
            num_heads=4 if hidden_dim % 4 == 0 else 2,
            transformer_layers=2,
            transformer_ffn_dim=hidden_dim * 2,
            context_hidden_dim=max(1, hidden_dim // 2),
            num_classes=num_classes,
            dropout=dropout,
            modality_dropout=modality_dropout,
            use_language_embedding=use_language_embedding,
            use_reliability_gates=use_reliability_gates,
            use_context=use_context,
        )
    raise ValueError(f"Unknown model {name!r}")
