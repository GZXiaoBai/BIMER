from __future__ import annotations

from torch import nn

from .baselines import (
    EarlyFusionContext,
    EarlyFusionMLP,
    MajorityClassifier,
    UnimodalClassifier,
)
from .model import LanguageAwareGatedFusion, QualityAwareLanguageGatedFusion
from .normalization import NormalizedModel
from .v4_model import AdaptiveContextPrototypeFusion


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
    use_quality_input: bool = True,
    use_adaptive_context_gate: bool = True,
    context_gate_override: float | None = None,
    prototype_temperature: float = 0.07,
    majority_class: int = 0,
    use_input_normalization: bool = False,
) -> nn.Module:
    if name == "majority":
        model = MajorityClassifier(majority_class, num_classes=num_classes)
    elif name == "text":
        model = UnimodalClassifier(
            "text", input_dim=text_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    elif name == "audio":
        model = UnimodalClassifier(
            "audio", input_dim=audio_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    elif name == "vision":
        model = UnimodalClassifier(
            "vision", input_dim=vision_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    elif name == "early_mlp":
        model = EarlyFusionMLP(
            (text_dim, audio_dim, vision_dim),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif name == "early_context":
        model = EarlyFusionContext(
            (text_dim, audio_dim, vision_dim),
            hidden_dim=max(1, hidden_dim // 2),
            num_classes=num_classes,
            dropout=dropout,
        )
    elif name in {"lagf", "quality_lagf"}:
        model_class = (
            QualityAwareLanguageGatedFusion if name == "quality_lagf" else LanguageAwareGatedFusion
        )
        model = model_class(
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
            use_quality_input=use_quality_input,
        )
    elif name == "adaptive_context_prototype":
        if not use_context:
            context_gate_override = 0.0
        model = AdaptiveContextPrototypeFusion(
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
            use_quality_input=use_quality_input,
            use_adaptive_context_gate=use_adaptive_context_gate,
            context_gate_override=context_gate_override,
            prototype_temperature=prototype_temperature,
        )
    else:
        raise ValueError(f"Unknown model {name!r}")
    if use_input_normalization:
        return NormalizedModel(model, (text_dim, audio_dim, vision_dim))
    return model
