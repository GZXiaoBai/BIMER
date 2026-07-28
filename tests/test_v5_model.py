from __future__ import annotations

import pytest
import torch

from bimer.model_factory import build_model
from bimer.v5_model import (
    ASRConsistentTextAdapter,
    jensen_shannon_consistency_loss,
)


def test_asr_adapter_is_identity_initialized_and_learns_a_residual() -> None:
    adapter = ASRConsistentTextAdapter(
        text_dim=8,
        quality_dim=4,
        bottleneck_dim=4,
        dropout=0.0,
    )
    text = torch.randn(2, 3, 8, requires_grad=True)
    quality = torch.randn(2, 3, 4)

    output = adapter(text, quality)

    torch.testing.assert_close(output, text)
    output.square().mean().backward()
    assert adapter.output_projection.weight.grad is not None
    assert torch.isfinite(adapter.output_projection.weight.grad).all()


def test_asr_adapter_validates_text_quality_shape() -> None:
    adapter = ASRConsistentTextAdapter(text_dim=8, bottleneck_dim=4)
    with pytest.raises(ValueError, match="text_quality"):
        adapter(torch.randn(1, 2, 8), torch.randn(1, 2, 3))
    with pytest.raises(ValueError, match="text_features"):
        adapter(torch.randn(1, 2, 7), torch.randn(1, 2, 4))


def test_v5_model_factory_preserves_fusion_contract_with_missing_modalities() -> None:
    model = build_model(
        "asr_consistent_quality_lagf",
        text_dim=8,
        audio_dim=6,
        vision_dim=5,
        hidden_dim=8,
        num_classes=3,
        dropout=0.0,
        modality_dropout=0.0,
        use_language_embedding=False,
    )
    mask = torch.tensor([[[True, False, False], [True, True, False]]])
    output = model(
        text_features=torch.randn(1, 2, 8),
        audio_features=torch.randn(1, 2, 6),
        vision_features=torch.randn(1, 2, 5),
        modality_mask=mask,
        modality_quality=torch.ones(1, 2, 3, 4),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        language_ids=torch.zeros(1, dtype=torch.long),
    )

    assert output.logits.shape == (1, 2, 3)
    assert output.gates.shape == (1, 2, 3)
    assert torch.isfinite(output.logits).all()


def test_js_consistency_is_symmetric_zero_for_equal_logits_and_ignores_padding() -> None:
    clean = torch.tensor([[[2.0, 0.0], [100.0, -100.0]]], requires_grad=True)
    whisper = torch.tensor([[[0.0, 2.0], [-100.0, 100.0]]], requires_grad=True)
    mask = torch.tensor([[True, False]])

    forward = jensen_shannon_consistency_loss(clean, whisper, mask)
    reverse = jensen_shannon_consistency_loss(whisper, clean, mask)
    equal = jensen_shannon_consistency_loss(clean, clean, mask)
    forward.backward()

    assert forward.item() > 0
    torch.testing.assert_close(forward, reverse)
    assert equal.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isfinite(clean.grad).all()


def test_js_consistency_validates_shapes_and_empty_masks() -> None:
    logits = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="same shape"):
        jensen_shannon_consistency_loss(logits, torch.randn(1, 1, 3), torch.ones(1, 2))
    with pytest.raises(ValueError, match="attention_mask"):
        jensen_shannon_consistency_loss(logits, logits, torch.ones(1, 1))
    zero = jensen_shannon_consistency_loss(
        logits.requires_grad_(),
        logits.detach().clone().requires_grad_(),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    zero.backward()
    assert zero.item() == 0.0
