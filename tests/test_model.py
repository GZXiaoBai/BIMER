import torch

from bimer.model import (
    LanguageAwareGatedFusion,
    QualityAwareLanguageGatedFusion,
    apply_modality_dropout,
)


def _inputs():
    torch.manual_seed(7)
    return {
        "text_features": torch.randn(2, 3, 4),
        "audio_features": torch.randn(2, 3, 6),
        "vision_features": torch.randn(2, 3, 5),
        "modality_mask": torch.tensor(
            [
                [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
                [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
            ],
            dtype=torch.bool,
        ),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "language_ids": torch.tensor([0, 1]),
    }


def test_language_aware_model_outputs_logits_and_normalized_gates():
    model = LanguageAwareGatedFusion(
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        d_model=8,
        num_heads=2,
        transformer_layers=1,
        transformer_ffn_dim=16,
        context_hidden_dim=4,
        dropout=0.0,
        modality_dropout=0.0,
    ).eval()
    output = model(**_inputs())
    assert output.logits.shape == (2, 3, 7)
    assert output.gates.shape == (2, 3, 3)
    assert torch.allclose(output.gates[0, 1, 2], torch.tensor(0.0))
    assert torch.allclose(output.gates[0, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(output.gates[0, :3].sum(dim=-1), torch.ones(3))
    assert torch.isfinite(output.logits).all()


def test_modality_dropout_never_removes_every_available_modality():
    mask = torch.ones(64, 4, 3, dtype=torch.bool)
    dropped = apply_modality_dropout(mask, probability=0.95)
    assert torch.all(dropped.sum(dim=-1) >= 1)


def test_modality_dropout_drops_exactly_one_modality_when_selected():
    mask = torch.tensor(
        [[[1, 1, 1], [1, 1, 0], [1, 0, 0]]],
        dtype=torch.bool,
    )

    kept = apply_modality_dropout(mask, probability=1.0)

    assert kept.sum(dim=-1).tolist() == [[2, 1, 1]]
    assert torch.equal(kept[0, 2], mask[0, 2])


def test_ablation_switches_still_produce_finite_predictions():
    model = LanguageAwareGatedFusion(
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        d_model=8,
        num_heads=2,
        transformer_layers=1,
        transformer_ffn_dim=16,
        context_hidden_dim=4,
        dropout=0.0,
        modality_dropout=0.0,
        use_language_embedding=False,
        use_reliability_gates=False,
        use_context=False,
    ).eval()
    output = model(**_inputs())
    assert torch.isfinite(output.logits).all()


def test_cross_modal_transformer_disables_nested_tensor_for_device_compatibility():
    model = LanguageAwareGatedFusion(
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        d_model=8,
        num_heads=2,
        transformer_layers=1,
        transformer_ffn_dim=16,
        context_hidden_dim=4,
    )

    assert model.cross_modal_transformer.enable_nested_tensor is False
    assert model.cross_modal_transformer.use_nested_tensor is False


def test_quality_aware_gate_accepts_zero_quality_without_nan():
    model = QualityAwareLanguageGatedFusion(
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        d_model=8,
        num_heads=2,
        transformer_layers=1,
        transformer_ffn_dim=16,
        context_hidden_dim=4,
        dropout=0.0,
        modality_dropout=0.0,
    ).eval()
    inputs = _inputs()
    inputs["modality_quality"] = torch.zeros(2, 3, 3, 4)

    output = model(**inputs)

    assert model.gate_networks[0][0].in_features == 12
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.gates).all()
    assert torch.allclose(output.gates[0, :3].sum(dim=-1), torch.ones(3))


def test_quality_input_ablation_keeps_outputs_invariant_to_quality_values():
    model = QualityAwareLanguageGatedFusion(
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        d_model=8,
        num_heads=2,
        transformer_layers=1,
        transformer_ffn_dim=16,
        context_hidden_dim=4,
        dropout=0.0,
        modality_dropout=0.0,
        use_quality_input=False,
    ).eval()
    first = _inputs()
    first["modality_quality"] = torch.zeros(2, 3, 3, 4)
    second = _inputs()
    second["modality_quality"] = torch.ones(2, 3, 3, 4)

    with torch.no_grad():
        first_output = model(**first)
        second_output = model(**second)

    torch.testing.assert_close(first_output.gates, second_output.gates)
    torch.testing.assert_close(first_output.logits, second_output.logits)
