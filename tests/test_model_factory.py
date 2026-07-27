import pytest

from bimer.model_factory import build_model
from bimer.normalization import NormalizedModel


@pytest.mark.parametrize(
    "name",
    [
        "majority",
        "text",
        "audio",
        "vision",
        "early_mlp",
        "early_context",
        "lagf",
        "quality_lagf",
        "adaptive_context_prototype",
    ],
)
def test_model_factory_builds_every_required_experiment(name):
    model = build_model(
        name,
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        hidden_dim=8,
        num_classes=2,
        majority_class=0,
    )
    assert model is not None


def test_model_factory_rejects_unknown_experiment():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("mystery")


def test_model_factory_can_wrap_models_with_checkpointed_input_normalization():
    model = build_model(
        "audio",
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        hidden_dim=8,
        num_classes=2,
        use_input_normalization=True,
    )

    assert isinstance(model, NormalizedModel)
    assert model.normalizer.audio_mean.shape == (6,)


def test_model_factory_builds_v4_with_frozen_context_and_prototype_settings():
    model = build_model(
        "adaptive_context_prototype",
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        hidden_dim=8,
        num_classes=2,
        use_language_embedding=False,
        use_adaptive_context_gate=False,
        context_gate_override=0.0,
        prototype_temperature=0.07,
    )

    assert model.context_mixer.gate_override == 0.0
    assert model.prototypes.shape == (2, 8)
