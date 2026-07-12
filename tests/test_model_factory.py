import pytest

from bimer.model_factory import build_model


@pytest.mark.parametrize(
    "name",
    ["text", "audio", "vision", "early_mlp", "early_context", "lagf"],
)
def test_model_factory_builds_every_required_experiment(name):
    model = build_model(
        name,
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        hidden_dim=8,
        num_classes=2,
    )
    assert model is not None


def test_model_factory_rejects_unknown_experiment():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("mystery")
