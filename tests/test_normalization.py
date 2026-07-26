import numpy as np
import torch

from bimer.feature_store import FeatureShard
from bimer.normalization import InputNormalizer, compute_input_statistics


def test_input_statistics_ignore_unavailable_modality_rows():
    shard = FeatureShard(
        sample_ids=np.array(["one", "two"]),
        text=np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32),
        audio=np.array([[2.0, 4.0], [4.0, 8.0]], dtype=np.float32),
        vision=np.array([[1.0, 2.0], [1000.0, 1000.0]], dtype=np.float32),
        modality_mask=np.array([[1, 1, 1], [1, 1, 0]], dtype=np.bool_),
    )

    statistics = compute_input_statistics([shard])

    np.testing.assert_allclose(statistics["text"].mean, [2.0, 4.0])
    np.testing.assert_allclose(statistics["vision"].mean, [1.0, 2.0])
    np.testing.assert_allclose(statistics["vision"].std, [1.0, 1.0])


def test_input_normalizer_uses_checkpointed_statistics():
    shard = FeatureShard(
        sample_ids=np.array(["one", "two"]),
        text=np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32),
        audio=np.array([[2.0, 4.0], [4.0, 8.0]], dtype=np.float32),
        vision=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        modality_mask=np.ones((2, 3), dtype=np.bool_),
    )
    statistics = compute_input_statistics([shard])
    normalizer = InputNormalizer((2, 2, 2))
    normalizer.set_statistics(statistics)

    text, audio, vision = normalizer(
        torch.from_numpy(shard.text),
        torch.from_numpy(shard.audio),
        torch.from_numpy(shard.vision),
    )

    assert torch.allclose(text.mean(dim=0), torch.zeros(2))
    assert torch.allclose(audio.mean(dim=0), torch.zeros(2))
    assert torch.allclose(vision.mean(dim=0), torch.zeros(2))
    assert set(normalizer.state_dict()) == {
        "text_mean", "text_std", "audio_mean", "audio_std", "vision_mean", "vision_std"
    }
