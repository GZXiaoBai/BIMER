import numpy as np
import torch

from bimer.feature_extractors import (
    mean_pool_hidden,
    uniform_frame_indices,
    vision_modality_available,
)


def test_mean_pool_hidden_ignores_padding_tokens():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool_hidden(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_uniform_frame_indices_return_exact_requested_count():
    assert uniform_frame_indices(5, 4).tolist() == [0, 1, 3, 4]
    repeated = uniform_frame_indices(2, 4)
    assert repeated.shape == (4,)
    assert repeated[0] == 0 and repeated[-1] == 1


def test_visual_modality_requires_four_detected_faces():
    assert vision_modality_available(np.array([1, 1, 1, 1], dtype=bool)) is True
    assert vision_modality_available(np.array([1, 1, 1, 0], dtype=bool)) is False
