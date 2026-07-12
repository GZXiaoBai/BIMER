import numpy as np
import pytest
import torch

from bimer.losses import masked_weighted_cross_entropy, sqrt_inverse_class_weights
from bimer.metrics import bootstrap_weighted_f1, classification_metrics


def test_sqrt_inverse_weights_are_normalized_and_favor_rare_classes():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    weights = sqrt_inverse_class_weights(labels, num_classes=3)
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_masked_loss_ignores_padded_utterances():
    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]])
    labels = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    loss = masked_weighted_cross_entropy(logits, labels, mask)
    assert loss.item() < 0.05


def test_classification_metrics_return_required_report_fields():
    report = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        label_names=("neutral", "joy"),
    )
    assert report["accuracy"] == pytest.approx(0.75)
    assert set(report) >= {
        "weighted_f1",
        "macro_f1",
        "accuracy",
        "per_class_f1",
        "confusion_matrix",
    }


def test_bootstrap_interval_is_deterministic_for_a_seed():
    truth = np.array([0, 0, 1, 1, 1, 0])
    prediction = np.array([0, 1, 1, 1, 0, 0])
    first = bootstrap_weighted_f1(truth, prediction, iterations=100, seed=42)
    second = bootstrap_weighted_f1(truth, prediction, iterations=100, seed=42)
    assert first == second
    assert first[0] <= first[1]

