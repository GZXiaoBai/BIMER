import numpy as np
import pytest
import torch

from bimer.losses import (
    masked_classification_loss,
    masked_weighted_cross_entropy,
    sqrt_inverse_class_weights,
)
from bimer.metrics import (
    bootstrap_weighted_f1,
    classification_metrics,
    cluster_bootstrap_weighted_f1,
    paired_cluster_bootstrap_weighted_f1_delta,
)


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


def test_balanced_softmax_uses_training_class_counts():
    logits = torch.zeros(1, 1, 2, requires_grad=True)
    labels = torch.tensor([[1]])
    mask = torch.tensor([[True]])

    balanced = masked_classification_loss(
        logits,
        labels,
        mask,
        loss_name="balanced_softmax",
        class_counts=torch.tensor([9.0, 1.0]),
    )
    ordinary = masked_classification_loss(
        logits,
        labels,
        mask,
        loss_name="weighted_ce",
    )

    assert balanced > ordinary
    balanced.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_gamma_zero_matches_weighted_cross_entropy():
    logits = torch.tensor([[[1.0, -0.5], [0.2, 0.4]]], requires_grad=True)
    labels = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, True]])
    weights = torch.tensor([0.75, 1.25])

    focal = masked_classification_loss(
        logits,
        labels,
        mask,
        loss_name="focal",
        class_weights=weights,
        focal_gamma=0.0,
    )
    weighted = masked_weighted_cross_entropy(
        logits,
        labels,
        mask,
        class_weights=weights,
    )

    assert focal.item() == pytest.approx(float(weighted.detach()))


def test_balanced_softmax_rejects_missing_or_zero_class_counts():
    logits = torch.zeros(1, 1, 2)
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])

    with pytest.raises(ValueError, match="class_counts"):
        masked_classification_loss(
            logits,
            labels,
            mask,
            loss_name="balanced_softmax",
        )
    with pytest.raises(ValueError, match="positive"):
        masked_classification_loss(
            logits,
            labels,
            mask,
            loss_name="balanced_softmax",
            class_counts=torch.tensor([1.0, 0.0]),
        )


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


def test_cluster_bootstrap_resamples_complete_dialogues():
    truth = np.array([0, 1, 1, 0])
    prediction = np.array([0, 1, 0, 0])
    contexts = np.array(["d1", "d1", "d2", "d2"])

    first = cluster_bootstrap_weighted_f1(
        truth, prediction, contexts, iterations=100, seed=42
    )
    second = cluster_bootstrap_weighted_f1(
        truth, prediction, contexts, iterations=100, seed=42
    )

    assert first == second
    assert first[0] <= first[1]


def test_paired_cluster_bootstrap_reports_candidate_improvement():
    truth = np.array([0, 1, 0, 1])
    baseline = np.array([1, 0, 1, 0])
    candidate = truth.copy()
    contexts = np.array(["d1", "d1", "d2", "d2"])

    low, high = paired_cluster_bootstrap_weighted_f1_delta(
        truth, baseline, candidate, contexts, iterations=100, seed=42
    )

    assert low > 0.0
    assert high > 0.0
