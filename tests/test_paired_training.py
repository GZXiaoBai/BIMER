import importlib
import sys

import numpy as np
import pytest
import torch

from bimer.model import QualityAwareLanguageGatedFusion
from bimer.paired_training import (
    BalancedCorruptionPairSampler,
    CorruptionPair,
    build_corruption_pairs,
    collate_corruption_pairs,
    gate_ranking_loss,
)
from bimer.training import DialogueExample


def test_paired_training_can_load_without_importing_training_module():
    sys.modules.pop("bimer.paired_training", None)
    sys.modules.pop("bimer.training", None)

    importlib.import_module("bimer.paired_training")

    assert "bimer.training" not in sys.modules


def _example(
    *,
    feature_value: float = 1.0,
    available=True,
    dataset: str = "meld",
) -> DialogueExample:
    mask = np.full((2, 3), available, dtype=np.bool_)
    return DialogueExample(
        dataset=dataset,
        sample_ids=("meld:d:0", "meld:d:1"),
        text=np.full((2, 4), feature_value, dtype=np.float32),
        audio=np.full((2, 6), feature_value, dtype=np.float32),
        vision=np.full((2, 5), feature_value, dtype=np.float32),
        modality_mask=mask,
        labels=np.asarray([0, 1], dtype=np.int64),
        language_id=0,
    )


def test_build_corruption_pairs_requires_identical_samples_and_labels():
    clean = _example()
    corrupted = _example(feature_value=2.0)

    pairs = build_corruption_pairs(
        [clean],
        [corrupted],
        corrupted_modality="audio",
        severity=10.0,
    )

    assert pairs == (
        CorruptionPair(
            clean=clean,
            corrupted=corrupted,
            corrupted_modality="audio",
            severity=10.0,
        ),
    )

    mismatched = DialogueExample(
        **{
            **{
                field: getattr(corrupted, field)
                for field in (
                    "dataset",
                    "text",
                    "audio",
                    "vision",
                    "modality_mask",
                    "labels",
                    "language_id",
                    "modality_quality",
                )
            },
            "sample_ids": ("meld:other:0", "meld:other:1"),
        }
    )
    with pytest.raises(ValueError, match="matching clean window"):
        build_corruption_pairs([clean], [mismatched], corrupted_modality="audio")


def test_collate_corruption_pairs_preserves_alignment_and_modality_index():
    batch = collate_corruption_pairs(
        [
            CorruptionPair(
                clean=_example(),
                corrupted=_example(feature_value=2.0),
                corrupted_modality="vision",
                severity=0.5,
            )
        ]
    )

    assert batch.clean.sample_ids == batch.corrupted.sample_ids
    assert batch.corrupted_modality.tolist() == [2]
    assert batch.severity.tolist() == [0.5]


def test_gate_ranking_loss_pushes_corrupted_gate_below_clean_gate():
    clean = torch.tensor([[[0.30, 0.50, 0.20]]], requires_grad=True)
    corrupted = torch.tensor([[[0.30, 0.47, 0.23]]], requires_grad=True)
    mask = torch.ones((1, 1, 3), dtype=torch.bool)
    attention = torch.ones((1, 1), dtype=torch.bool)

    loss = gate_ranking_loss(
        clean,
        corrupted,
        clean_modality_mask=mask,
        corrupted_modality_mask=mask,
        attention_mask=attention,
        corrupted_modality=torch.tensor([1]),
        margin=0.10,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.07)
    assert clean.grad[0, 0, 1].item() < 0
    assert corrupted.grad[0, 0, 1].item() > 0


def test_gate_ranking_loss_excludes_hard_missing_modalities_without_nan():
    clean = torch.rand((1, 2, 3), requires_grad=True)
    corrupted = torch.rand((1, 2, 3), requires_grad=True)
    clean_mask = torch.ones((1, 2, 3), dtype=torch.bool)
    corrupted_mask = clean_mask.clone()
    corrupted_mask[:, :, 2] = False

    loss = gate_ranking_loss(
        clean,
        corrupted,
        clean_modality_mask=clean_mask,
        corrupted_modality_mask=corrupted_mask,
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        corrupted_modality=torch.tensor([2]),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(clean.grad).all()


def test_ranking_optimization_reduces_the_corrupted_modality_gate():
    torch.manual_seed(42)
    model = QualityAwareLanguageGatedFusion(
        text_dim=4,
        audio_dim=6,
        vision_dim=5,
        d_model=8,
        num_heads=2,
        transformer_layers=1,
        transformer_ffn_dim=16,
        context_hidden_dim=4,
        num_classes=2,
        dropout=0.0,
        modality_dropout=0.0,
        use_context=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    common = {
        "text_features": torch.randn(1, 2, 4),
        "audio_features": torch.randn(1, 2, 6),
        "vision_features": torch.randn(1, 2, 5),
        "modality_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "language_ids": torch.zeros(1, dtype=torch.long),
    }
    clean_quality = torch.ones(1, 2, 3, 4)
    corrupted_quality = clean_quality.clone()
    corrupted_quality[:, :, 1] = 0.0

    with torch.no_grad():
        initial_clean = model(**common, modality_quality=clean_quality).gates[:, :, 1]
        initial_corrupted = model(**common, modality_quality=corrupted_quality).gates[:, :, 1]
        initial_difference = float((initial_clean - initial_corrupted).mean())
    for _ in range(40):
        optimizer.zero_grad()
        clean_output = model(**common, modality_quality=clean_quality)
        corrupted_output = model(**common, modality_quality=corrupted_quality)
        loss = gate_ranking_loss(
            clean_output.gates,
            corrupted_output.gates,
            clean_modality_mask=common["modality_mask"],
            corrupted_modality_mask=common["modality_mask"],
            attention_mask=common["attention_mask"],
            corrupted_modality=torch.tensor([1]),
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_clean = model(**common, modality_quality=clean_quality).gates[:, :, 1]
        final_corrupted = model(**common, modality_quality=corrupted_quality).gates[:, :, 1]
        final_difference = float((final_clean - final_corrupted).mean())

    assert final_difference > initial_difference
    assert final_difference >= 0.09


def test_corruption_pair_rejects_every_alignment_mismatch():
    clean = _example()
    corrupted = _example(feature_value=2.0)
    with pytest.raises(ValueError, match="corrupted_modality"):
        CorruptionPair(clean, corrupted, "invalid")  # type: ignore[arg-type]

    other_dataset = DialogueExample(
        **{
            field: ("emotiontalk" if field == "dataset" else getattr(corrupted, field))
            for field in (
                "dataset",
                "sample_ids",
                "text",
                "audio",
                "vision",
                "modality_mask",
                "labels",
                "language_id",
                "modality_quality",
            )
        }
    )
    with pytest.raises(ValueError, match="same dataset"):
        CorruptionPair(clean, other_dataset, "audio")

    other_language = DialogueExample(
        **{
            field: (1 if field == "language_id" else getattr(corrupted, field))
            for field in (
                "dataset",
                "sample_ids",
                "text",
                "audio",
                "vision",
                "modality_mask",
                "labels",
                "language_id",
                "modality_quality",
            )
        }
    )
    with pytest.raises(ValueError, match="same language"):
        CorruptionPair(clean, other_language, "audio")

    other_labels = DialogueExample(
        **{
            field: (
                np.asarray([1, 0], dtype=np.int64)
                if field == "labels"
                else getattr(corrupted, field)
            )
            for field in (
                "dataset",
                "sample_ids",
                "text",
                "audio",
                "vision",
                "modality_mask",
                "labels",
                "language_id",
                "modality_quality",
            )
        }
    )
    with pytest.raises(ValueError, match="identical labels"):
        CorruptionPair(clean, other_labels, "audio")


def test_pair_sampler_balances_languages_and_reshuffles_each_epoch():
    meld = CorruptionPair(_example(), _example(feature_value=2.0), "audio")
    emotiontalk_example = _example(dataset="emotiontalk")
    emotiontalk_corrupted = _example(feature_value=3.0, dataset="emotiontalk")
    emotiontalk = CorruptionPair(
        emotiontalk_example,
        emotiontalk_corrupted,
        "vision",
    )
    sampler = BalancedCorruptionPairSampler([meld, meld, emotiontalk], seed=7)

    assert len(sampler) == 4
    assert sorted(iter(sampler)) == [0, 1, 2, 2]
    sampler.set_epoch(1)
    assert len(list(iter(sampler))) == 4
    with pytest.raises(ValueError, match="non-negative"):
        sampler.set_epoch(-1)

    single_language = BalancedCorruptionPairSampler([meld, meld], seed=3)
    assert len(single_language) == 2
    assert sorted(iter(single_language)) == [0, 1]


def test_pair_build_and_collation_reject_duplicates_and_empty_batches():
    clean = _example()
    corrupted = _example(feature_value=2.0)
    with pytest.raises(ValueError, match="duplicate sample windows"):
        build_corruption_pairs([clean, clean], [corrupted], corrupted_modality="text")
    with pytest.raises(ValueError, match="duplicate sample windows"):
        build_corruption_pairs([clean], [corrupted, corrupted], corrupted_modality="text")
    with pytest.raises(ValueError, match="at least one"):
        build_corruption_pairs([], [], corrupted_modality="text")
    with pytest.raises(ValueError, match="empty"):
        collate_corruption_pairs([])

    batch = collate_corruption_pairs([CorruptionPair(clean, corrupted, "audio", severity=10.0)])
    moved = batch.to(torch.device("cpu"))
    assert moved.corrupted_modality.device.type == "cpu"
    assert moved.severity.tolist() == [10.0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"corrupted_gates": torch.ones(1, 1, 3)}, "same shape"),
        (
            {
                "clean_gates": torch.ones(1, 2),
                "corrupted_gates": torch.ones(1, 2),
            },
            r"\[batch, sequence, 3\]",
        ),
        ({"clean_modality_mask": torch.ones(1, 1, 3)}, "must match"),
        ({"corrupted_modality_mask": torch.ones(1, 1, 3)}, "must match"),
        ({"attention_mask": torch.ones(1, 1)}, "attention_mask"),
        ({"corrupted_modality": torch.tensor([0, 1])}, r"\[batch\]"),
        ({"margin": -0.1}, "non-negative"),
        ({"corrupted_modality": torch.tensor([3])}, "0, 1, or 2"),
    ],
)
def test_gate_ranking_loss_validates_tensor_contract(change, message):
    arguments = {
        "clean_gates": torch.ones(1, 2, 3),
        "corrupted_gates": torch.ones(1, 2, 3),
        "clean_modality_mask": torch.ones(1, 2, 3),
        "corrupted_modality_mask": torch.ones(1, 2, 3),
        "attention_mask": torch.ones(1, 2),
        "corrupted_modality": torch.tensor([0]),
        "margin": 0.1,
    }
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        gate_ranking_loss(**arguments)
