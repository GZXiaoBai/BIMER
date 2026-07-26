import numpy as np
import pytest
import torch

from bimer.feature_store import FeatureShard, FeatureStore
from bimer.text_adaptation import (
    BalancedTextSampler,
    LoraTextAdaptationConfig,
    SupervisedContrastiveLoss,
    compose_feature_stores,
    replace_text_features,
    rewrite_text_feature_store,
)


def test_supervised_contrastive_loss_prefers_same_class_neighbors():
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    labels = torch.tensor([0, 0, 1, 1])
    aligned = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    mixed = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        requires_grad=True,
    )

    aligned_loss = loss_fn(aligned, labels)
    mixed_loss = loss_fn(mixed, labels)

    assert aligned_loss < mixed_loss
    aligned_loss.backward()
    assert aligned.grad is not None
    assert torch.isfinite(aligned.grad).all()


def test_replace_text_features_preserves_all_other_cached_arrays():
    shard = FeatureShard(
        sample_ids=np.asarray(["a", "b"]),
        text=np.zeros((2, 4), dtype=np.float32),
        audio=np.arange(12, dtype=np.float32).reshape(2, 6),
        vision=np.arange(10, dtype=np.float32).reshape(2, 5),
        modality_mask=np.asarray([[True, True, True], [True, False, True]]),
        modality_quality=np.full((2, 3, 4), 0.5, dtype=np.float32),
    )
    replacements = {
        "a": np.ones(4, dtype=np.float32),
        "b": np.full(4, 2.0, dtype=np.float32),
    }

    updated = replace_text_features(shard, replacements, expected_dim=4)

    np.testing.assert_array_equal(updated.text, np.asarray([[1] * 4, [2] * 4]))
    np.testing.assert_array_equal(updated.audio, shard.audio)
    np.testing.assert_array_equal(updated.vision, shard.vision)
    np.testing.assert_array_equal(updated.modality_mask, shard.modality_mask)
    np.testing.assert_array_equal(updated.modality_quality, shard.modality_quality)
    np.testing.assert_array_equal(updated.sample_ids, shard.sample_ids)


def test_replace_text_features_rejects_missing_or_wrong_width_rows():
    shard = FeatureShard(
        sample_ids=np.asarray(["a"]),
        text=np.zeros((1, 4), dtype=np.float32),
        audio=np.zeros((1, 6), dtype=np.float32),
        vision=np.zeros((1, 5), dtype=np.float32),
        modality_mask=np.ones((1, 3), dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="missing adapted text"):
        replace_text_features(shard, {}, expected_dim=4)
    with pytest.raises(ValueError, match="width"):
        replace_text_features(
            shard,
            {"a": np.ones(3, dtype=np.float32)},
            expected_dim=4,
        )


def test_lora_config_is_fixed_to_the_predeclared_search_space():
    config = LoraTextAdaptationConfig(learning_rate=1e-4)

    assert config.rank == 8
    assert config.alpha == 16
    assert config.dropout == 0.1
    assert config.max_epochs == 5
    assert config.max_length == 128
    assert config.contrastive_weight == 0.1
    with pytest.raises(ValueError, match="learning_rate"):
        LoraTextAdaptationConfig(learning_rate=3e-4)


def test_balanced_text_sampler_alternates_languages_and_reshuffles_each_epoch():
    datasets = ["meld", "meld", "meld", "emotiontalk", "emotiontalk"]
    sampler = BalancedTextSampler(datasets, seed=42)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert [datasets[index] for index in first[:4]] == [
        "meld",
        "emotiontalk",
        "meld",
        "emotiontalk",
    ]
    assert first != second
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_rewrite_text_feature_store_preserves_shards_and_sample_ids(tmp_path):
    source = FeatureStore(tmp_path / "source")
    destination = FeatureStore(tmp_path / "destination")
    shard = FeatureShard(
        sample_ids=np.asarray(["a", "b"]),
        text=np.zeros((2, 4), dtype=np.float32),
        audio=np.ones((2, 6), dtype=np.float32),
        vision=np.ones((2, 5), dtype=np.float32),
        modality_mask=np.ones((2, 3), dtype=np.bool_),
    )
    source.write("meld", "train", 3, shard)

    written = rewrite_text_feature_store(
        source,
        destination,
        replacements={
            "a": np.ones(4, dtype=np.float32),
            "b": np.full(4, 2.0, dtype=np.float32),
        },
        partitions=(("meld", "train"),),
        expected_dim=4,
    )

    assert written == [destination.path("meld", "train", 3)]
    updated = destination.read(written[0])
    assert updated.sample_ids.tolist() == ["a", "b"]
    np.testing.assert_array_equal(updated.text, [[1, 1, 1, 1], [2, 2, 2, 2]])


def test_compose_feature_stores_replaces_only_requested_corrupted_modality(tmp_path):
    base = FeatureStore(tmp_path / "base")
    replacement = FeatureStore(tmp_path / "replacement")
    output = FeatureStore(tmp_path / "output")
    original = FeatureShard(
        sample_ids=np.asarray(["a", "b"]),
        text=np.ones((2, 4), dtype=np.float32),
        audio=np.full((2, 6), 2, dtype=np.float32),
        vision=np.full((2, 5), 3, dtype=np.float32),
        modality_mask=np.ones((2, 3), dtype=np.bool_),
        modality_quality=np.full((2, 3, 4), 0.75, dtype=np.float32),
    )
    corrupted = FeatureShard(
        sample_ids=original.sample_ids,
        text=np.full_like(original.text, 8),
        audio=np.full_like(original.audio, 9),
        vision=np.full_like(original.vision, 7),
        modality_mask=np.asarray([[True, True, True], [True, False, True]]),
        modality_quality=np.full((2, 3, 4), 0.25, dtype=np.float32),
    )
    base.write("meld", "test", 0, original)
    replacement.write("meld", "test", 0, corrupted)

    compose_feature_stores(
        base,
        output,
        replacements={"audio": replacement},
        partitions=[("meld", "test")],
    )

    merged = output.read(output.path("meld", "test", 0))
    np.testing.assert_allclose(merged.text, original.text)
    np.testing.assert_allclose(merged.vision, original.vision)
    np.testing.assert_allclose(merged.audio[0], corrupted.audio[0])
    np.testing.assert_allclose(merged.audio[1], 0.0)
    assert merged.modality_mask[:, 1].tolist() == [True, False]
    np.testing.assert_allclose(
        merged.modality_quality[0, 1],
        corrupted.modality_quality[0, 1],
    )
    np.testing.assert_allclose(merged.modality_quality[1, 1], 0.0)
