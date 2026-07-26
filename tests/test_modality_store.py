import numpy as np
import pytest

import bimer.modality_store as modality_store
from bimer.feature_store import FeatureStore
from bimer.modality_store import (
    MODALITY_DIMS,
    ModalityShard,
    ModalityStore,
    merge_staged_shard,
)


def _write_staging_triplet(
    root,
    *,
    text_ids=("a", "b"),
    audio_ids=("a", "b"),
    vision_ids=("a", "b"),
    vision_available=(True, False),
):
    settings = {
        "text": (text_ids, (True, True)),
        "audio": (audio_ids, (True, True)),
        "vision": (vision_ids, vision_available),
    }
    for offset, (modality, width) in enumerate(MODALITY_DIMS.items(), start=1):
        sample_ids, available = settings[modality]
        ModalityStore(root, modality, width).write(
            "emotiontalk",
            "validation",
            0,
            ModalityShard(
                sample_ids=np.asarray(sample_ids),
                features=np.full((2, width), offset, dtype=np.float32),
                available=np.asarray(available, dtype=np.bool_),
            ),
        )


def test_modality_store_round_trips_and_validates_expected_ids(tmp_path):
    store = ModalityStore(tmp_path, "text", 768)
    shard = ModalityShard(
        sample_ids=np.array(["a", "b"]),
        features=np.ones((2, 768), dtype=np.float32),
        available=np.array([True, True]),
    )
    path = store.write("emotiontalk", "validation", 0, shard)
    loaded = store.read_verified("emotiontalk", "validation", 0, np.array(["a", "b"]))
    assert path.is_file()
    assert loaded.sample_ids.tolist() == ["a", "b"]
    assert loaded.quality.shape == (2, 4)
    assert not list(tmp_path.rglob("*.tmp"))


def test_modality_store_round_trips_continuous_quality(tmp_path):
    quality = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    store = ModalityStore(tmp_path, "audio", 1024)
    store.write(
        "meld",
        "train",
        0,
        ModalityShard(
            sample_ids=np.array(["a"]),
            features=np.ones((1, 1024), dtype=np.float32),
            available=np.array([True]),
            quality=quality,
        ),
    )

    loaded = store.read(store.path("meld", "train", 0))

    np.testing.assert_allclose(loaded.quality, quality)


def test_modality_store_rejects_non_finite_features(tmp_path):
    store = ModalityStore(tmp_path, "audio", 1024)
    with pytest.raises(ValueError, match="features must be finite"):
        store.write(
            "emotiontalk",
            "validation",
            0,
            ModalityShard(
                sample_ids=np.array(["a"]),
                features=np.full((1, 1024), np.nan, dtype=np.float32),
                available=np.array([True]),
            ),
        )


def test_merge_staged_shard_rejects_reordered_ids(tmp_path):
    _write_staging_triplet(tmp_path, audio_ids=("b", "a"))
    with pytest.raises(ValueError, match="unexpected sample IDs"):
        merge_staged_shard(
            staging_root=tmp_path,
            final_store=FeatureStore(tmp_path / "final"),
            dataset="emotiontalk",
            split="validation",
            shard_index=0,
            expected_sample_ids=np.array(["a", "b"]),
        )


def test_merge_staged_shard_zeroes_unavailable_rows(tmp_path):
    _write_staging_triplet(tmp_path)
    final_store = FeatureStore(tmp_path / "final")
    path = merge_staged_shard(
        staging_root=tmp_path,
        final_store=final_store,
        dataset="emotiontalk",
        split="validation",
        shard_index=0,
        expected_sample_ids=np.array(["a", "b"]),
    )
    shard = final_store.read(path)
    assert shard.modality_mask.tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert np.all(shard.vision[1] == 0)
    assert np.all(shard.text == 1)
    assert np.all(shard.audio == 2)
    assert shard.modality_quality.shape == (2, 3, 4)
    assert np.all(shard.modality_quality[1, 2] == 0)


def test_merge_reuses_verified_final_shard_without_staging(tmp_path):
    final_store = FeatureStore(tmp_path / "final")
    _write_staging_triplet(tmp_path)
    first = merge_staged_shard(
        staging_root=tmp_path,
        final_store=final_store,
        dataset="emotiontalk",
        split="validation",
        shard_index=0,
        expected_sample_ids=np.array(["a", "b"]),
    )
    for path in (tmp_path / "staging").rglob("*.npz"):
        path.unlink()
    second = merge_staged_shard(
        staging_root=tmp_path,
        final_store=final_store,
        dataset="emotiontalk",
        split="validation",
        shard_index=0,
        expected_sample_ids=np.array(["a", "b"]),
    )
    assert second == first


def test_merge_replaced_modality_shard_preserves_unmodified_modalities(tmp_path):
    base_store = FeatureStore(tmp_path / "base")
    output_store = FeatureStore(tmp_path / "output")
    base_store.write(
        "emotiontalk",
        "test",
        0,
        modality_store.FeatureShard(
            sample_ids=np.array(["a", "b"]),
            text=np.full((2, 768), 1, dtype=np.float32),
            audio=np.full((2, 1024), 2, dtype=np.float32),
            vision=np.full((2, 512), 3, dtype=np.float32),
            modality_mask=np.array([[True, True, True], [True, True, False]], dtype=np.bool_),
        ),
    )
    ModalityStore(tmp_path / "replacement", "audio", 1024).write(
        "emotiontalk",
        "test",
        0,
        ModalityShard(
            sample_ids=np.array(["a", "b"]),
            features=np.full((2, 1024), 9, dtype=np.float32),
            available=np.array([True, False]),
            quality=np.asarray(
                [[0.1, 0.2, 0.3, 0.4], [0.9, 0.9, 0.9, 0.9]],
                dtype=np.float32,
            ),
        ),
    )

    path = modality_store.merge_replaced_modality_shard(
        base_store=base_store,
        staging_root=tmp_path / "replacement",
        final_store=output_store,
        dataset="emotiontalk",
        split="test",
        shard_index=0,
        expected_sample_ids=np.array(["a", "b"]),
        modality="audio",
    )

    merged = output_store.read(path)
    assert np.all(merged.text == 1)
    assert np.all(merged.vision == 3)
    assert np.all(merged.audio[0] == 9)
    assert np.all(merged.audio[1] == 0)
    assert merged.modality_mask.tolist() == [
        [True, True, True],
        [True, False, False],
    ]
    np.testing.assert_allclose(merged.modality_quality[0, 1], [0.1, 0.2, 0.3, 0.4])
    assert np.all(merged.modality_quality[1, 1] == 0)


def test_seed_staging_from_base_excludes_recomputed_modality(tmp_path):
    base_store = FeatureStore(tmp_path / "base")
    base_store.write(
        "meld",
        "test",
        0,
        modality_store.FeatureShard(
            sample_ids=np.array(["a", "b"]),
            text=np.full((2, 768), 1, dtype=np.float32),
            audio=np.full((2, 1024), 2, dtype=np.float32),
            vision=np.full((2, 512), 3, dtype=np.float32),
            modality_mask=np.array([[True, True, True], [True, False, False]], dtype=np.bool_),
        ),
    )

    paths = modality_store.seed_staging_from_base_shard(
        base_store=base_store,
        staging_root=tmp_path / "condition",
        dataset="meld",
        split="test",
        shard_index=0,
        expected_sample_ids=np.array(["a", "b"]),
        recompute_modality="vision",
    )

    assert {path.parent.name for path in paths} == {"text", "audio"}
    assert not ModalityStore(tmp_path / "condition", "vision", 512).path("meld", "test", 0).exists()
    text = ModalityStore(tmp_path / "condition", "text", 768).read_verified(
        "meld", "test", 0, np.array(["a", "b"])
    )
    audio = ModalityStore(tmp_path / "condition", "audio", 1024).read_verified(
        "meld", "test", 0, np.array(["a", "b"])
    )
    assert text.available.tolist() == [True, True]
    assert audio.available.tolist() == [True, False]
    assert np.all(text.features == 1)
    assert np.all(audio.features[0] == 2)
    assert np.all(audio.features[1] == 0)
