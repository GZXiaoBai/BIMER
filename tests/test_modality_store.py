import numpy as np
import pytest

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
    loaded = store.read_verified(
        "emotiontalk", "validation", 0, np.array(["a", "b"])
    )
    assert path.is_file()
    assert loaded.sample_ids.tolist() == ["a", "b"]
    assert not list(tmp_path.rglob("*.tmp"))


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
