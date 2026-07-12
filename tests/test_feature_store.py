import numpy as np

from bimer.feature_store import FeatureShard, FeatureStore


def test_feature_store_round_trips_without_pickle(tmp_path):
    store = FeatureStore(tmp_path)
    shard = FeatureShard(
        sample_ids=np.array(["meld:train:d1:0", "meld:train:d1:1"]),
        text=np.ones((2, 768), dtype=np.float32),
        audio=np.ones((2, 1024), dtype=np.float32) * 2,
        vision=np.ones((2, 512), dtype=np.float32) * 3,
        modality_mask=np.array([[1, 1, 1], [1, 1, 0]], dtype=np.bool_),
    )
    path = store.write("meld", "train", 0, shard)
    loaded = store.read(path)
    assert loaded.sample_ids.tolist() == shard.sample_ids.tolist()
    assert loaded.vision.shape == (2, 512)
    assert loaded.modality_mask[1].tolist() == [True, True, False]


def test_feature_shard_rejects_mismatched_row_counts():
    try:
        FeatureShard(
            sample_ids=np.array(["one"]),
            text=np.ones((2, 3), dtype=np.float32),
            audio=np.ones((1, 4), dtype=np.float32),
            vision=np.ones((1, 5), dtype=np.float32),
            modality_mask=np.ones((1, 3), dtype=np.bool_),
        )
    except ValueError as exc:
        assert "row count" in str(exc)
    else:
        raise AssertionError("FeatureShard accepted mismatched rows")
