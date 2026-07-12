import numpy as np

from bimer.feature_store import FeatureShard, FeatureStore
from bimer.feature_extraction_runner import DatasetFeatureExtractionRunner
from bimer.schema import UtteranceRecord


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


def test_dataset_feature_runner_writes_masks_for_missing_vision(tmp_path):
    class Text:
        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    class Audio:
        def encode(self, waveforms):
            return np.ones((len(waveforms), 6), dtype=np.float32)

    records = [
        UtteranceRecord(
            dataset="meld",
            split="train",
            dialogue_id="d1",
            utterance_id=index,
            text="line",
            emotion="neutral",
            language="en",
            start_seconds=0.0,
            end_seconds=1.0,
            video_path=tmp_path / f"{index}.mp4",
        )
        for index in range(2)
    ]
    runner = DatasetFeatureExtractionRunner(
        text_extractor=Text(),
        audio_extractor=Audio(),
        waveform_loader=lambda _: np.ones(160, dtype=np.float32),
        vision_loader=lambda path: (
            np.ones(5, dtype=np.float32),
            path.stem == "0",
        ),
    )
    paths = runner.run(records, FeatureStore(tmp_path / "features"), shard_size=1)
    assert len(paths) == 2
    second = FeatureStore(tmp_path / "features").read(paths[1])
    assert second.modality_mask.tolist() == [[True, True, False]]
    assert np.all(second.vision == 0)
