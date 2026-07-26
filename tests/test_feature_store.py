from pathlib import Path

import numpy as np
import pytest

from bimer.feature_extraction_runner import DatasetFeatureExtractionRunner
from bimer.feature_store import FeatureShard, FeatureStore
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
    assert loaded.modality_quality.shape == (2, 3, 4)
    assert not loaded.modality_quality[1, 2].any()


def test_feature_store_preserves_explicit_modality_quality(tmp_path):
    store = FeatureStore(tmp_path)
    quality = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    shard = FeatureShard(
        sample_ids=np.array(["one", "two"]),
        text=np.ones((2, 4), dtype=np.float32),
        audio=np.ones((2, 6), dtype=np.float32),
        vision=np.ones((2, 5), dtype=np.float32),
        modality_mask=np.ones((2, 3), dtype=np.bool_),
        modality_quality=quality,
    )

    loaded = store.read(store.write("meld", "train", 0, shard))

    np.testing.assert_allclose(loaded.modality_quality, quality)


def test_feature_store_reads_legacy_shard_without_quality(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        sample_ids=np.array(["one"]),
        text=np.ones((1, 4), dtype=np.float32),
        audio=np.ones((1, 6), dtype=np.float32),
        vision=np.ones((1, 5), dtype=np.float32),
        modality_mask=np.array([[1, 1, 0]], dtype=np.bool_),
    )

    loaded = FeatureStore(tmp_path).read(path)

    assert loaded.modality_quality.shape == (1, 3, 4)
    assert loaded.modality_quality[0, :2].tolist() == [[1.0] * 4, [1.0] * 4]
    assert loaded.modality_quality[0, 2].tolist() == [0.0] * 4


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


def test_feature_shard_rejects_non_matrix_features():
    with pytest.raises(ValueError, match="text must be a matrix"):
        FeatureShard(
            sample_ids=np.array(["one"]),
            text=np.ones(768, dtype=np.float32),
            audio=np.ones((1, 6), dtype=np.float32),
            vision=np.ones((1, 5), dtype=np.float32),
            modality_mask=np.ones((1, 3), dtype=np.bool_),
        )


def test_feature_shard_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="sample_ids must be unique"):
        FeatureShard(
            sample_ids=np.array(["same", "same"]),
            text=np.ones((2, 4), dtype=np.float32),
            audio=np.ones((2, 6), dtype=np.float32),
            vision=np.ones((2, 5), dtype=np.float32),
            modality_mask=np.ones((2, 3), dtype=np.bool_),
        )


def test_feature_shard_rejects_non_finite_values():
    text = np.ones((1, 4), dtype=np.float32)
    text[0, 0] = np.nan
    with pytest.raises(ValueError, match="text features must be finite"):
        FeatureShard(
            sample_ids=np.array(["one"]),
            text=text,
            audio=np.ones((1, 6), dtype=np.float32),
            vision=np.ones((1, 5), dtype=np.float32),
            modality_mask=np.ones((1, 3), dtype=np.bool_),
        )


def test_feature_store_does_not_publish_partial_write(tmp_path, monkeypatch):
    store = FeatureStore(tmp_path)
    shard = FeatureShard(
        sample_ids=np.array(["one"]),
        text=np.ones((1, 4), dtype=np.float32),
        audio=np.ones((1, 6), dtype=np.float32),
        vision=np.ones((1, 5), dtype=np.float32),
        modality_mask=np.ones((1, 3), dtype=np.bool_),
    )

    def fail_after_opening(path, **_arrays):
        if hasattr(path, "write"):
            path.write(b"partial")
            path.flush()
        else:
            Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(np, "savez_compressed", fail_after_opening)

    with pytest.raises(RuntimeError, match="simulated disk failure"):
        store.write("emotiontalk", "validation", 0, shard)

    assert not store.path("emotiontalk", "validation", 0).exists()
    assert not list(tmp_path.rglob("*.tmp"))


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
        waveform_loader=lambda _: np.ones(400, dtype=np.float32),
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


def test_dataset_feature_runner_resumes_verified_existing_shard(tmp_path):
    record = UtteranceRecord(
        dataset="emotiontalk",
        split="validation",
        dialogue_id="d1",
        utterance_id=0,
        text="line",
        emotion="neutral",
        language="zh",
        start_seconds=0.0,
        end_seconds=1.0,
        video_path=tmp_path / "0.mp4",
    )
    store = FeatureStore(tmp_path / "features")
    existing = FeatureShard(
        sample_ids=np.array([record.sample_id]),
        text=np.ones((1, 768), dtype=np.float32),
        audio=np.ones((1, 1024), dtype=np.float32),
        vision=np.ones((1, 512), dtype=np.float32),
        modality_mask=np.ones((1, 3), dtype=np.bool_),
    )
    existing_path = store.write("emotiontalk", "validation", 0, existing)

    class MustNotRun:
        def encode(self, _values):
            raise AssertionError("verified shard should be resumed")

    runner = DatasetFeatureExtractionRunner(
        text_extractor=MustNotRun(),
        audio_extractor=MustNotRun(),
        waveform_loader=lambda _: np.ones(160, dtype=np.float32),
        vision_loader=lambda _: (np.ones(512, dtype=np.float32), True),
    )

    assert runner.run([record], store, shard_size=1) == [existing_path]


def test_dataset_feature_runner_rejects_stale_existing_shard(tmp_path):
    record = UtteranceRecord(
        dataset="emotiontalk",
        split="validation",
        dialogue_id="d1",
        utterance_id=0,
        text="line",
        emotion="neutral",
        language="zh",
        start_seconds=0.0,
        end_seconds=1.0,
        video_path=tmp_path / "0.mp4",
    )
    store = FeatureStore(tmp_path / "features")
    stale = FeatureShard(
        sample_ids=np.array(["emotiontalk:validation:stale:0"]),
        text=np.ones((1, 768), dtype=np.float32),
        audio=np.ones((1, 1024), dtype=np.float32),
        vision=np.ones((1, 512), dtype=np.float32),
        modality_mask=np.ones((1, 3), dtype=np.bool_),
    )
    store.write("emotiontalk", "validation", 0, stale)
    runner = DatasetFeatureExtractionRunner(
        text_extractor=object(),
        audio_extractor=object(),
        waveform_loader=lambda _: np.ones(160, dtype=np.float32),
        vision_loader=lambda _: (np.ones(512, dtype=np.float32), True),
    )

    with pytest.raises(ValueError, match="sample IDs"):
        runner.run([record], store, shard_size=1)
