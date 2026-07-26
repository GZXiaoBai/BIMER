from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bimer.feature_store import FeatureShard, FeatureStore
from bimer.quality_attachment import QualityAttachmentRunner
from bimer.schema import UtteranceRecord


def test_quality_attachment_reuses_features_and_replaces_quality(tmp_path):
    records = [
        UtteranceRecord(
            dataset="meld",
            split="train",
            dialogue_id="d",
            utterance_id=index,
            text="human text",
            emotion="neutral",
            language="en",
            start_seconds=float(index),
            end_seconds=float(index + 1),
            video_path=f"{index}.mp4",
        )
        for index in range(2)
    ]
    base = FeatureStore(tmp_path / "base")
    original = FeatureShard(
        sample_ids=np.asarray([record.sample_id for record in records]),
        text=np.ones((2, 4), np.float32),
        audio=np.full((2, 6), 2, np.float32),
        vision=np.full((2, 5), 3, np.float32),
        modality_mask=np.asarray([[True, True, True], [True, True, False]]),
    )
    base.write("meld", "train", 0, original)

    runner = QualityAttachmentRunner(
        audio_quality_loader=lambda path: np.asarray([0.1, 0.2, 0.3, 0.4]),
        vision_quality_loader=lambda path: np.asarray([0.5, 0.6, 0.7, 0.8]),
        audio_executor_factory=ThreadPoolExecutor,
        vision_executor_factory=ThreadPoolExecutor,
        workers=1,
        queue_capacity=1,
    )
    paths = runner.run(records, base, FeatureStore(tmp_path / "output"))

    assert len(paths) == 1
    attached = FeatureStore(tmp_path / "output").read(paths[0])
    np.testing.assert_array_equal(attached.text, original.text)
    np.testing.assert_array_equal(attached.audio, original.audio)
    np.testing.assert_array_equal(attached.vision, original.vision)
    np.testing.assert_allclose(attached.modality_quality[0, 1], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(attached.modality_quality[0, 2], [0.5, 0.6, 0.7, 0.8])
    assert np.all(attached.modality_quality[1, 2] == 0)


def test_quality_attachment_resumes_verified_output(tmp_path):
    record = UtteranceRecord(
        dataset="emotiontalk",
        split="validation",
        dialogue_id="d",
        utterance_id=0,
        text="text",
        emotion="neutral",
        language="zh",
        start_seconds=0,
        end_seconds=1,
        video_path="x.mp4",
    )
    shard = FeatureShard(
        sample_ids=np.asarray([record.sample_id]),
        text=np.ones((1, 4), np.float32),
        audio=np.ones((1, 6), np.float32),
        vision=np.ones((1, 5), np.float32),
        modality_mask=np.ones((1, 3), np.bool_),
    )
    base = FeatureStore(tmp_path / "base")
    output = FeatureStore(tmp_path / "output")
    base.write("emotiontalk", "validation", 0, shard)
    existing = output.write("emotiontalk", "validation", 0, shard)
    runner = QualityAttachmentRunner(
        audio_quality_loader=lambda path: (_ for _ in ()).throw(AssertionError()),
        vision_quality_loader=lambda path: (_ for _ in ()).throw(AssertionError()),
        audio_executor_factory=ThreadPoolExecutor,
        vision_executor_factory=ThreadPoolExecutor,
        workers=1,
        queue_capacity=1,
    )

    assert runner.run([record], base, output) == [existing]
