from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import json
import multiprocessing
import threading

import numpy as np
import pytest
import torch

from bimer.feature_store import FeatureStore
from bimer.feature_extraction_runner import DatasetFeatureExtractionRunner
from bimer.parallel_feature_extraction import (
    CpuWorkerError,
    ParallelFeatureExtractionConfig,
    ParallelFeatureExtractionRunner,
    encode_adaptive,
    extract_audio_stage,
    extract_text_stage,
    extract_vision_stage,
    prefetched_map,
)
from bimer.modality_store import ModalityShard, ModalityStore
from bimer.schema import UtteranceRecord


class OomAboveTwoEncoder:
    def __init__(self):
        self.attempts = []

    def encode(self, values, *, batch_size):
        self.attempts.append(batch_size)
        if batch_size > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return np.ones((len(values), 3), dtype=np.float32)


class BrokenEncoder:
    def encode(self, _values, *, batch_size):
        raise RuntimeError(f"bad model at batch {batch_size}")


def square(value):
    return value * value


def fail_on_two(value):
    if value == 2:
        raise ValueError("decode failed")
    return value


def make_records(count):
    return [
        UtteranceRecord(
            dataset="emotiontalk",
            split="validation",
            dialogue_id="d1",
            utterance_id=index,
            text=f"line {index}",
            emotion="neutral",
            language="zh",
            start_seconds=float(index),
            end_seconds=float(index + 1),
            video_path=f"{index}.mp4",
        )
        for index in range(count)
    ]


class MustNotConstruct:
    def __call__(self):
        raise AssertionError("verified staging must skip extractor construction")


class FakeAudioExtractor:
    def encode(self, waveforms, *, batch_size=8):
        del batch_size
        return np.stack(
            [
                np.full(1024, float(np.asarray(waveform).sum()), dtype=np.float32)
                for waveform in waveforms
            ]
        )


class FakeVisionExtractor:
    def __init__(self):
        self.encoded_values = []

    def encode_clips(self, clips, *, batch_size):
        del batch_size
        values = [int(clip[0, 0, 0, 0]) for clip in clips]
        self.encoded_values.extend(values)
        return np.stack(
            [np.full(512, value, dtype=np.float32) for value in values]
        )


class FakeTextExtractor:
    def encode(self, texts, *, batch_size=64):
        del batch_size
        return np.stack(
            [
                np.full(768, int(text.rsplit(" ", 1)[1]) + 1, dtype=np.float32)
                for text in texts
            ]
        )


def test_encode_adaptive_halves_cuda_oom_batch():
    encoder = OomAboveTwoEncoder()
    result = encode_adaptive(encoder, [1, 2, 3], initial_batch_size=8)
    assert encoder.attempts == [8, 4, 2]
    assert result.shape == (3, 3)


def test_encode_adaptive_reraises_non_oom():
    with pytest.raises(RuntimeError, match="bad model"):
        encode_adaptive(BrokenEncoder(), [1], initial_batch_size=8)


def test_prefetched_map_preserves_input_order():
    result = list(
        prefetched_map(
            square,
            range(5),
            workers=2,
            queue_capacity=2,
            executor_factory=ThreadPoolExecutor,
        )
    )
    assert result == [0, 1, 4, 9, 16]


def test_prefetched_map_propagates_worker_failure_with_input_index():
    with pytest.raises(CpuWorkerError, match="input 2.*decode failed"):
        list(
            prefetched_map(
                fail_on_two,
                range(4),
                workers=2,
                queue_capacity=2,
                executor_factory=ThreadPoolExecutor,
            )
        )


def test_prefetched_map_runs_with_spawn_process_pool():
    executor_factory = partial(
        ProcessPoolExecutor,
        mp_context=multiprocessing.get_context("spawn"),
    )
    result = list(
        prefetched_map(
            square,
            range(4),
            workers=2,
            queue_capacity=2,
            executor_factory=executor_factory,
        )
    )
    assert result == [0, 1, 4, 9]


def test_text_stage_skips_verified_staging_shard(tmp_path):
    records = make_records(2)
    ids = np.asarray([record.sample_id for record in records])
    ModalityStore(tmp_path, "text", 768).write(
        "emotiontalk",
        "validation",
        0,
        ModalityShard(
            sample_ids=ids,
            features=np.ones((2, 768), dtype=np.float32),
            available=np.ones(2, dtype=np.bool_),
        ),
    )


def test_text_stage_reports_completed_samples_and_shard(tmp_path, capsys):
    records = make_records(2)
    extract_text_stage(
        records,
        tmp_path,
        FakeTextExtractor,
        shard_size=2,
        batch_size=64,
    )

    output = capsys.readouterr().out
    assert "[bimer:text]" in output
    assert "samples=2/2" in output
    assert "shard=0" in output

    extract_text_stage(
        records,
        tmp_path,
        MustNotConstruct(),
        shard_size=2,
        batch_size=64,
    )


def test_audio_stage_preserves_order_and_masks_empty_audio(tmp_path):
    records = make_records(3)
    waveforms = {
        "0.mp4": np.ones(2, dtype=np.float32),
        "1.mp4": np.empty(0, dtype=np.float32),
        "2.mp4": np.ones(2, dtype=np.float32) * 3,
    }
    extract_audio_stage(
        records,
        tmp_path,
        FakeAudioExtractor,
        waveform_loader=lambda path: waveforms[str(path)],
        shard_size=3,
        batch_size=2,
        workers=2,
        queue_capacity=2,
        executor_factory=ThreadPoolExecutor,
    )

    ids = np.asarray([record.sample_id for record in records])
    shard = ModalityStore(tmp_path, "audio", 1024).read_verified(
        "emotiontalk", "validation", 0, ids
    )
    assert shard.sample_ids.tolist() == ids.tolist()
    assert shard.available.tolist() == [True, False, True]
    assert np.all(shard.features[1] == 0)
    assert shard.features[:, 0].tolist() == [2.0, 0.0, 6.0]


def test_vision_stage_encodes_only_available_clips(tmp_path):
    records = make_records(3)
    available = {"0.mp4": True, "1.mp4": False, "2.mp4": True}
    extractor = FakeVisionExtractor()

    def load_clip(path):
        value = int(str(path).split(".")[0]) + 1
        clip = np.full((16, 4, 4, 3), value, dtype=np.uint8)
        return clip, available[str(path)]

    extract_vision_stage(
        records,
        tmp_path,
        lambda: extractor,
        prepared_loader=load_clip,
        shard_size=3,
        batch_size=8,
        workers=2,
        queue_capacity=2,
        executor_factory=ThreadPoolExecutor,
    )

    ids = np.asarray([record.sample_id for record in records])
    shard = ModalityStore(tmp_path, "vision", 512).read_verified(
        "emotiontalk", "validation", 0, ids
    )
    assert extractor.encoded_values == [1, 3]
    assert shard.available.tolist() == [True, False, True]
    assert shard.features[:, 0].tolist() == [1.0, 0.0, 3.0]


def test_audio_stage_records_correct_sample_for_later_shard_failure(tmp_path):
    records = make_records(5)

    def load_waveform(path):
        if str(path) == "3.mp4":
            raise RuntimeError("decode failed")
        return np.ones(2, dtype=np.float32)

    with pytest.raises(CpuWorkerError, match="decode failed"):
        extract_audio_stage(
            records,
            tmp_path,
            FakeAudioExtractor,
            waveform_loader=load_waveform,
            shard_size=2,
            batch_size=2,
            workers=2,
            queue_capacity=2,
            executor_factory=ThreadPoolExecutor,
        )

    error_path = tmp_path / "staging" / "errors.jsonl"
    error = json.loads(error_path.read_text(encoding="utf-8").splitlines()[-1])
    assert error["modality"] == "audio"
    assert error["sample_id"] == records[3].sample_id
    assert not ModalityStore(tmp_path, "audio", 1024).path(
        "emotiontalk", "validation", 1
    ).exists()


def test_parallel_runner_overlaps_branches_and_merges_after_both(tmp_path):
    barrier = threading.Barrier(2)
    completed = set()
    lock = threading.Lock()

    class ProbeRunner(ParallelFeatureExtractionRunner):
        def _run_text_then_audio(self, records, completed_shards):
            del records, completed_shards
            barrier.wait(timeout=2)
            with lock:
                completed.add("gpu0")

        def _run_vision(self, records, completed_shards):
            del records, completed_shards
            barrier.wait(timeout=2)
            with lock:
                completed.add("gpu1")

        def _merge_all(self, records, final_store):
            del records, final_store
            assert completed == {"gpu0", "gpu1"}
            return []

    runner = ProbeRunner(
        staging_root=tmp_path,
        config=ParallelFeatureExtractionConfig(shard_size=2),
        text_extractor_factory=MustNotConstruct(),
        audio_extractor_factory=MustNotConstruct(),
        vision_extractor_factory=MustNotConstruct(),
        waveform_loader=lambda _path: np.ones(2, dtype=np.float32),
        prepared_loader=lambda _path: (np.ones((16, 4, 4, 3)), True),
        audio_executor_factory=ThreadPoolExecutor,
        vision_executor_factory=ThreadPoolExecutor,
    )
    assert runner.run(make_records(2), FeatureStore(tmp_path / "final")) == []


def test_parallel_runner_does_not_merge_after_branch_failure(tmp_path):
    class FailingRunner(ParallelFeatureExtractionRunner):
        merge_calls = 0

        def _run_text_then_audio(self, records, completed_shards):
            del records, completed_shards
            raise RuntimeError("audio failed")

        def _run_vision(self, records, completed_shards):
            del records, completed_shards

        def _merge_all(self, records, final_store):
            del records, final_store
            self.merge_calls += 1
            return []

    runner = FailingRunner(
        staging_root=tmp_path,
        config=ParallelFeatureExtractionConfig(shard_size=2),
        text_extractor_factory=MustNotConstruct(),
        audio_extractor_factory=MustNotConstruct(),
        vision_extractor_factory=MustNotConstruct(),
        waveform_loader=lambda _path: np.ones(2, dtype=np.float32),
        prepared_loader=lambda _path: (np.ones((16, 4, 4, 3)), True),
        audio_executor_factory=ThreadPoolExecutor,
        vision_executor_factory=ThreadPoolExecutor,
    )
    with pytest.raises(RuntimeError, match="audio failed"):
        runner.run(make_records(2), FeatureStore(tmp_path / "final"))
    assert runner.merge_calls == 0


def _combine_shards(store, dataset="emotiontalk", split="validation"):
    shards = store.read_all(dataset, split)
    return {
        "sample_ids": np.concatenate([shard.sample_ids for shard in shards]),
        "text": np.concatenate([shard.text for shard in shards]),
        "audio": np.concatenate([shard.audio for shard in shards]),
        "vision": np.concatenate([shard.vision for shard in shards]),
        "modality_mask": np.concatenate(
            [shard.modality_mask for shard in shards]
        ),
    }


def test_parallel_features_match_serial_for_sixteen_samples(tmp_path):
    records = make_records(16)

    def load_waveform(path):
        value = int(path.stem) + 1
        return np.asarray([value], dtype=np.float32)

    def load_serial_vision(path):
        value = int(path.stem) + 1
        available = value % 4 != 0
        return np.full(512, value, dtype=np.float32), available

    serial_store = FeatureStore(tmp_path / "serial")
    DatasetFeatureExtractionRunner(
        text_extractor=FakeTextExtractor(),
        audio_extractor=FakeAudioExtractor(),
        waveform_loader=load_waveform,
        vision_loader=load_serial_vision,
    ).run(records, serial_store, shard_size=4)

    def prepare_parallel_vision(path):
        value = int(path.stem) + 1
        clip = np.full((16, 4, 4, 3), value, dtype=np.uint8)
        return clip, value % 4 != 0

    parallel_store = FeatureStore(tmp_path / "parallel")
    runner = ParallelFeatureExtractionRunner(
        staging_root=tmp_path / "parallel",
        config=ParallelFeatureExtractionConfig(
            shard_size=4,
            text_batch_size=8,
            audio_batch_size=2,
            vision_batch_size=3,
            audio_workers=2,
            vision_workers=2,
            queue_capacity=2,
        ),
        text_extractor_factory=FakeTextExtractor,
        audio_extractor_factory=FakeAudioExtractor,
        vision_extractor_factory=FakeVisionExtractor,
        waveform_loader=load_waveform,
        prepared_loader=prepare_parallel_vision,
        audio_executor_factory=ThreadPoolExecutor,
        vision_executor_factory=ThreadPoolExecutor,
    )
    runner.run(records, parallel_store)

    old = _combine_shards(serial_store)
    new = _combine_shards(parallel_store)
    np.testing.assert_array_equal(new["sample_ids"], old["sample_ids"])
    np.testing.assert_array_equal(new["modality_mask"], old["modality_mask"])
    for modality in ("text", "audio", "vision"):
        np.testing.assert_allclose(
            new[modality], old[modality], rtol=1e-4, atol=1e-5
        )


def test_verified_final_shards_skip_all_parallel_factories(tmp_path):
    records = make_records(2)
    final_store = FeatureStore(tmp_path / "final")
    staging_root = tmp_path / "staging-root"
    for modality, width in (("text", 768), ("audio", 1024), ("vision", 512)):
        ModalityStore(staging_root, modality, width).write(
            "emotiontalk",
            "validation",
            0,
            ModalityShard(
                sample_ids=np.asarray([record.sample_id for record in records]),
                features=np.ones((2, width), dtype=np.float32),
                available=np.ones(2, dtype=np.bool_),
            ),
        )
    runner = ParallelFeatureExtractionRunner(
        staging_root=staging_root,
        config=ParallelFeatureExtractionConfig(shard_size=2),
        text_extractor_factory=MustNotConstruct(),
        audio_extractor_factory=MustNotConstruct(),
        vision_extractor_factory=MustNotConstruct(),
        waveform_loader=lambda _path: np.ones(2, dtype=np.float32),
        prepared_loader=lambda _path: (np.ones((16, 4, 4, 3)), True),
        audio_executor_factory=ThreadPoolExecutor,
        vision_executor_factory=ThreadPoolExecutor,
    )
    runner.run(records, final_store)

    for path in (staging_root / "staging").rglob("*.npz"):
        path.unlink()
    assert runner.run(records, final_store) == [
        final_store.path("emotiontalk", "validation", 0)
    ]
