from __future__ import annotations

from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import gc
from itertools import islice
import json
from pathlib import Path
import threading
import time
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

from .feature_store import FeatureStore
from .modality_store import (
    ModalityShard,
    ModalityStore,
    merge_staged_shard,
    verified_final_shard,
)
from .schema import UtteranceRecord


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
_ERROR_LOCK = threading.Lock()


class CpuWorkerError(RuntimeError):
    def __init__(self, index: int, error: BaseException) -> None:
        super().__init__(f"CPU worker failed at input {index}: {error}")
        self.index = index


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
    )


def encode_adaptive(
    encoder: object,
    values: Sequence[object],
    *,
    initial_batch_size: int,
) -> np.ndarray:
    if initial_batch_size <= 0:
        raise ValueError("initial_batch_size must be positive")
    batch_size = initial_batch_size
    while True:
        try:
            return encoder.encode(values, batch_size=batch_size)  # type: ignore[attr-defined]
        except Exception as error:
            if not _is_cuda_oom(error) or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def prefetched_map(
    function: Callable[[InputT], OutputT],
    values: Iterable[InputT],
    *,
    workers: int,
    queue_capacity: int,
    executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
) -> Iterator[OutputT]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if queue_capacity <= 0:
        raise ValueError("queue_capacity must be positive")
    source = iter(enumerate(values))
    pending = deque()
    with executor_factory(max_workers=workers) as executor:
        for index, value in islice(source, workers + queue_capacity):
            pending.append((index, executor.submit(function, value)))
        try:
            while pending:
                index, future = pending.popleft()
                try:
                    yield future.result()
                except Exception as error:
                    for _, other in pending:
                        other.cancel()
                    raise CpuWorkerError(index, error) from error
                replacement = next(source, None)
                if replacement is not None:
                    next_index, next_value = replacement
                    pending.append(
                        (next_index, executor.submit(function, next_value))
                    )
        finally:
            for _, future in pending:
                future.cancel()


def record_shards(
    records: Sequence[UtteranceRecord],
    shard_size: int,
) -> Iterator[tuple[int, Sequence[UtteranceRecord], np.ndarray]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        chunk = records[start : start + shard_size]
        sample_ids = np.asarray(
            [record.sample_id for record in chunk], dtype=str
        )
        yield shard_index, chunk, sample_ids


def _dataset_split(
    records: Sequence[UtteranceRecord],
) -> tuple[str, str] | None:
    if not records:
        return None
    groups = {(record.dataset, str(record.split)) for record in records}
    if len(groups) != 1:
        raise ValueError("extract one dataset split at a time")
    return next(iter(groups))


def _pending_shards(
    records: Sequence[UtteranceRecord],
    store: ModalityStore,
    *,
    shard_size: int,
    completed_shards: set[int] | frozenset[int],
) -> list[tuple[int, Sequence[UtteranceRecord], np.ndarray]]:
    group = _dataset_split(records)
    if group is None:
        return []
    dataset, split = group
    pending = []
    for shard_index, chunk, sample_ids in record_shards(records, shard_size):
        if shard_index in completed_shards:
            continue
        path = store.path(dataset, split, shard_index)
        if path.is_file():
            store.read_verified(dataset, split, shard_index, sample_ids)
            continue
        pending.append((shard_index, chunk, sample_ids))
    return pending


def validate_stage_output(
    modality: str,
    features: np.ndarray,
    rows: int,
    width: int,
) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32)
    if output.shape != (rows, width):
        raise ValueError(
            f"{modality} returned {output.shape}, expected {(rows, width)}"
        )
    if not np.isfinite(output).all():
        raise ValueError(f"{modality} returned non-finite features")
    return output


def _report_progress(
    modality: str,
    *,
    completed: int,
    total: int,
    shard_index: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    print(
        f"[bimer:{modality}] shard={shard_index} "
        f"samples={completed}/{total} rate={completed / elapsed:.2f}/s "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def _record_cpu_error(
    staging_root: Path | str,
    modality: str,
    record: UtteranceRecord,
    error: CpuWorkerError,
) -> None:
    cause = error.__cause__ or error
    payload = {
        "modality": modality,
        "sample_id": record.sample_id,
        "path": str(record.video_path or record.audio_path or ""),
        "error_type": type(cause).__name__,
        "message": str(cause),
    }
    path = Path(staging_root) / "staging" / "errors.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ERROR_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_text_stage(
    records: Sequence[UtteranceRecord],
    staging_root: Path | str,
    extractor_factory: Callable[[], object],
    *,
    shard_size: int,
    batch_size: int,
    completed_shards: set[int] | frozenset[int] = frozenset(),
) -> list[Path]:
    store = ModalityStore(staging_root, "text", 768)
    pending = _pending_shards(
        records,
        store,
        shard_size=shard_size,
        completed_shards=completed_shards,
    )
    if not pending:
        return []
    extractor = extractor_factory()
    written = []
    total = sum(len(chunk) for _, chunk, _ in pending)
    completed = 0
    started_at = time.monotonic()
    for shard_index, chunk, sample_ids in pending:
        features = encode_adaptive(
            extractor,
            [record.text for record in chunk],
            initial_batch_size=batch_size,
        )
        features = validate_stage_output(
            "text", features, len(chunk), store.output_dim
        )
        written.append(
            store.write(
                chunk[0].dataset,
                str(chunk[0].split),
                shard_index,
                ModalityShard(
                    sample_ids,
                    features,
                    np.ones(len(chunk), dtype=np.bool_),
                ),
            )
        )
        completed += len(chunk)
        _report_progress(
            "text",
            completed=completed,
            total=total,
            shard_index=shard_index,
            started_at=started_at,
        )
    return written


def _media_paths(
    shards: Sequence[tuple[int, Sequence[UtteranceRecord], np.ndarray]],
) -> tuple[list[Path], list[UtteranceRecord]]:
    paths: list[Path] = []
    flattened_records: list[UtteranceRecord] = []
    for _, chunk, _ in shards:
        for record in chunk:
            if record.video_path is None:
                raise ValueError(f"record {record.sample_id} has no video_path")
            paths.append(Path(record.video_path))
            flattened_records.append(record)
    return paths, flattened_records


def extract_audio_stage(
    records: Sequence[UtteranceRecord],
    staging_root: Path | str,
    extractor_factory: Callable[[], object],
    *,
    waveform_loader: Callable[[Path], np.ndarray],
    shard_size: int,
    batch_size: int,
    workers: int,
    queue_capacity: int,
    executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
    completed_shards: set[int] | frozenset[int] = frozenset(),
) -> list[Path]:
    store = ModalityStore(staging_root, "audio", 1024)
    pending = _pending_shards(
        records,
        store,
        shard_size=shard_size,
        completed_shards=completed_shards,
    )
    if not pending:
        return []
    paths, flattened_records = _media_paths(pending)
    decoded = iter(
        prefetched_map(
            waveform_loader,
            paths,
            workers=workers,
            queue_capacity=queue_capacity,
            executor_factory=executor_factory,
        )
    )
    extractor = extractor_factory()
    written: list[Path] = []
    consumed = 0
    total = sum(len(chunk) for _, chunk, _ in pending)
    started_at = time.monotonic()
    try:
        for shard_index, chunk, sample_ids in pending:
            waveforms = [next(decoded) for _ in chunk]
            consumed += len(chunk)
            available = np.asarray(
                [np.asarray(waveform).size > 0 for waveform in waveforms],
                dtype=np.bool_,
            )
            safe_waveforms = [
                np.asarray(waveform, dtype=np.float32)
                if np.asarray(waveform).size
                else np.zeros(160, dtype=np.float32)
                for waveform in waveforms
            ]
            features = encode_adaptive(
                extractor,
                safe_waveforms,
                initial_batch_size=batch_size,
            )
            features = validate_stage_output(
                "audio", features, len(chunk), store.output_dim
            )
            features[~available] = 0.0
            written.append(
                store.write(
                    chunk[0].dataset,
                    str(chunk[0].split),
                    shard_index,
                    ModalityShard(sample_ids, features, available),
                )
            )
            _report_progress(
                "audio",
                completed=consumed,
                total=total,
                shard_index=shard_index,
                started_at=started_at,
            )
    except CpuWorkerError as error:
        record_index = min(error.index, len(flattened_records) - 1)
        _record_cpu_error(
            staging_root,
            "audio",
            flattened_records[record_index],
            error,
        )
        raise
    finally:
        decoded.close()
    return written


class _VisionEncoderAdapter:
    def __init__(self, extractor: object) -> None:
        self.extractor = extractor

    def encode(
        self,
        clips: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> np.ndarray:
        return self.extractor.encode_clips(  # type: ignore[attr-defined]
            clips, batch_size=batch_size
        )


def extract_vision_stage(
    records: Sequence[UtteranceRecord],
    staging_root: Path | str,
    extractor_factory: Callable[[], object],
    *,
    prepared_loader: Callable[[Path], tuple[np.ndarray, bool]],
    shard_size: int,
    batch_size: int,
    workers: int,
    queue_capacity: int,
    executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
    completed_shards: set[int] | frozenset[int] = frozenset(),
) -> list[Path]:
    store = ModalityStore(staging_root, "vision", 512)
    pending = _pending_shards(
        records,
        store,
        shard_size=shard_size,
        completed_shards=completed_shards,
    )
    if not pending:
        return []
    paths, flattened_records = _media_paths(pending)
    prepared_iterator = iter(
        prefetched_map(
            prepared_loader,
            paths,
            workers=workers,
            queue_capacity=queue_capacity,
            executor_factory=executor_factory,
        )
    )
    adapter = _VisionEncoderAdapter(extractor_factory())
    written: list[Path] = []
    consumed = 0
    total = sum(len(chunk) for _, chunk, _ in pending)
    started_at = time.monotonic()
    try:
        for shard_index, chunk, sample_ids in pending:
            prepared = [next(prepared_iterator) for _ in chunk]
            consumed += len(chunk)
            available = np.asarray(
                [item[1] for item in prepared], dtype=np.bool_
            )
            features = np.zeros(
                (len(chunk), store.output_dim), dtype=np.float32
            )
            positions = np.flatnonzero(available)
            if positions.size:
                clips = [prepared[int(index)][0] for index in positions]
                encoded = encode_adaptive(
                    adapter,
                    clips,
                    initial_batch_size=batch_size,
                )
                encoded = validate_stage_output(
                    "vision", encoded, len(clips), store.output_dim
                )
                features[positions] = encoded
            written.append(
                store.write(
                    chunk[0].dataset,
                    str(chunk[0].split),
                    shard_index,
                    ModalityShard(sample_ids, features, available),
                )
            )
            _report_progress(
                "vision",
                completed=consumed,
                total=total,
                shard_index=shard_index,
                started_at=started_at,
            )
    except CpuWorkerError as error:
        record_index = min(error.index, len(flattened_records) - 1)
        _record_cpu_error(
            staging_root,
            "vision",
            flattened_records[record_index],
            error,
        )
        raise
    finally:
        prepared_iterator.close()
    return written


@dataclass(frozen=True, slots=True)
class ParallelFeatureExtractionConfig:
    shard_size: int = 1024
    text_batch_size: int = 64
    audio_batch_size: int = 8
    vision_batch_size: int = 8
    audio_workers: int = 4
    vision_workers: int = 4
    queue_capacity: int = 8

    def __post_init__(self) -> None:
        for name in (
            "shard_size",
            "text_batch_size",
            "audio_batch_size",
            "vision_batch_size",
            "audio_workers",
            "vision_workers",
            "queue_capacity",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class ParallelFeatureExtractionRunner:
    def __init__(
        self,
        *,
        staging_root: Path | str,
        config: ParallelFeatureExtractionConfig,
        text_extractor_factory: Callable[[], object],
        audio_extractor_factory: Callable[[], object],
        vision_extractor_factory: Callable[[], object],
        waveform_loader: Callable[[Path], np.ndarray],
        prepared_loader: Callable[[Path], tuple[np.ndarray, bool]],
        audio_executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
        vision_executor_factory: Callable[..., Executor] = ProcessPoolExecutor,
    ) -> None:
        self.staging_root = Path(staging_root)
        self.config = config
        self.text_extractor_factory = text_extractor_factory
        self.audio_extractor_factory = audio_extractor_factory
        self.vision_extractor_factory = vision_extractor_factory
        self.waveform_loader = waveform_loader
        self.prepared_loader = prepared_loader
        self.audio_executor_factory = audio_executor_factory
        self.vision_executor_factory = vision_executor_factory

    def _run_text_then_audio(
        self,
        records: Sequence[UtteranceRecord],
        completed_shards: set[int],
    ) -> None:
        extract_text_stage(
            records,
            self.staging_root,
            self.text_extractor_factory,
            shard_size=self.config.shard_size,
            batch_size=self.config.text_batch_size,
            completed_shards=completed_shards,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        extract_audio_stage(
            records,
            self.staging_root,
            self.audio_extractor_factory,
            waveform_loader=self.waveform_loader,
            shard_size=self.config.shard_size,
            batch_size=self.config.audio_batch_size,
            workers=self.config.audio_workers,
            queue_capacity=self.config.queue_capacity,
            executor_factory=self.audio_executor_factory,
            completed_shards=completed_shards,
        )

    def _run_vision(
        self,
        records: Sequence[UtteranceRecord],
        completed_shards: set[int],
    ) -> None:
        extract_vision_stage(
            records,
            self.staging_root,
            self.vision_extractor_factory,
            prepared_loader=self.prepared_loader,
            shard_size=self.config.shard_size,
            batch_size=self.config.vision_batch_size,
            workers=self.config.vision_workers,
            queue_capacity=self.config.queue_capacity,
            executor_factory=self.vision_executor_factory,
            completed_shards=completed_shards,
        )

    def _completed_final_shards(
        self,
        records: Sequence[UtteranceRecord],
        final_store: FeatureStore,
    ) -> set[int]:
        group = _dataset_split(records)
        if group is None:
            return set()
        dataset, split = group
        completed = set()
        for shard_index, _, sample_ids in record_shards(
            records, self.config.shard_size
        ):
            if verified_final_shard(
                final_store,
                dataset,
                split,
                shard_index,
                sample_ids,
            ) is not None:
                completed.add(shard_index)
        return completed

    def _merge_all(
        self,
        records: Sequence[UtteranceRecord],
        final_store: FeatureStore,
    ) -> list[Path]:
        group = _dataset_split(records)
        if group is None:
            return []
        dataset, split = group
        return [
            merge_staged_shard(
                staging_root=self.staging_root,
                final_store=final_store,
                dataset=dataset,
                split=split,
                shard_index=shard_index,
                expected_sample_ids=sample_ids,
            )
            for shard_index, _, sample_ids in record_shards(
                records, self.config.shard_size
            )
        ]

    def run(
        self,
        records: Sequence[UtteranceRecord],
        final_store: FeatureStore,
    ) -> list[Path]:
        if not records:
            return []
        _dataset_split(records)
        completed_shards = self._completed_final_shards(records, final_store)
        shard_count = sum(
            1 for _ in record_shards(records, self.config.shard_size)
        )
        if len(completed_shards) != shard_count:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="bimer-vision-gpu"
            ) as executor:
                gpu1 = executor.submit(
                    self._run_vision, records, completed_shards
                )
                # Keep Hugging Face text/audio model construction on the
                # calling thread. XLS-R can deadlock when CUDA is initialized
                # from a worker thread, while the independent vision branch
                # still overlaps on the second GPU.
                self._run_text_then_audio(records, completed_shards)
                gpu1.result()
        return self._merge_all(records, final_store)


def load_waveform_worker(
    video_path: Path,
    *,
    audio_snr: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    from .feature_extraction_runner import load_full_waveform
    from .robustness import add_noise_at_snr

    waveform = load_full_waveform(video_path)
    if audio_snr is not None and waveform.size:
        waveform = add_noise_at_snr(waveform, snr_db=audio_snr, seed=seed)
    return waveform


_VISION_FACE_CROPPER: object | None = None
_VISION_FRAME_DROP = 0.0
_VISION_SEED = 42


def initialize_vision_worker(
    yunet_model: Path | str,
    frame_drop_fraction: float = 0.0,
    seed: int = 42,
) -> None:
    from .feature_extractors import YuNetFaceCropper

    global _VISION_FACE_CROPPER, _VISION_FRAME_DROP, _VISION_SEED
    _VISION_FACE_CROPPER = YuNetFaceCropper(yunet_model)
    _VISION_FRAME_DROP = frame_drop_fraction
    _VISION_SEED = seed


def prepare_video_worker(video_path: Path) -> tuple[np.ndarray, bool]:
    from .feature_extractors import prepare_video_clip

    if _VISION_FACE_CROPPER is None:
        raise RuntimeError("vision worker was not initialized")
    return prepare_video_clip(
        video_path,
        face_cropper=_VISION_FACE_CROPPER,  # type: ignore[arg-type]
        frame_drop_fraction=_VISION_FRAME_DROP,
        seed=_VISION_SEED,
    )
