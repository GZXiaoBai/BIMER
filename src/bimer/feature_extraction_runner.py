from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .feature_extractors import prepare_audio_waveforms
from .feature_store import FeatureShard, FeatureStore
from .quality import audio_quality, text_quality
from .schema import UtteranceRecord


class BatchEncoder(Protocol):
    def encode(self, values: Sequence[object]) -> np.ndarray: ...


def load_full_waveform(video_path: Path) -> np.ndarray:
    path = Path(video_path)
    if not path.is_file():
        return np.empty(0, dtype=np.float32)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "-",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        warnings.warn(
            f"Audio unavailable for {path}: ffmpeg exited {error.returncode}",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


class DatasetFeatureExtractionRunner:
    def __init__(
        self,
        *,
        text_extractor: BatchEncoder,
        audio_extractor: BatchEncoder,
        waveform_loader: Callable[[Path], np.ndarray] = load_full_waveform,
        vision_loader: Callable[
            [Path],
            tuple[np.ndarray, bool] | tuple[np.ndarray, bool, np.ndarray],
        ],
    ) -> None:
        self.text_extractor = text_extractor
        self.audio_extractor = audio_extractor
        self.waveform_loader = waveform_loader
        self.vision_loader = vision_loader

    def run(
        self,
        records: Sequence[UtteranceRecord],
        store: FeatureStore,
        *,
        shard_size: int = 1024,
    ) -> list[Path]:
        if not records:
            return []
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        groups = {(record.dataset, str(record.split)) for record in records}
        if len(groups) != 1:
            raise ValueError("extract one dataset split at a time")
        dataset, split = next(iter(groups))
        written: list[Path] = []
        for shard_index, start in enumerate(range(0, len(records), shard_size)):
            chunk = records[start : start + shard_size]
            expected_sample_ids = np.asarray([record.sample_id for record in chunk], dtype=str)
            shard_path = store.path(dataset, split, shard_index)
            if shard_path.is_file():
                existing = store.read(shard_path)
                if not np.array_equal(existing.sample_ids.astype(str), expected_sample_ids):
                    raise ValueError(f"existing shard {shard_path} has unexpected sample IDs")
                written.append(shard_path)
                continue
            video_paths: list[Path] = []
            for record in chunk:
                if record.video_path is None:
                    raise ValueError(f"record {record.sample_id} has no video_path")
                video_paths.append(Path(record.video_path))

            text = self.text_extractor.encode([record.text for record in chunk])
            waveforms = [self.waveform_loader(path) for path in video_paths]
            safe_waveforms, audio_available = prepare_audio_waveforms(waveforms)
            audio = self.audio_extractor.encode(safe_waveforms)
            vision_features: list[np.ndarray] = []
            vision_available: list[bool] = []
            vision_quality_rows: list[np.ndarray] = []
            for path in video_paths:
                vision_result = self.vision_loader(path)
                feature, available = vision_result[:2]
                quality = (
                    np.asarray(vision_result[2], dtype=np.float32)
                    if len(vision_result) == 3
                    else np.full(4, float(available), dtype=np.float32)
                )
                flattened = np.asarray(feature, dtype=np.float32).reshape(-1)
                if not available:
                    flattened = np.zeros_like(flattened)
                vision_features.append(flattened)
                vision_available.append(available)
                vision_quality_rows.append(quality)
            vision = np.stack(vision_features)
            modality_mask = np.stack(
                (
                    np.ones(len(chunk), dtype=np.bool_),
                    audio_available.astype(np.bool_),
                    np.asarray(vision_available, dtype=np.bool_),
                ),
                axis=-1,
            )
            modality_quality = np.stack(
                (
                    np.stack(
                        [
                            text_quality(
                                record.text,
                                source=record.text_source,
                                asr_confidence=record.asr_confidence,
                            )
                            for record in chunk
                        ]
                    ),
                    np.stack([audio_quality(waveform) for waveform in waveforms]),
                    np.stack(vision_quality_rows),
                ),
                axis=1,
            ).astype(np.float32)
            shard = FeatureShard(
                sample_ids=expected_sample_ids,
                text=np.asarray(text, dtype=np.float32),
                audio=np.asarray(audio, dtype=np.float32),
                vision=vision,
                modality_mask=modality_mask,
                modality_quality=modality_quality,
            )
            written.append(store.write(dataset, split, shard_index, shard))
        return written
