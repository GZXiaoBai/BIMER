from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .feature_store import FeatureShard, FeatureStore
from .schema import UtteranceRecord


class BatchEncoder(Protocol):
    def encode(self, values: Sequence[object]) -> np.ndarray: ...


def load_full_waveform(video_path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
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
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


class DatasetFeatureExtractionRunner:
    def __init__(
        self,
        *,
        text_extractor: BatchEncoder,
        audio_extractor: BatchEncoder,
        waveform_loader: Callable[[Path], np.ndarray] = load_full_waveform,
        vision_loader: Callable[[Path], tuple[np.ndarray, bool]],
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
            video_paths: list[Path] = []
            for record in chunk:
                if record.video_path is None:
                    raise ValueError(f"record {record.sample_id} has no video_path")
                video_paths.append(Path(record.video_path))

            text = self.text_extractor.encode([record.text for record in chunk])
            waveforms = [self.waveform_loader(path) for path in video_paths]
            audio_available = np.asarray([waveform.size > 0 for waveform in waveforms])
            safe_waveforms = [
                waveform if waveform.size else np.zeros(160, dtype=np.float32)
                for waveform in waveforms
            ]
            audio = self.audio_extractor.encode(safe_waveforms)
            vision_features: list[np.ndarray] = []
            vision_available: list[bool] = []
            for path in video_paths:
                feature, available = self.vision_loader(path)
                flattened = np.asarray(feature, dtype=np.float32).reshape(-1)
                if not available:
                    flattened = np.zeros_like(flattened)
                vision_features.append(flattened)
                vision_available.append(available)
            vision = np.stack(vision_features)
            modality_mask = np.stack(
                (
                    np.ones(len(chunk), dtype=np.bool_),
                    audio_available.astype(np.bool_),
                    np.asarray(vision_available, dtype=np.bool_),
                ),
                axis=-1,
            )
            shard = FeatureShard(
                sample_ids=np.asarray([record.sample_id for record in chunk]),
                text=np.asarray(text, dtype=np.float32),
                audio=np.asarray(audio, dtype=np.float32),
                vision=vision,
                modality_mask=modality_mask,
            )
            written.append(store.write(dataset, split, shard_index, shard))
        return written
