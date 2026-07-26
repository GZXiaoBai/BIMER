from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


MODALITY_QUALITY_NAMES: dict[str, tuple[str, str, str, str]] = {
    "text": ("source_human", "confidence", "length", "completeness"),
    "audio": ("duration", "rms", "voiced_ratio", "clipping_free"),
    "vision": ("face_ratio", "frame_ratio", "stability", "face_area"),
}


def _bounded(values: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def text_quality(
    text: str,
    *,
    source: Literal["human", "whisper"] = "human",
    asr_confidence: float | None = None,
    maximum_units: int = 128,
) -> np.ndarray:
    clean = text.strip()
    source_score = 1.0 if source == "human" else 0.0
    confidence = 1.0 if source == "human" else float(asr_confidence or 0.0)
    length_score = min(len(clean) / max(1, maximum_units), 1.0)
    completeness = 1.0 if clean else 0.0
    return _bounded((source_score, confidence, length_score, completeness))


def audio_quality(
    waveform: np.ndarray,
    *,
    sampling_rate: int = 16000,
    maximum_duration_seconds: float = 15.0,
) -> np.ndarray:
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration = signal.size / max(1, sampling_rate)
    duration_score = min(duration / maximum_duration_seconds, 1.0)
    if not signal.size:
        return _bounded((0.0, 0.0, 0.0, 0.0))
    rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    rms_score = float(np.clip((rms_db + 60.0) / 60.0, 0.0, 1.0))
    frame_size = max(1, int(round(sampling_rate * 0.02)))
    frame_count = int(np.ceil(signal.size / frame_size))
    padded = np.pad(signal, (0, frame_count * frame_size - signal.size))
    frame_rms = np.sqrt(np.mean(np.square(padded.reshape(frame_count, frame_size)), axis=1))
    voiced_ratio = float(np.mean(frame_rms >= 1e-3))
    clipping_free = 1.0 - float(np.mean(np.abs(signal) >= 0.99))
    return _bounded((duration_score, rms_score, voiced_ratio, clipping_free))


def vision_quality(
    detected_faces: np.ndarray | Sequence[bool],
    bboxes: Sequence[tuple[float, float, float, float] | None],
    *,
    expected_frames: int = 16,
) -> np.ndarray:
    detected = np.asarray(detected_faces, dtype=np.bool_)
    if len(bboxes) != len(detected):
        raise ValueError("bboxes and detected_faces must have equal length")
    face_ratio = float(detected.mean()) if detected.size else 0.0
    decoded_ratio = min(len(detected) / max(1, expected_frames), 1.0)
    valid = np.asarray([bbox for bbox in bboxes if bbox is not None], dtype=np.float64)
    if not len(valid):
        stability = 0.0
        area = 0.0
    else:
        centers = valid[:, :2] + valid[:, 2:] / 2.0
        areas = valid[:, 2] * valid[:, 3]
        area = float(np.mean(areas))
        if len(valid) == 1:
            stability = 0.5
        else:
            center_jitter = float(np.linalg.norm(np.std(centers, axis=0)))
            area_jitter = float(np.std(areas) / max(np.mean(areas), 1e-6))
            stability = float(np.exp(-(4.0 * center_jitter + 2.0 * area_jitter)))
    return _bounded((face_ratio, decoded_ratio, stability, area))
