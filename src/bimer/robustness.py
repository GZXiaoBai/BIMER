from __future__ import annotations

import numpy as np


def add_noise_at_snr(
    waveform: np.ndarray,
    *,
    snr_db: float,
    seed: int = 42,
) -> np.ndarray:
    signal = np.asarray(waveform, dtype=np.float32)
    signal_power = float(np.mean(signal**2))
    if signal_power <= 0:
        raise ValueError("cannot add calibrated noise to a silent waveform")
    generator = np.random.default_rng(seed)
    noise = generator.normal(size=signal.shape).astype(np.float32)
    raw_noise_power = float(np.mean(noise**2))
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(target_noise_power / raw_noise_power)
    return signal + noise


def drop_video_frames(
    frames: np.ndarray,
    *,
    fraction: float,
    seed: int = 42,
) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    output = np.asarray(frames).copy()
    count = int(round(output.shape[0] * fraction))
    if count:
        generator = np.random.default_rng(seed)
        indices = generator.choice(output.shape[0], size=count, replace=False)
        output[indices] = 0
    return output


def mask_feature_modality(
    feature: np.ndarray,
    modality_mask: np.ndarray,
    *,
    modality_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if modality_index not in {0, 1, 2}:
        raise ValueError("modality_index must be 0, 1, or 2")
    cleared = np.zeros_like(feature)
    updated = np.asarray(modality_mask, dtype=np.bool_).copy()
    updated[:, modality_index] = False
    return cleared, updated
