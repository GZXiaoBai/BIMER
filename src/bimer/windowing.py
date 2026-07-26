from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .schema import UtteranceRecord


@dataclass(frozen=True, slots=True)
class ContextWindow:
    records: tuple[UtteranceRecord, ...]


def make_context_windows(
    records: Iterable[UtteranceRecord],
    *,
    max_length: int = 32,
    overlap: int = 8,
) -> list[ContextWindow]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if overlap < 0 or overlap >= max_length:
        raise ValueError("overlap must satisfy 0 <= overlap < max_length")

    grouped: dict[tuple[str, str, str], list[UtteranceRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.dataset, str(record.split), record.effective_context_id)].append(record)

    windows: list[ContextWindow] = []
    stride = max_length - overlap
    for key in sorted(grouped):
        dialogue = sorted(grouped[key], key=lambda item: item.utterance_id)
        for start in range(0, len(dialogue), stride):
            chunk = dialogue[start : start + max_length]
            if not chunk:
                continue
            windows.append(ContextWindow(tuple(chunk)))
            if start + max_length >= len(dialogue):
                break
    return windows


def merge_window_probabilities(
    windows: Sequence[ContextWindow],
    predictions: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    if len(windows) != len(predictions):
        raise ValueError("windows and predictions must have equal length")

    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for window, prediction in zip(windows, predictions, strict=True):
        if prediction.shape[0] != len(window.records):
            raise ValueError("prediction length does not match its context window")
        for record, probabilities in zip(window.records, prediction, strict=True):
            values[record.sample_id].append(np.asarray(probabilities, dtype=np.float64))

    return {
        sample_id: np.mean(np.stack(sample_predictions), axis=0)
        for sample_id, sample_predictions in values.items()
    }
