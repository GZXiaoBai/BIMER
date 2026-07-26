from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .feature_store import FeatureStore
from .labels import EMOTION_LABELS
from .modality_store import MODALITY_DIMS
from .schema import UtteranceRecord


def compute_feature_statistics(
    records: Iterable[UtteranceRecord],
    store: FeatureStore,
    *,
    dataset: str,
    split: str,
    expected_dims: Mapping[str, int] = MODALITY_DIMS,
) -> dict[str, object]:
    selected = [
        record for record in records if record.dataset == dataset and str(record.split) == split
    ]
    if not selected:
        raise ValueError(f"manifest has no records for {dataset} {split}")
    manifest_ids = [record.sample_id for record in selected]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest sample IDs must be unique")

    paths = store.paths(dataset, split)
    if not paths:
        raise ValueError(f"feature store has no shards for {dataset} {split}")

    accumulators = {
        modality: {
            "available_count": 0,
            "unavailable_count": 0,
            "available_zero_vector_count": 0,
            "nonfinite_row_count": 0,
            "norm_sum": 0.0,
            "norm_square_sum": 0.0,
        }
        for modality in ("text", "audio", "vision")
    }
    observed_ids: list[str] = []
    modality_indices = {"text": 0, "audio": 1, "vision": 2}

    for path in paths:
        shard = store.read(path)
        sample_ids = shard.sample_ids.astype(str).tolist()
        duplicates = set(observed_ids).intersection(sample_ids)
        if duplicates:
            preview = ", ".join(sorted(duplicates)[:3])
            raise ValueError(f"duplicate cached feature IDs: {preview}")
        observed_ids.extend(sample_ids)

        for modality, modality_index in modality_indices.items():
            values = np.asarray(getattr(shard, modality))
            expected_width = int(expected_dims[modality])
            if values.shape != (len(sample_ids), expected_width):
                raise ValueError(
                    f"{modality} features in {path} must have shape "
                    f"[{len(sample_ids)}, {expected_width}]"
                )
            available = np.asarray(shard.modality_mask[:, modality_index], dtype=np.bool_)
            finite_rows = np.isfinite(values).all(axis=1)
            nonfinite_count = int((~finite_rows).sum())
            if nonfinite_count:
                raise ValueError(f"{modality} features in {path} contain non-finite rows")
            norms = np.linalg.norm(values, axis=1)
            active_norms = norms[available]
            accumulator = accumulators[modality]
            accumulator["available_count"] += int(available.sum())
            accumulator["unavailable_count"] += int((~available).sum())
            accumulator["available_zero_vector_count"] += int(np.isclose(active_norms, 0.0).sum())
            accumulator["nonfinite_row_count"] += nonfinite_count
            accumulator["norm_sum"] += float(active_norms.sum(dtype=np.float64))
            accumulator["norm_square_sum"] += float(
                np.square(active_norms, dtype=np.float64).sum(dtype=np.float64)
            )

    observed_set = set(observed_ids)
    manifest_set = set(manifest_ids)
    modality_report: dict[str, dict[str, object]] = {}
    for modality, accumulator in accumulators.items():
        available_count = int(accumulator["available_count"])
        norm_sum = float(accumulator["norm_sum"])
        norm_square_sum = float(accumulator["norm_square_sum"])
        mean_norm = norm_sum / available_count if available_count else 0.0
        variance = (
            max(0.0, norm_square_sum / available_count - mean_norm * mean_norm)
            if available_count
            else 0.0
        )
        modality_report[modality] = {
            "dimension": int(expected_dims[modality]),
            "available_count": available_count,
            "unavailable_count": int(accumulator["unavailable_count"]),
            "available_rate": available_count / len(observed_ids),
            "available_zero_vector_count": int(accumulator["available_zero_vector_count"]),
            "nonfinite_row_count": int(accumulator["nonfinite_row_count"]),
            "mean_l2_norm": mean_norm,
            "std_l2_norm": variance**0.5,
        }

    label_counts = Counter(str(record.emotion) for record in selected)
    return {
        "dataset": dataset,
        "split": split,
        "sample_count": len(selected),
        "feature_sample_count": len(observed_ids),
        "shard_count": len(paths),
        "label_counts": {label: int(label_counts.get(label, 0)) for label in EMOTION_LABELS},
        "missing_manifest_samples": len(manifest_set - observed_set),
        "unexpected_feature_samples": len(observed_set - manifest_set),
        "modalities": modality_report,
    }


def write_feature_statistics(report: Mapping[str, object], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
