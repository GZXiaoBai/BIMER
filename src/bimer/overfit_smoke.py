from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .baselines import UnimodalClassifier
from .feature_store import FeatureStore
from .labels import EMOTION_LABELS, emotion_index
from .losses import masked_weighted_cross_entropy
from .schema import UtteranceRecord
from .training import (
    DialogueExample,
    MultimodalBatch,
    collate_dialogues,
    evaluate_batches,
    train_epoch,
)

MODALITIES = ("text", "audio", "vision")


def build_overfit_example(
    records: Iterable[UtteranceRecord],
    store: FeatureStore,
    *,
    dataset: str,
    split: str,
    modality: str,
    sample_count: int = 16,
) -> DialogueExample:
    if modality not in MODALITIES:
        raise ValueError("modality must be text, audio, or vision")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    selected_records = [
        record for record in records if record.dataset == dataset and str(record.split) == split
    ]
    record_by_id = {record.sample_id: record for record in selected_records}
    if len(record_by_id) != len(selected_records):
        raise ValueError("manifest sample IDs must be unique")
    if not record_by_id:
        raise ValueError(f"manifest has no records for {dataset} {split}")

    modality_index = {"text": 0, "audio": 1, "vision": 2}[modality]
    rows: list[
        tuple[
            UtteranceRecord,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    seen: set[str] = set()
    for path in store.paths(dataset, split):
        shard = store.read(path)
        for index, raw_sample_id in enumerate(shard.sample_ids.tolist()):
            sample_id = str(raw_sample_id)
            if sample_id in seen or sample_id not in record_by_id:
                continue
            if not bool(shard.modality_mask[index, modality_index]):
                continue
            seen.add(sample_id)
            rows.append(
                (
                    record_by_id[sample_id],
                    shard.text[index],
                    shard.audio[index],
                    shard.vision[index],
                    shard.modality_mask[index],
                )
            )
            if len(rows) == sample_count:
                break
        if len(rows) == sample_count:
            break
    if len(rows) != sample_count:
        raise ValueError(f"only {len(rows)} {modality} rows are available; {sample_count} required")

    languages = {str(row[0].language) for row in rows}
    if len(languages) != 1:
        raise ValueError("overfit examples must use one language")
    return DialogueExample(
        dataset=dataset,
        sample_ids=tuple(row[0].sample_id for row in rows),
        text=np.stack([row[1] for row in rows]).astype(np.float32),
        audio=np.stack([row[2] for row in rows]).astype(np.float32),
        vision=np.stack([row[3] for row in rows]).astype(np.float32),
        modality_mask=np.stack([row[4] for row in rows]).astype(np.bool_),
        labels=np.asarray([emotion_index(str(row[0].emotion)) for row in rows], dtype=np.int64),
        language_id=0 if next(iter(languages)) == "en" else 1,
    )


@torch.no_grad()
def _batch_loss(model: nn.Module, batch: MultimodalBatch, *, device: torch.device) -> float:
    model.eval()
    device_batch = batch.to(device)
    output = model(**device_batch.model_inputs())
    loss = masked_weighted_cross_entropy(
        output.logits,
        device_batch.labels,
        device_batch.attention_mask,
    )
    return float(loss.cpu())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_unimodal_overfit_smoke(
    records: Iterable[UtteranceRecord],
    store: FeatureStore,
    *,
    dataset: str,
    split: str,
    modalities: Sequence[str] = MODALITIES,
    sample_count: int = 16,
    max_epochs: int = 200,
    learning_rate: float = 1e-2,
    target_accuracy: float = 0.95,
    hidden_dim: int = 64,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict[str, object]:
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1]")
    requested = tuple(modalities)
    if not requested or any(modality not in MODALITIES for modality in requested):
        raise ValueError("modalities must contain text, audio, or vision")
    materialized_records = list(records)
    reports: dict[str, dict[str, object]] = {}

    for modality in requested:
        _set_seed(seed)
        example = build_overfit_example(
            materialized_records,
            store,
            dataset=dataset,
            split=split,
            modality=modality,
            sample_count=sample_count,
        )
        batch = collate_dialogues([example])
        input_dim = int(getattr(example, modality).shape[1])
        model = UnimodalClassifier(
            modality,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=len(EMOTION_LABELS),
            dropout=0.0,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        initial_loss = _batch_loss(model, batch, device=device)
        accuracy = 0.0
        epochs = 0
        for epoch in range(1, max_epochs + 1):
            train_epoch(model, [batch], optimizer, device=device)
            evaluation = evaluate_batches(
                model,
                [batch],
                device=device,
                label_names=EMOTION_LABELS,
            )
            accuracy = float(evaluation.metrics["accuracy"])
            epochs = epoch
            if accuracy >= target_accuracy:
                break
        final_loss = _batch_loss(model, batch, device=device)
        reports[modality] = {
            "sample_count": sample_count,
            "sample_ids": list(example.sample_ids),
            "class_count": int(np.unique(example.labels).size),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "accuracy": accuracy,
            "epochs": epochs,
            "passed": bool(
                np.isfinite(final_loss)
                and final_loss < initial_loss
                and accuracy >= target_accuracy
            ),
        }

    return {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "sample_count": sample_count,
        "target_accuracy": target_accuracy,
        "device": str(device),
        "all_passed": all(bool(report["passed"]) for report in reports.values()),
        "modalities": reports,
    }


def write_overfit_smoke(report: Mapping[str, object], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
