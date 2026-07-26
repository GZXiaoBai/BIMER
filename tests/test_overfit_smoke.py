import json

import numpy as np
import torch

from bimer.feature_store import FeatureShard, FeatureStore
from bimer.overfit_smoke import (
    build_overfit_example,
    run_unimodal_overfit_smoke,
    write_overfit_smoke,
)
from bimer.schema import UtteranceRecord


def _records() -> list[UtteranceRecord]:
    labels = ("neutral", "joy", "neutral", "joy")
    return [
        UtteranceRecord(
            dataset="emotiontalk",
            split="train",
            dialogue_id="dialogue-1",
            utterance_id=index,
            text=f"line {index}",
            emotion=label,
            language="zh",
            start_seconds=float(index),
            end_seconds=float(index + 1),
        )
        for index, label in enumerate(labels)
    ]


def _store(tmp_path, records) -> FeatureStore:
    store = FeatureStore(tmp_path / "features")
    store.write(
        "emotiontalk",
        "train",
        0,
        FeatureShard(
            sample_ids=np.asarray([record.sample_id for record in records]),
            text=np.eye(4, dtype=np.float32),
            audio=np.concatenate(
                [np.eye(4, dtype=np.float32), np.ones((4, 2), np.float32)], axis=1
            ),
            vision=np.concatenate(
                [np.eye(4, dtype=np.float32), np.ones((4, 1), np.float32)], axis=1
            ),
            modality_mask=np.asarray(
                [
                    [True, True, True],
                    [True, True, False],
                    [True, True, True],
                    [True, True, True],
                ],
                dtype=np.bool_,
            ),
        ),
    )
    return store


def test_overfit_example_uses_only_rows_where_modality_is_available(tmp_path):
    records = _records()
    store = _store(tmp_path, records)

    example = build_overfit_example(
        records,
        store,
        dataset="emotiontalk",
        split="train",
        modality="vision",
        sample_count=3,
    )

    assert example.sample_ids == (
        records[0].sample_id,
        records[2].sample_id,
        records[3].sample_id,
    )
    assert example.modality_mask[:, 2].all()


def test_unimodal_overfit_smoke_memorizes_a_tiny_text_batch(tmp_path):
    records = _records()
    store = _store(tmp_path, records)

    report = run_unimodal_overfit_smoke(
        records,
        store,
        dataset="emotiontalk",
        split="train",
        modalities=("text",),
        sample_count=4,
        max_epochs=100,
        learning_rate=0.05,
        target_accuracy=1.0,
        hidden_dim=16,
        seed=42,
        device=torch.device("cpu"),
    )

    text_report = report["modalities"]["text"]
    assert text_report["passed"] is True
    assert text_report["accuracy"] == 1.0
    assert text_report["final_loss"] < text_report["initial_loss"]
    assert len(text_report["sample_ids"]) == 4


def test_write_overfit_smoke_creates_utf8_json(tmp_path):
    output = write_overfit_smoke(
        {"dataset": "emotiontalk", "status": "通过"},
        tmp_path / "reports" / "overfit.json",
    )

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "通过"
