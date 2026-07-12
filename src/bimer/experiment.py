from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .experiment_data import build_dialogue_examples
from .feature_store import FeatureStore
from .labels import EMOTION_LABELS, emotion_index
from .losses import sqrt_inverse_class_weights
from .manifest import read_manifest
from .metrics import bootstrap_weighted_f1
from .model_factory import build_model
from .training import (
    BalancedDialogueSampler,
    FitConfig,
    collate_dialogues,
    evaluate_batches,
    fit_model,
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: str = "lagf"
    seed: int = 42
    batch_size: int = 8
    max_epochs: int = 50
    patience: int = 7
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    hidden_dim: int = 256
    dropout: float = 0.2
    modality_dropout: float = 0.2
    use_language_embedding: bool = True
    use_reliability_gates: bool = True
    use_context: bool = True
    bootstrap_iterations: int = 2000


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_name(dataset: str, role: str) -> str:
    if role == "validation":
        return "dev" if dataset == "meld" else "validation"
    return role


def _loader(examples, *, batch_size: int, sampler=None):
    return DataLoader(
        examples,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_dialogues,
    )


def run_experiment(
    *,
    manifest_path: Path | str,
    feature_root: Path | str,
    output_directory: Path | str,
    config: ExperimentConfig,
    device_name: str = "auto",
) -> Path:
    set_reproducible_seed(config.seed)
    device = resolve_device(device_name)
    records = read_manifest(manifest_path)
    store = FeatureStore(feature_root)
    by_group = {
        (dataset, role): [
            record
            for record in records
            if record.dataset == dataset and str(record.split) == _split_name(dataset, role)
        ]
        for dataset in ("meld", "emotiontalk")
        for role in ("train", "validation", "test")
    }
    for key, group in by_group.items():
        if not group:
            raise ValueError(f"manifest has no records for {key[0]} {key[1]}")

    examples = {
        key: build_dialogue_examples(
            group,
            store.read_all(key[0], _split_name(key[0], key[1])),
        )
        for key, group in by_group.items()
    }
    train_examples = examples[("meld", "train")] + examples[("emotiontalk", "train")]
    sampler = BalancedDialogueSampler(train_examples, seed=config.seed)
    train_loader = _loader(train_examples, batch_size=config.batch_size, sampler=sampler)
    validation_loaders = {
        dataset: _loader(examples[(dataset, "validation")], batch_size=config.batch_size)
        for dataset in ("meld", "emotiontalk")
    }
    test_loaders = {
        dataset: _loader(examples[(dataset, "test")], batch_size=config.batch_size)
        for dataset in ("meld", "emotiontalk")
    }

    first = train_examples[0]
    unique_train_labels = torch.tensor(
        [emotion_index(str(record.emotion)) for record in records if str(record.split) == "train"],
        dtype=torch.long,
    )
    class_weights = sqrt_inverse_class_weights(unique_train_labels, num_classes=7)
    majority_class = int(torch.bincount(unique_train_labels, minlength=7).argmax())
    model_config = {
        "name": config.model,
        "text_dim": first.text.shape[1],
        "audio_dim": first.audio.shape[1],
        "vision_dim": first.vision.shape[1],
        "hidden_dim": config.hidden_dim,
        "num_classes": 7,
        "dropout": config.dropout,
        "modality_dropout": config.modality_dropout,
        "use_language_embedding": config.use_language_embedding,
        "use_reliability_gates": config.use_reliability_gates,
        "use_context": config.use_context,
        "majority_class": majority_class,
    }
    model = build_model(**model_config)
    output_root = Path(output_directory) / config.model / f"seed-{config.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "best.pt"

    history_payload: dict[str, object] | None = None
    if config.model != "majority":
        history = fit_model(
            model,
            train_batches=train_loader,
            validation_batches=validation_loaders,
            label_names=EMOTION_LABELS,
            checkpoint_path=checkpoint_path,
            config=FitConfig(
                max_epochs=config.max_epochs,
                patience=config.patience,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
            device=device,
            class_weights=class_weights,
            checkpoint_metadata={"model_config": model_config, "experiment": asdict(config)},
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        history_payload = {
            "best_epoch": history.best_epoch,
            "best_score": history.best_score,
            "epochs": [asdict(epoch) for epoch in history.epochs],
        }
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "metadata": {"model_config": model_config, "experiment": asdict(config)},
            },
            checkpoint_path,
        )

    test_results: dict[str, object] = {}
    for dataset, loader in test_loaders.items():
        report = evaluate_batches(model, loader, device=device, label_names=EMOTION_LABELS)
        confidence_interval = bootstrap_weighted_f1(
            report.truth,
            report.prediction,
            iterations=config.bootstrap_iterations,
            seed=config.seed,
        )
        test_results[dataset] = {
            **report.metrics,
            "weighted_f1_ci95": list(confidence_interval),
        }

    payload = {
        "config": asdict(config),
        "device": str(device),
        "history": history_payload,
        "test": test_results,
    }
    results_path = output_root / "results.json"
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return results_path


def aggregate_seed_results(paths: Sequence[Path | str], output_path: Path | str) -> Path:
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    summary: dict[str, object] = {"runs": len(payloads), "datasets": {}}
    for dataset in ("meld", "emotiontalk"):
        dataset_summary: dict[str, dict[str, float]] = {}
        for metric in ("weighted_f1", "macro_f1", "accuracy"):
            values = np.asarray([payload["test"][dataset][metric] for payload in payloads])
            dataset_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        summary["datasets"][dataset] = dataset_summary  # type: ignore[index]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _without_modality(examples, modality: str):
    index = {"text": 0, "audio": 1, "vision": 2}[modality]
    updated = []
    for example in examples:
        mask = example.modality_mask.copy()
        mask[:, index] = False
        feature_name = modality
        updated.append(
            replace(
                example,
                **{
                    feature_name: np.zeros_like(getattr(example, feature_name)),
                    "modality_mask": mask,
                },
            )
        )
    return updated


def evaluate_checkpoint(
    *,
    manifest_path: Path | str,
    feature_root: Path | str,
    checkpoint_path: Path | str,
    output_path: Path | str,
    missing_modality: str | None = None,
    bootstrap_iterations: int = 2000,
    device_name: str = "auto",
) -> Path:
    if missing_modality not in {None, "text", "audio", "vision"}:
        raise ValueError("missing_modality must be text, audio, vision, or None")
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("metadata", {}).get("model_config")
    if not model_config:
        raise ValueError("checkpoint does not contain model_config metadata")
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    records = read_manifest(manifest_path)
    store = FeatureStore(feature_root)
    results: dict[str, object] = {}
    for dataset in ("meld", "emotiontalk"):
        group = [
            record
            for record in records
            if record.dataset == dataset and str(record.split) == "test"
        ]
        examples = build_dialogue_examples(group, store.read_all(dataset, "test"))
        if missing_modality:
            examples = _without_modality(examples, missing_modality)
        loader = _loader(examples, batch_size=8)
        report = evaluate_batches(model, loader, device=device, label_names=EMOTION_LABELS)
        results[dataset] = {
            **report.metrics,
            "weighted_f1_ci95": list(
                bootstrap_weighted_f1(
                    report.truth,
                    report.prediction,
                    iterations=bootstrap_iterations,
                    seed=42,
                )
            ),
        }
    payload = {
        "condition": f"missing_{missing_modality}" if missing_modality else "standard",
        "checkpoint": str(checkpoint_path),
        "test": results,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
