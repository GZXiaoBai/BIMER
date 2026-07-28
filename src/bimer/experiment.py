from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .batching import DialogueExample, MultimodalBatch
from .experiment_data import build_dialogue_examples
from .experiment_protocol import ExperimentProtocolRunner, ProtocolSpec
from .feature_store import FeatureShard, FeatureStore
from .labels import EMOTION_LABELS, emotion_index
from .losses import sqrt_inverse_class_weights
from .manifest import read_manifest
from .metrics import cluster_bootstrap_weighted_f1
from .model_factory import build_model
from .normalization import NormalizedModel, compute_input_statistics
from .paired_training import (
    BalancedCorruptionPairSampler,
    CorruptionPair,
    ModalityName,
    PairedMultimodalBatch,
    build_corruption_pairs,
    collate_corruption_pairs,
)
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
    min_epochs: int = 15
    patience: int = 7
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    hidden_dim: int = 256
    dropout: float = 0.2
    modality_dropout: float = 0.2
    use_language_embedding: bool = True
    use_reliability_gates: bool = True
    use_context: bool = True
    use_quality_input: bool = True
    bootstrap_iterations: int = 2000
    training_scope: str = "joint"
    use_input_normalization: bool = True
    evaluate_test: bool = True
    augmentation_manifests: tuple[str, ...] = ()
    augmentation_feature_roots: tuple[str, ...] = ()
    classification_loss: str = "weighted_ce"
    focal_gamma: float = 2.0
    augmentation_modalities: tuple[str, ...] = ()
    augmentation_severities: tuple[float, ...] = ()
    corrupted_classification_weight: float = 0.5
    gate_ranking_weight: float = 0.0
    gate_ranking_margin: float = 0.10
    prototype_loss_weight: float = 0.0
    prototype_temperature: float = 0.07
    use_adaptive_context_gate: bool = True
    context_gate_override: float | None = None
    protocol_stage: str = "standard"


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


def _loader(
    examples: Sequence[DialogueExample],
    *,
    batch_size: int,
    sampler: Sampler[int] | None = None,
) -> DataLoader[MultimodalBatch]:
    return cast(
        DataLoader[MultimodalBatch],
        DataLoader(
            cast(Any, examples),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_dialogues,
        ),
    )


def run_experiment(
    *,
    manifest_path: Path | str,
    feature_root: Path | str,
    output_directory: Path | str,
    config: ExperimentConfig,
    device_name: str = "auto",
) -> Path:
    spec = ProtocolSpec.from_config(config)
    output = Path(output_directory)
    result_path = (
        output / config.model / config.training_scope / f"seed-{config.seed}" / "results.json"
    )
    status_path = (
        output
        / "_protocol"
        / f"{spec.stage}-{config.model}-{config.training_scope}-seed-{config.seed}.json"
    )
    runner = ExperimentProtocolRunner(
        spec,
        status_path=status_path,
        result_path=result_path,
    )
    result = runner.run(
        lambda: _run_experiment_impl(
            manifest_path=manifest_path,
            feature_root=feature_root,
            output_directory=output_directory,
            config=config,
            device_name=device_name,
        )
    )
    return Path(result)


def _run_experiment_impl(
    *,
    manifest_path: Path | str,
    feature_root: Path | str,
    output_directory: Path | str,
    config: ExperimentConfig,
    device_name: str = "auto",
) -> Path:
    set_reproducible_seed(config.seed)
    if config.training_scope not in {"joint", "meld", "emotiontalk"}:
        raise ValueError("training_scope must be joint, meld, or emotiontalk")
    if len(config.augmentation_manifests) != len(config.augmentation_feature_roots):
        raise ValueError(
            "augmentation_manifests and augmentation_feature_roots must have equal length"
        )
    paired_requested = bool(config.augmentation_modalities)
    if config.classification_loss not in {
        "weighted_ce",
        "balanced_softmax",
        "focal",
    }:
        raise ValueError("unsupported classification_loss")
    if config.focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    if (
        config.corrupted_classification_weight < 0
        or config.gate_ranking_weight < 0
        or config.gate_ranking_margin < 0
        or config.prototype_loss_weight < 0
    ):
        raise ValueError("training objective weights and margin must be non-negative")
    if config.prototype_temperature <= 0:
        raise ValueError("prototype_temperature must be positive")
    if config.gate_ranking_weight > 0 and not paired_requested:
        raise ValueError("gate ranking requires paired augmentations")
    if paired_requested and len(config.augmentation_modalities) != len(
        config.augmentation_manifests
    ):
        raise ValueError("augmentation_modalities must identify every paired augmentation")
    if config.augmentation_severities and len(config.augmentation_severities) != len(
        config.augmentation_manifests
    ):
        raise ValueError("augmentation_severities must identify every paired augmentation")
    invalid_modalities = set(config.augmentation_modalities) - {
        "text",
        "audio",
        "vision",
    }
    if invalid_modalities:
        raise ValueError("augmentation modalities must be text, audio, or vision")
    device = resolve_device(device_name)
    records = read_manifest(manifest_path)
    store = FeatureStore(feature_root)
    training_datasets = (
        ("meld", "emotiontalk") if config.training_scope == "joint" else (config.training_scope,)
    )
    available_datasets = tuple(
        dataset
        for dataset in ("meld", "emotiontalk")
        if any(record.dataset == dataset for record in records)
    )
    missing_training_datasets = set(training_datasets) - set(available_datasets)
    if missing_training_datasets:
        missing = ", ".join(sorted(missing_training_datasets))
        raise ValueError(f"manifest has no records for training datasets: {missing}")
    selection_datasets = training_datasets
    evaluation_datasets = available_datasets if config.evaluate_test else ()

    by_group = {
        (dataset, role): [
            record
            for record in records
            if record.dataset == dataset and str(record.split) == _split_name(dataset, role)
        ]
        for dataset in training_datasets
        for role in ("train", "validation")
    }
    by_group.update(
        {
            (dataset, "test"): [
                record
                for record in records
                if record.dataset == dataset and str(record.split) == "test"
            ]
            for dataset in evaluation_datasets
        }
    )
    for key, group in by_group.items():
        if not group:
            raise ValueError(f"manifest has no records for {key[0]} {key[1]}")

    examples = {
        key: build_dialogue_examples(
            group,
            store.read_all(key[0], _split_name(key[0], key[1])),
            overlap=8 if key[1] == "train" else 0,
        )
        for key, group in by_group.items()
    }
    clean_train_examples = [
        example for dataset in training_datasets for example in examples[(dataset, "train")]
    ]
    train_examples = list(clean_train_examples)
    augmentation_shards: list[FeatureShard] = []
    augmentation_summaries: list[dict[str, object]] = []
    corruption_pairs: list[CorruptionPair] = []
    for augmentation_index, (manifest_name, feature_name) in enumerate(
        zip(
            config.augmentation_manifests,
            config.augmentation_feature_roots,
            strict=True,
        )
    ):
        augmentation_records = read_manifest(manifest_name)
        if any(str(record.split) != "train" for record in augmentation_records):
            raise ValueError("augmentation manifests may only contain training records")
        unexpected = {
            record.dataset
            for record in augmentation_records
            if record.dataset not in training_datasets
        }
        if unexpected:
            raise ValueError(
                "augmentation manifest contains datasets outside training scope: "
                + ", ".join(sorted(unexpected))
            )
        augmentation_store = FeatureStore(feature_name)
        view_examples = []
        for dataset in training_datasets:
            group = [record for record in augmentation_records if record.dataset == dataset]
            if not group:
                continue
            shards = augmentation_store.read_all(dataset, "train")
            if not shards:
                raise ValueError(f"augmentation feature root has no {dataset} train shards")
            augmentation_shards.extend(shards)
            view_examples.extend(build_dialogue_examples(group, shards, overlap=8))
        if not view_examples:
            raise ValueError("augmentation manifest contains no usable training records")
        if paired_requested:
            modality = cast(
                ModalityName,
                config.augmentation_modalities[augmentation_index],
            )
            severity = (
                config.augmentation_severities[augmentation_index]
                if config.augmentation_severities
                else 1.0
            )
            corruption_pairs.extend(
                build_corruption_pairs(
                    clean_train_examples,
                    view_examples,
                    corrupted_modality=modality,
                    severity=severity,
                )
            )
        else:
            train_examples.extend(view_examples)
        augmentation_summaries.append(
            {
                "manifest": str(manifest_name),
                "features": str(feature_name),
                "examples": len(view_examples),
                "paired": paired_requested,
                "corrupted_modality": (
                    config.augmentation_modalities[augmentation_index] if paired_requested else None
                ),
            }
        )
    sampler = BalancedDialogueSampler(train_examples, seed=config.seed)
    train_loader = _loader(train_examples, batch_size=config.batch_size, sampler=sampler)
    paired_train_loader: DataLoader[PairedMultimodalBatch] | None = None
    if corruption_pairs:
        paired_sampler = BalancedCorruptionPairSampler(
            cast(Any, corruption_pairs),
            seed=config.seed,
        )
        paired_train_loader = cast(
            DataLoader[PairedMultimodalBatch],
            DataLoader(
                cast(Any, corruption_pairs),
                batch_size=config.batch_size,
                sampler=paired_sampler,
                shuffle=False,
                collate_fn=collate_corruption_pairs,
            ),
        )
    validation_loaders: dict[str, Iterable[MultimodalBatch]] = {
        dataset: _loader(examples[(dataset, "validation")], batch_size=config.batch_size)
        for dataset in selection_datasets
    }
    test_loaders = {
        dataset: _loader(examples[(dataset, "test")], batch_size=config.batch_size)
        for dataset in evaluation_datasets
    }

    first = train_examples[0]
    unique_train_labels = torch.tensor(
        [
            emotion_index(str(record.emotion))
            for record in records
            if str(record.split) == "train" and record.dataset in training_datasets
        ],
        dtype=torch.long,
    )
    class_weights = sqrt_inverse_class_weights(unique_train_labels, num_classes=7)
    class_counts = torch.bincount(unique_train_labels, minlength=7).to(torch.float32)
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
        "use_quality_input": config.use_quality_input,
        "use_adaptive_context_gate": config.use_adaptive_context_gate,
        "context_gate_override": config.context_gate_override,
        "prototype_temperature": config.prototype_temperature,
        "majority_class": majority_class,
        "use_input_normalization": config.use_input_normalization,
    }
    model = build_model(**model_config)
    if isinstance(model, NormalizedModel):
        normalization_shards = [
            shard for dataset in training_datasets for shard in store.read_all(dataset, "train")
        ] + augmentation_shards
        model.set_statistics(compute_input_statistics(normalization_shards))
    output_root = (
        Path(output_directory) / config.model / config.training_scope / f"seed-{config.seed}"
    )
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
                min_epochs=config.min_epochs,
                patience=config.patience,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                classification_loss=config.classification_loss,
                focal_gamma=config.focal_gamma,
                corrupted_classification_weight=config.corrupted_classification_weight,
                gate_ranking_weight=config.gate_ranking_weight,
                gate_ranking_margin=config.gate_ranking_margin,
                prototype_loss_weight=config.prototype_loss_weight,
            ),
            device=device,
            class_weights=(
                None if config.classification_loss == "balanced_softmax" else class_weights
            ),
            class_counts=class_counts,
            paired_train_batches=paired_train_loader,
            checkpoint_metadata={
                "model_config": model_config,
                "experiment": asdict(config),
                "training_distribution": {
                    "class_counts": class_counts.tolist(),
                    "class_weights": class_weights.tolist(),
                },
            },
            selection_datasets=selection_datasets,
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

    context_id_by_sample = {record.sample_id: record.effective_context_id for record in records}
    language_by_sample = {record.sample_id: str(record.language) for record in records}
    validation_results: dict[str, object] = {}
    validation_prediction_directory = output_root / "validation_predictions"
    validation_prediction_directory.mkdir(parents=True, exist_ok=True)
    for dataset, loader in validation_loaders.items():
        report = evaluate_batches(model, loader, device=device, label_names=EMOTION_LABELS)
        validation_results[dataset] = report.metrics
        validation_context_ids = np.asarray(
            [context_id_by_sample[sample_id] for sample_id in report.sample_ids],
            dtype=str,
        )
        validation_prediction_payload: dict[str, np.ndarray] = {
            "sample_ids": np.asarray(report.sample_ids, dtype=str),
            "context_ids": validation_context_ids,
            "languages": np.asarray(
                [language_by_sample[sample_id] for sample_id in report.sample_ids],
                dtype=str,
            ),
            "truth": report.truth.astype(np.int64),
            "prediction": report.prediction.astype(np.int64),
            "probabilities": report.probabilities.astype(np.float32),
            "gates": report.gates.astype(np.float32),
            "modality_quality": report.modality_quality.astype(np.float32),
            "modality_available": report.modality_available.astype(np.bool_),
            "context_lengths": report.context_lengths.astype(np.int64),
        }
        if report.context_gates is not None:
            validation_prediction_payload["context_gates"] = report.context_gates.astype(np.float32)
        if report.prototype_logits is not None:
            validation_prediction_payload["prototype_logits"] = report.prototype_logits.astype(
                np.float32
            )
        if report.representations is not None:
            validation_prediction_payload["representations"] = report.representations.astype(
                np.float32
            )
        if report.local_prediction is not None:
            validation_prediction_payload["local_prediction"] = report.local_prediction.astype(
                np.int64
            )
        if report.fixed_context_prediction is not None:
            validation_prediction_payload["fixed_context_prediction"] = (
                report.fixed_context_prediction.astype(np.int64)
            )
        cast(Any, np.savez_compressed)(
            validation_prediction_directory / f"{dataset}.npz",
            **validation_prediction_payload,
        )

    test_results: dict[str, object] = {}
    prediction_directory = output_root / "predictions"
    if test_loaders:
        prediction_directory.mkdir(parents=True, exist_ok=True)
    for dataset, loader in test_loaders.items():
        report = evaluate_batches(model, loader, device=device, label_names=EMOTION_LABELS)
        context_ids = np.asarray(
            [context_id_by_sample[sample_id] for sample_id in report.sample_ids],
            dtype=str,
        )
        confidence_interval = cluster_bootstrap_weighted_f1(
            report.truth,
            report.prediction,
            context_ids,
            iterations=config.bootstrap_iterations,
            seed=config.seed,
        )
        test_results[dataset] = {
            **report.metrics,
            "weighted_f1_ci95": list(confidence_interval),
            "bootstrap_unit": "context",
        }
        test_prediction_payload: dict[str, np.ndarray] = {
            "sample_ids": np.asarray(report.sample_ids, dtype=str),
            "context_ids": context_ids,
            "truth": report.truth.astype(np.int64),
            "prediction": report.prediction.astype(np.int64),
            "probabilities": report.probabilities.astype(np.float32),
            "gates": report.gates.astype(np.float32),
            "modality_quality": report.modality_quality.astype(np.float32),
            "modality_available": report.modality_available.astype(np.bool_),
            "context_lengths": report.context_lengths.astype(np.int64),
        }
        if report.context_gates is not None:
            test_prediction_payload["context_gates"] = report.context_gates.astype(np.float32)
        if report.prototype_logits is not None:
            test_prediction_payload["prototype_logits"] = report.prototype_logits.astype(np.float32)
        if report.representations is not None:
            test_prediction_payload["representations"] = report.representations.astype(np.float32)
        if report.local_prediction is not None:
            test_prediction_payload["local_prediction"] = report.local_prediction.astype(np.int64)
        if report.fixed_context_prediction is not None:
            test_prediction_payload["fixed_context_prediction"] = (
                report.fixed_context_prediction.astype(np.int64)
            )
        cast(Any, np.savez_compressed)(
            prediction_directory / f"{dataset}.npz",
            **test_prediction_payload,
        )

    payload = {
        "config": asdict(config),
        "device": str(device),
        "training_datasets": list(training_datasets),
        "selection_datasets": list(selection_datasets),
        "evaluation_datasets": list(evaluation_datasets),
        "training_views": {
            "clean_examples": len(clean_train_examples),
            "augmentations": augmentation_summaries,
            "total_examples": len(train_examples),
            "paired_examples": len(corruption_pairs),
        },
        "history": history_payload,
        "validation": validation_results,
        "test": test_results,
    }
    results_path = output_root / "results.json"
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return results_path


def aggregate_seed_results(paths: Sequence[Path | str], output_path: Path | str) -> Path:
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not payloads:
        raise ValueError("at least one result path is required")
    datasets = tuple(
        dataset for dataset in ("meld", "emotiontalk") if dataset in payloads[0]["test"]
    )
    if any(set(payload["test"]) != set(datasets) for payload in payloads):
        raise ValueError("all seed results must contain the same test datasets")
    summary: dict[str, object] = {"runs": len(payloads), "datasets": {}}
    for dataset in datasets:
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


def _without_modality(
    examples: Sequence[DialogueExample],
    modality: Literal["text", "audio", "vision"],
) -> list[DialogueExample]:
    index = {"text": 0, "audio": 1, "vision": 2}[modality]
    updated: list[DialogueExample] = []
    for example in examples:
        mask = example.modality_mask.copy()
        mask[:, index] = False
        quality = np.asarray(example.modality_quality).copy()
        quality[:, index] = 0.0
        feature_name = modality
        updated.append(
            replace(
                example,
                **{
                    feature_name: np.zeros_like(getattr(example, feature_name)),
                    "modality_mask": mask,
                    "modality_quality": quality,
                },
            )
        )
    return updated


def _normalize_missing_modalities(
    missing_modality: str | Sequence[str] | None,
) -> tuple[str, ...]:
    requested = (
        (missing_modality,) if isinstance(missing_modality, str) else tuple(missing_modality or ())
    )
    allowed = ("text", "audio", "vision")
    invalid = set(requested) - set(allowed)
    if invalid:
        raise ValueError("missing_modality must contain only text, audio, or vision")
    normalized = tuple(modality for modality in allowed if modality in requested)
    if len(normalized) == len(allowed):
        raise ValueError("at least one modality must remain")
    return normalized


def _without_modalities(
    examples: Sequence[DialogueExample],
    modalities: Sequence[str],
) -> list[DialogueExample]:
    updated = list(examples)
    for modality in modalities:
        updated = _without_modality(
            updated,
            cast(Literal["text", "audio", "vision"], modality),
        )
    return updated


def evaluate_checkpoint(
    *,
    manifest_path: Path | str,
    feature_root: Path | str,
    checkpoint_path: Path | str,
    output_path: Path | str,
    missing_modality: str | Sequence[str] | None = None,
    condition_name: str | None = None,
    bootstrap_iterations: int = 2000,
    device_name: str = "auto",
    evaluation_role: str = "test",
) -> Path:
    if evaluation_role not in {"validation", "test"}:
        raise ValueError("evaluation_role must be validation or test")
    missing_modalities = _normalize_missing_modalities(missing_modality)
    if condition_name and missing_modalities:
        raise ValueError("condition_name cannot be combined with missing_modality")
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
    path = Path(output_path)
    prediction_directory = path.parent / f"{path.stem}.predictions"
    prediction_directory.mkdir(parents=True, exist_ok=True)
    for dataset in ("meld", "emotiontalk"):
        split = _split_name(dataset, evaluation_role)
        group = [
            record for record in records if record.dataset == dataset and str(record.split) == split
        ]
        if not group:
            continue
        examples = build_dialogue_examples(
            group,
            store.read_all(dataset, split),
            overlap=0,
        )
        if missing_modalities:
            examples = _without_modalities(examples, missing_modalities)
        loader = _loader(examples, batch_size=8)
        report = evaluate_batches(model, loader, device=device, label_names=EMOTION_LABELS)
        context_id_by_sample = {record.sample_id: record.effective_context_id for record in group}
        context_ids = np.asarray(
            [context_id_by_sample[sample_id] for sample_id in report.sample_ids],
            dtype=str,
        )
        results[dataset] = {
            **report.metrics,
            "weighted_f1_ci95": list(
                cluster_bootstrap_weighted_f1(
                    report.truth,
                    report.prediction,
                    context_ids,
                    iterations=bootstrap_iterations,
                    seed=42,
                )
            ),
            "bootstrap_unit": "context",
        }
        prediction_payload: dict[str, np.ndarray] = {
            "sample_ids": np.asarray(report.sample_ids, dtype=str),
            "context_ids": context_ids,
            "truth": report.truth.astype(np.int64),
            "prediction": report.prediction.astype(np.int64),
            "probabilities": report.probabilities.astype(np.float32),
            "gates": report.gates.astype(np.float32),
            "modality_quality": report.modality_quality.astype(np.float32),
            "modality_available": report.modality_available.astype(np.bool_),
            "context_lengths": report.context_lengths.astype(np.int64),
        }
        if report.context_gates is not None:
            prediction_payload["context_gates"] = report.context_gates.astype(np.float32)
        if report.prototype_logits is not None:
            prediction_payload["prototype_logits"] = report.prototype_logits.astype(np.float32)
        if report.representations is not None:
            prediction_payload["representations"] = report.representations.astype(np.float32)
        if report.local_prediction is not None:
            prediction_payload["local_prediction"] = report.local_prediction.astype(np.int64)
        if report.fixed_context_prediction is not None:
            prediction_payload["fixed_context_prediction"] = report.fixed_context_prediction.astype(
                np.int64
            )
        cast(Any, np.savez_compressed)(
            prediction_directory / f"{dataset}.npz",
            **prediction_payload,
        )
    payload = {
        "condition": (
            condition_name
            or (f"missing_{'_'.join(missing_modalities)}" if missing_modalities else "standard")
        ),
        "missing_modalities": list(missing_modalities),
        "manifest": str(manifest_path),
        "feature_root": str(feature_root),
        "checkpoint": str(checkpoint_path),
        evaluation_role: results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
