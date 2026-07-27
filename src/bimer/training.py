from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Sampler

from .batching import DialogueExample, MultimodalBatch
from .batching import collate_dialogues as collate_dialogues
from .losses import PrototypeContrastiveLoss, masked_classification_loss
from .metrics import classification_metrics

if TYPE_CHECKING:
    from .paired_training import PairedMultimodalBatch


class BalancedDialogueSampler(Sampler[int]):
    """Alternate MELD and EmotionTalk, oversampling the smaller collection."""

    def __init__(self, examples: Sequence[DialogueExample], *, seed: int = 42) -> None:
        self.examples = examples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        groups: dict[str, list[int]] = {"meld": [], "emotiontalk": []}
        for index, example in enumerate(self.examples):
            groups.setdefault(example.dataset, []).append(index)
        if not groups["meld"] or not groups["emotiontalk"]:
            indices = list(range(len(self.examples)))
            random.Random(self.seed + self.epoch).shuffle(indices)
            return iter(indices)

        generator = random.Random(self.seed + self.epoch)
        for indices in groups.values():
            generator.shuffle(indices)
        target = max(len(groups["meld"]), len(groups["emotiontalk"]))
        balanced: list[int] = []
        for offset in range(target):
            balanced.append(groups["meld"][offset % len(groups["meld"])])
            balanced.append(groups["emotiontalk"][offset % len(groups["emotiontalk"])])
        return iter(balanced)

    def __len__(self) -> int:
        counts = {
            dataset: sum(example.dataset == dataset for example in self.examples)
            for dataset in ("meld", "emotiontalk")
        }
        if counts["meld"] and counts["emotiontalk"]:
            return 2 * max(counts.values())
        return len(self.examples)


def train_epoch(
    model: nn.Module,
    batches: Iterable[MultimodalBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    class_weights: Tensor | None = None,
    classification_loss: str = "weighted_ce",
    class_counts: Tensor | None = None,
    focal_gamma: float = 2.0,
    paired_batches: Iterable["PairedMultimodalBatch"] | None = None,
    corrupted_classification_weight: float = 0.5,
    gate_ranking_weight: float = 0.0,
    gate_ranking_margin: float = 0.10,
    prototype_loss_weight: float = 0.0,
) -> float:
    return _train_epoch_report(
        model,
        batches,
        optimizer,
        device=device,
        class_weights=class_weights,
        classification_loss=classification_loss,
        class_counts=class_counts,
        focal_gamma=focal_gamma,
        paired_batches=paired_batches,
        corrupted_classification_weight=corrupted_classification_weight,
        gate_ranking_weight=gate_ranking_weight,
        gate_ranking_margin=gate_ranking_margin,
        prototype_loss_weight=prototype_loss_weight,
    ).loss


@dataclass(frozen=True, slots=True)
class TrainEpochReport:
    loss: float
    mean_gradient_norm: float
    clean_classification_loss: float
    corrupted_classification_loss: float
    gate_ranking_loss: float
    prototype_loss: float


def _train_epoch_report(
    model: nn.Module,
    batches: Iterable[MultimodalBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    class_weights: Tensor | None = None,
    classification_loss: str = "weighted_ce",
    class_counts: Tensor | None = None,
    focal_gamma: float = 2.0,
    paired_batches: Iterable["PairedMultimodalBatch"] | None = None,
    corrupted_classification_weight: float = 0.5,
    gate_ranking_weight: float = 0.0,
    gate_ranking_margin: float = 0.10,
    prototype_loss_weight: float = 0.0,
) -> TrainEpochReport:
    from .paired_training import gate_ranking_loss

    if corrupted_classification_weight < 0 or gate_ranking_weight < 0 or prototype_loss_weight < 0:
        raise ValueError("training loss weights must be non-negative")
    model.train()
    losses: list[float] = []
    clean_losses: list[float] = []
    corrupted_losses: list[float] = []
    ranking_losses: list[float] = []
    prototype_losses: list[float] = []
    gradient_norms: list[float] = []
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if class_counts is not None:
        class_counts = class_counts.to(device)
    pair_iterator = iter(paired_batches) if paired_batches is not None else None

    def next_pair() -> "PairedMultimodalBatch | None":
        nonlocal pair_iterator
        if paired_batches is None:
            return None
        assert pair_iterator is not None
        try:
            return next(pair_iterator)
        except StopIteration:
            pair_iterator = iter(paired_batches)
            try:
                return next(pair_iterator)
            except StopIteration as exc:
                raise ValueError("paired training batches are empty") from exc

    for raw_batch in batches:
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch.model_inputs())
        loss = masked_classification_loss(
            output.logits,
            batch.labels,
            batch.attention_mask,
            loss_name=classification_loss,
            class_weights=class_weights,
            class_counts=class_counts,
            focal_gamma=focal_gamma,
        )
        clean_losses.append(float(loss.detach().cpu()))
        if prototype_loss_weight > 0:
            fusion_model = getattr(model, "model", model)
            prototypes = getattr(fusion_model, "prototypes", None)
            if output.representations is None or prototypes is None:
                raise ValueError(
                    "prototype_loss_weight requires model representations and prototypes"
                )
            prototype_loss = PrototypeContrastiveLoss(
                temperature=float(getattr(fusion_model, "prototype_temperature", 0.07))
            )(
                output.representations,
                prototypes,
                batch.labels,
                batch.attention_mask,
            )
            loss = loss + prototype_loss_weight * prototype_loss
            prototype_losses.append(float(prototype_loss.detach().cpu()))
        raw_pair = next_pair()
        if raw_pair is not None:
            pair = raw_pair.to(device)
            fusion_model = getattr(model, "model", model)
            previous_modality_dropout = getattr(
                fusion_model,
                "modality_dropout",
                None,
            )
            if previous_modality_dropout is not None:
                fusion_model.modality_dropout = 0.0
            try:
                clean_pair_output = model(**pair.clean.model_inputs())
                corrupted_output = model(**pair.corrupted.model_inputs())
            finally:
                if previous_modality_dropout is not None:
                    fusion_model.modality_dropout = previous_modality_dropout
            corrupted_loss = masked_classification_loss(
                corrupted_output.logits,
                pair.corrupted.labels,
                pair.corrupted.attention_mask,
                loss_name=classification_loss,
                class_weights=class_weights,
                class_counts=class_counts,
                focal_gamma=focal_gamma,
            )
            ranking_loss = gate_ranking_loss(
                clean_pair_output.gates,
                corrupted_output.gates,
                clean_modality_mask=pair.clean.modality_mask,
                corrupted_modality_mask=pair.corrupted.modality_mask,
                attention_mask=pair.clean.attention_mask & pair.corrupted.attention_mask,
                corrupted_modality=pair.corrupted_modality,
                margin=gate_ranking_margin,
            )
            loss = (
                loss
                + corrupted_classification_weight * corrupted_loss
                + gate_ranking_weight * ranking_loss
            )
            corrupted_losses.append(float(corrupted_loss.detach().cpu()))
            ranking_losses.append(float(ranking_loss.detach().cpu()))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))
    if not losses:
        raise ValueError("training batches are empty")
    return TrainEpochReport(
        loss=float(np.mean(losses)),
        mean_gradient_norm=float(np.mean(gradient_norms)),
        clean_classification_loss=float(np.mean(clean_losses)),
        corrupted_classification_loss=(
            float(np.mean(corrupted_losses)) if corrupted_losses else 0.0
        ),
        gate_ranking_loss=(float(np.mean(ranking_losses)) if ranking_losses else 0.0),
        prototype_loss=(float(np.mean(prototype_losses)) if prototype_losses else 0.0),
    )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    metrics: dict[str, object]
    truth: np.ndarray
    prediction: np.ndarray
    probabilities: np.ndarray
    gates: np.ndarray
    modality_quality: np.ndarray
    modality_available: np.ndarray
    sample_ids: tuple[str, ...]
    context_lengths: np.ndarray
    context_gates: np.ndarray | None
    prototype_logits: np.ndarray | None
    representations: np.ndarray | None
    local_prediction: np.ndarray | None
    fixed_context_prediction: np.ndarray | None


@torch.no_grad()
def evaluate_batches(
    model: nn.Module,
    batches: Iterable[MultimodalBatch],
    *,
    device: torch.device,
    label_names: Sequence[str],
) -> EvaluationReport:
    model.eval()
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    modality_quality: list[np.ndarray] = []
    modality_available: list[np.ndarray] = []
    context_lengths: list[int] = []
    context_gates: list[np.ndarray] = []
    prototype_logits: list[np.ndarray] = []
    representations: list[np.ndarray] = []
    local_prediction: list[np.ndarray] = []
    fixed_context_prediction: list[np.ndarray] = []
    sample_ids: list[str] = []
    for raw_batch in batches:
        batch = raw_batch.to(device)
        output = model(**batch.model_inputs())
        active = batch.attention_mask.bool()
        truth.append(batch.labels[active].cpu().numpy())
        active_logits = output.logits[active]
        prediction.append(active_logits.argmax(dim=-1).cpu().numpy())
        probabilities.append(torch.softmax(active_logits, dim=-1).cpu().numpy())
        gates.append(output.gates[active].cpu().numpy())
        modality_quality.append(batch.modality_quality[active].cpu().numpy())
        modality_available.append(batch.modality_mask[active].cpu().numpy())
        for row_length in batch.attention_mask.sum(dim=1).cpu().tolist():
            context_lengths.extend([int(row_length)] * int(row_length))
        if output.context_gates is not None:
            context_gates.append(output.context_gates[active].cpu().numpy())
        if output.prototype_logits is not None:
            prototype_logits.append(output.prototype_logits[active].cpu().numpy())
        if output.representations is not None:
            representations.append(output.representations[active].cpu().numpy())
        if output.local_logits is not None:
            local_prediction.append(output.local_logits[active].argmax(dim=-1).cpu().numpy())
        if output.fixed_context_logits is not None:
            fixed_context_prediction.append(
                output.fixed_context_logits[active].argmax(dim=-1).cpu().numpy()
            )
        sample_ids.extend(sample_id for row in batch.sample_ids for sample_id in row)
    if not truth:
        raise ValueError("evaluation batches are empty")
    all_truth = np.concatenate(truth)
    all_prediction = np.concatenate(prediction)
    all_probabilities = np.concatenate(probabilities)
    all_gates = np.concatenate(gates)
    return EvaluationReport(
        metrics=classification_metrics(
            all_truth,
            all_prediction,
            label_names=label_names,
        ),
        truth=all_truth,
        prediction=all_prediction,
        probabilities=all_probabilities,
        gates=all_gates,
        modality_quality=np.concatenate(modality_quality),
        modality_available=np.concatenate(modality_available),
        sample_ids=tuple(sample_ids),
        context_lengths=np.asarray(context_lengths, dtype=np.int64),
        context_gates=np.concatenate(context_gates) if context_gates else None,
        prototype_logits=np.concatenate(prototype_logits) if prototype_logits else None,
        representations=np.concatenate(representations) if representations else None,
        local_prediction=np.concatenate(local_prediction) if local_prediction else None,
        fixed_context_prediction=(
            np.concatenate(fixed_context_prediction) if fixed_context_prediction else None
        ),
    )


def validation_selection_score(
    reports: dict[str, dict[str, object]],
    *,
    datasets: Sequence[str] = ("meld", "emotiontalk"),
) -> float:
    try:
        scores = [float(reports[dataset]["weighted_f1"]) for dataset in datasets]
    except KeyError as exc:
        raise ValueError("validation reports do not include every selection dataset") from exc
    if not scores:
        raise ValueError("at least one selection dataset is required")
    return sum(scores) / len(scores)


@dataclass(frozen=True, slots=True)
class FitConfig:
    max_epochs: int = 50
    min_epochs: int = 15
    patience: int = 7
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    classification_loss: str = "weighted_ce"
    focal_gamma: float = 2.0
    corrupted_classification_weight: float = 0.5
    gate_ranking_weight: float = 0.0
    gate_ranking_margin: float = 0.10
    prototype_loss_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class EpochSummary:
    epoch: int
    train_loss: float
    selection_score: float
    validation: dict[str, dict[str, object]]
    gradient_norm: float
    prediction_histograms: dict[str, tuple[int, ...]]
    gate_means: dict[str, tuple[float, ...]]
    collapse_flags: dict[str, bool]
    clean_classification_loss: float
    corrupted_classification_loss: float
    gate_ranking_loss: float
    prototype_loss: float


@dataclass(frozen=True, slots=True)
class FitHistory:
    best_epoch: int
    best_score: float
    epochs: tuple[EpochSummary, ...]


def fit_model(
    model: nn.Module,
    *,
    train_batches: Iterable[MultimodalBatch],
    validation_batches: dict[str, Iterable[MultimodalBatch]],
    label_names: Sequence[str],
    checkpoint_path: Path | str,
    config: FitConfig,
    device: torch.device,
    class_weights: Tensor | None = None,
    class_counts: Tensor | None = None,
    paired_train_batches: Iterable["PairedMultimodalBatch"] | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    selection_datasets: Sequence[str] = ("meld", "emotiontalk"),
) -> FitHistory:
    missing_validation = set(selection_datasets) - set(validation_batches)
    if missing_validation:
        missing = ", ".join(sorted(missing_validation))
        raise ValueError(f"validation_batches are missing selection datasets: {missing}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.to(device)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    summaries: list[EpochSummary] = []

    for epoch in range(1, config.max_epochs + 1):
        sampler = getattr(train_batches, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch - 1)
        paired_sampler = getattr(paired_train_batches, "sampler", None)
        if paired_sampler is not None and hasattr(paired_sampler, "set_epoch"):
            paired_sampler.set_epoch(epoch - 1)
        train_report = _train_epoch_report(
            model,
            train_batches,
            optimizer,
            device=device,
            class_weights=class_weights,
            classification_loss=config.classification_loss,
            class_counts=class_counts,
            focal_gamma=config.focal_gamma,
            paired_batches=paired_train_batches,
            corrupted_classification_weight=config.corrupted_classification_weight,
            gate_ranking_weight=config.gate_ranking_weight,
            gate_ranking_margin=config.gate_ranking_margin,
            prototype_loss_weight=config.prototype_loss_weight,
        )
        validation: dict[str, dict[str, object]] = {}
        prediction_histograms: dict[str, tuple[int, ...]] = {}
        gate_means: dict[str, tuple[float, ...]] = {}
        collapse_flags: dict[str, bool] = {}
        for dataset, batches in validation_batches.items():
            report = evaluate_batches(
                model,
                batches,
                device=device,
                label_names=label_names,
            )
            validation[dataset] = report.metrics
            histogram = np.bincount(report.prediction, minlength=len(label_names))
            prediction_histograms[dataset] = tuple(int(value) for value in histogram)
            gate_means[dataset] = tuple(float(value) for value in report.gates.mean(axis=0))
            collapse_flags[dataset] = bool(np.count_nonzero(histogram) < 2)
        score = validation_selection_score(validation, datasets=selection_datasets)
        summaries.append(
            EpochSummary(
                epoch=epoch,
                train_loss=train_report.loss,
                selection_score=score,
                validation=validation,
                gradient_norm=train_report.mean_gradient_norm,
                prediction_histograms=prediction_histograms,
                gate_means=gate_means,
                collapse_flags=collapse_flags,
                clean_classification_loss=train_report.clean_classification_loss,
                corrupted_classification_loss=(train_report.corrupted_classification_loss),
                gate_ranking_loss=train_report.gate_ranking_loss,
                prototype_loss=train_report.prototype_loss,
            )
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": score,
                    "fit_config": asdict(config),
                    "label_names": tuple(label_names),
                    "metadata": checkpoint_metadata or {},
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if epoch >= config.min_epochs and stale_epochs >= config.patience:
                break

    return FitHistory(
        best_epoch=best_epoch,
        best_score=best_score,
        epochs=tuple(summaries),
    )
