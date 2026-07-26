from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Sampler

from .losses import masked_classification_loss
from .metrics import classification_metrics

if TYPE_CHECKING:
    from .paired_training import PairedMultimodalBatch


@dataclass(frozen=True, slots=True)
class DialogueExample:
    dataset: str
    sample_ids: tuple[str, ...]
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    modality_mask: np.ndarray
    labels: np.ndarray
    language_id: int
    modality_quality: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.sample_ids)
        if not all(
            array.shape[0] == rows
            for array in (self.text, self.audio, self.vision, self.modality_mask, self.labels)
        ):
            raise ValueError("dialogue feature arrays must share a row count")
        if self.modality_mask.shape != (rows, 3):
            raise ValueError("modality_mask must have shape [utterances, 3]")
        if self.modality_quality is None:
            object.__setattr__(
                self,
                "modality_quality",
                np.repeat(self.modality_mask.astype(np.float32)[..., None], 4, axis=-1),
            )
        if np.asarray(self.modality_quality).shape != (rows, 3, 4):
            raise ValueError("modality_quality must have shape [utterances, 3, 4]")
        if self.language_id not in {0, 1}:
            raise ValueError("language_id must be 0 for English or 1 for Chinese")


@dataclass(slots=True)
class MultimodalBatch:
    text_features: Tensor
    audio_features: Tensor
    vision_features: Tensor
    modality_mask: Tensor
    modality_quality: Tensor
    attention_mask: Tensor
    language_ids: Tensor
    labels: Tensor
    sample_ids: tuple[tuple[str, ...], ...]

    def model_inputs(self) -> dict[str, Tensor]:
        return {
            "text_features": self.text_features,
            "audio_features": self.audio_features,
            "vision_features": self.vision_features,
            "modality_mask": self.modality_mask,
            "modality_quality": self.modality_quality,
            "attention_mask": self.attention_mask,
            "language_ids": self.language_ids,
        }

    def to(self, device: torch.device) -> "MultimodalBatch":
        return MultimodalBatch(
            text_features=self.text_features.to(device),
            audio_features=self.audio_features.to(device),
            vision_features=self.vision_features.to(device),
            modality_mask=self.modality_mask.to(device),
            modality_quality=self.modality_quality.to(device),
            attention_mask=self.attention_mask.to(device),
            language_ids=self.language_ids.to(device),
            labels=self.labels.to(device),
            sample_ids=self.sample_ids,
        )


def collate_dialogues(examples: Sequence[DialogueExample]) -> MultimodalBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    max_length = max(len(example.sample_ids) for example in examples)

    def zeros(dimension: int) -> Tensor:
        return torch.zeros(batch_size, max_length, dimension, dtype=torch.float32)

    text = zeros(examples[0].text.shape[1])
    audio = zeros(examples[0].audio.shape[1])
    vision = zeros(examples[0].vision.shape[1])
    modality_mask = torch.zeros(batch_size, max_length, 3, dtype=torch.bool)
    modality_quality = torch.zeros(batch_size, max_length, 3, 4, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    languages = torch.empty(batch_size, dtype=torch.long)
    sample_ids: list[tuple[str, ...]] = []

    for index, example in enumerate(examples):
        length = len(example.sample_ids)
        text[index, :length] = torch.from_numpy(example.text)
        audio[index, :length] = torch.from_numpy(example.audio)
        vision[index, :length] = torch.from_numpy(example.vision)
        modality_mask[index, :length] = torch.from_numpy(example.modality_mask)
        modality_quality[index, :length] = torch.from_numpy(
            np.asarray(example.modality_quality, dtype=np.float32)
        )
        attention_mask[index, :length] = True
        labels[index, :length] = torch.from_numpy(example.labels)
        languages[index] = example.language_id
        sample_ids.append(example.sample_ids)

    return MultimodalBatch(
        text_features=text,
        audio_features=audio,
        vision_features=vision,
        modality_mask=modality_mask,
        modality_quality=modality_quality,
        attention_mask=attention_mask,
        language_ids=languages,
        labels=labels,
        sample_ids=tuple(sample_ids),
    )


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
    ).loss


@dataclass(frozen=True, slots=True)
class TrainEpochReport:
    loss: float
    mean_gradient_norm: float
    clean_classification_loss: float
    corrupted_classification_loss: float
    gate_ranking_loss: float


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
) -> TrainEpochReport:
    from .paired_training import gate_ranking_loss

    if corrupted_classification_weight < 0 or gate_ranking_weight < 0:
        raise ValueError("paired loss weights must be non-negative")
    model.train()
    losses: list[float] = []
    clean_losses: list[float] = []
    corrupted_losses: list[float] = []
    ranking_losses: list[float] = []
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
                attention_mask=pair.clean.attention_mask
                & pair.corrupted.attention_mask,
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
        gate_ranking_loss=(
            float(np.mean(ranking_losses)) if ranking_losses else 0.0
        ),
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
                corrupted_classification_loss=(
                    train_report.corrupted_classification_loss
                ),
                gate_ranking_loss=train_report.gate_ranking_loss,
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
