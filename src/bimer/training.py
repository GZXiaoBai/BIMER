from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Sampler

from .losses import masked_weighted_cross_entropy
from .metrics import classification_metrics


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

    def __post_init__(self) -> None:
        rows = len(self.sample_ids)
        if not all(
            array.shape[0] == rows
            for array in (self.text, self.audio, self.vision, self.modality_mask, self.labels)
        ):
            raise ValueError("dialogue feature arrays must share a row count")
        if self.modality_mask.shape != (rows, 3):
            raise ValueError("modality_mask must have shape [utterances, 3]")
        if self.language_id not in {0, 1}:
            raise ValueError("language_id must be 0 for English or 1 for Chinese")


@dataclass(slots=True)
class MultimodalBatch:
    text_features: Tensor
    audio_features: Tensor
    vision_features: Tensor
    modality_mask: Tensor
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
            "attention_mask": self.attention_mask,
            "language_ids": self.language_ids,
        }

    def to(self, device: torch.device) -> "MultimodalBatch":
        return MultimodalBatch(
            text_features=self.text_features.to(device),
            audio_features=self.audio_features.to(device),
            vision_features=self.vision_features.to(device),
            modality_mask=self.modality_mask.to(device),
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
        attention_mask[index, :length] = True
        labels[index, :length] = torch.from_numpy(example.labels)
        languages[index] = example.language_id
        sample_ids.append(example.sample_ids)

    return MultimodalBatch(
        text_features=text,
        audio_features=audio,
        vision_features=vision,
        modality_mask=modality_mask,
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

    def __iter__(self) -> Iterator[int]:
        groups: dict[str, list[int]] = {"meld": [], "emotiontalk": []}
        for index, example in enumerate(self.examples):
            groups.setdefault(example.dataset, []).append(index)
        if not groups["meld"] or not groups["emotiontalk"]:
            indices = list(range(len(self.examples)))
            random.Random(self.seed).shuffle(indices)
            return iter(indices)

        generator = random.Random(self.seed)
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
) -> float:
    model.train()
    losses: list[float] = []
    if class_weights is not None:
        class_weights = class_weights.to(device)
    for raw_batch in batches:
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch.model_inputs())
        loss = masked_weighted_cross_entropy(
            output.logits,
            batch.labels,
            batch.attention_mask,
            class_weights=class_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("training batches are empty")
    return float(np.mean(losses))


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    metrics: dict[str, object]
    truth: np.ndarray
    prediction: np.ndarray
    gates: np.ndarray


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
    gates: list[np.ndarray] = []
    for raw_batch in batches:
        batch = raw_batch.to(device)
        output = model(**batch.model_inputs())
        active = batch.attention_mask.bool()
        truth.append(batch.labels[active].cpu().numpy())
        prediction.append(output.logits.argmax(dim=-1)[active].cpu().numpy())
        gates.append(output.gates[active].cpu().numpy())
    if not truth:
        raise ValueError("evaluation batches are empty")
    all_truth = np.concatenate(truth)
    all_prediction = np.concatenate(prediction)
    all_gates = np.concatenate(gates)
    return EvaluationReport(
        metrics=classification_metrics(
            all_truth,
            all_prediction,
            label_names=label_names,
        ),
        truth=all_truth,
        prediction=all_prediction,
        gates=all_gates,
    )


def validation_selection_score(reports: dict[str, dict[str, object]]) -> float:
    required = ("meld", "emotiontalk")
    try:
        scores = [float(reports[dataset]["weighted_f1"]) for dataset in required]
    except KeyError as exc:
        raise ValueError("validation reports must include meld and emotiontalk") from exc
    return sum(scores) / len(scores)


@dataclass(frozen=True, slots=True)
class FitConfig:
    max_epochs: int = 50
    patience: int = 7
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2


@dataclass(frozen=True, slots=True)
class EpochSummary:
    epoch: int
    train_loss: float
    selection_score: float
    validation: dict[str, dict[str, object]]


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
) -> FitHistory:
    if set(validation_batches) != {"meld", "emotiontalk"}:
        raise ValueError("validation_batches must contain meld and emotiontalk")
    train_materialized = list(train_batches)
    validation_materialized = {
        dataset: list(batches) for dataset, batches in validation_batches.items()
    }
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
        train_loss = train_epoch(
            model,
            train_materialized,
            optimizer,
            device=device,
            class_weights=class_weights,
        )
        validation: dict[str, dict[str, object]] = {}
        for dataset, batches in validation_materialized.items():
            validation[dataset] = evaluate_batches(
                model,
                batches,
                device=device,
                label_names=label_names,
            ).metrics
        score = validation_selection_score(validation)
        summaries.append(
            EpochSummary(
                epoch=epoch,
                train_loss=train_loss,
                selection_score=score,
                validation=validation,
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
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    return FitHistory(
        best_epoch=best_epoch,
        best_score=best_score,
        epochs=tuple(summaries),
    )
