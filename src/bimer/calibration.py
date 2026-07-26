from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _validated_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("each probability row must have positive mass")
    return values / totals


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    values = _validated_probabilities(probabilities)
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def calibration_metrics(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float]:
    values = _validated_probabilities(probabilities)
    labels = np.asarray(truth, dtype=np.int64)
    if labels.shape != (len(values),):
        raise ValueError("truth must have shape [samples]")
    if (labels < 0).any() or (labels >= values.shape[1]).any():
        raise ValueError("truth contains an out-of-range class")
    if bins <= 0:
        raise ValueError("bins must be positive")
    prediction = values.argmax(axis=1)
    confidence = values.max(axis=1)
    correct = prediction == labels
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        active = (
            (confidence >= lower) & (confidence < upper)
            if index < bins - 1
            else (confidence >= lower) & (confidence <= upper)
        )
        if active.any():
            ece += float(active.mean()) * abs(
                float(correct[active].mean()) - float(confidence[active].mean())
            )
    one_hot = np.eye(values.shape[1], dtype=np.float64)[labels]
    brier = float(np.square(values - one_hot).sum(axis=1).mean())
    nll = float(-np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())
    return {
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "accuracy": float(correct.mean()),
    }


def fit_temperature(probabilities: np.ndarray, truth: np.ndarray) -> float:
    values = _validated_probabilities(probabilities)
    labels = np.asarray(truth, dtype=np.int64)
    if len(values) != len(labels):
        raise ValueError("probabilities and truth must have the same length")
    logits = torch.tensor(
        np.log(np.clip(values, 1e-12, 1.0)),
        dtype=torch.float64,
    )
    targets = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.2,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0))


def select_uncertainty_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
) -> float:
    values = _validated_probabilities(probabilities)
    labels = np.asarray(truth, dtype=np.int64)
    if labels.shape != (len(values),):
        raise ValueError("truth must have shape [samples]")
    predictions = values.argmax(axis=1)
    confidence = values.max(axis=1)
    overall_accuracy = float((predictions == labels).mean())
    candidates: list[tuple[float, float]] = []
    for threshold in np.arange(0.35, 0.8001, 0.05):
        active = confidence >= threshold
        if float(active.mean()) < 0.70:
            continue
        accuracy = float((predictions[active] == labels[active]).mean())
        candidates.append((round(float(threshold), 2), accuracy))
    if not candidates:
        return 0.50
    best_accuracy = max(accuracy for _, accuracy in candidates)
    if best_accuracy < overall_accuracy + 0.03:
        return 0.50
    return min(
        threshold
        for threshold, accuracy in candidates
        if abs(accuracy - best_accuracy) <= 1e-12
    )


@dataclass(frozen=True, slots=True)
class LanguageCalibration:
    temperature: float
    threshold: float
    enabled: bool
    before: dict[str, float]
    after: dict[str, float]


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    languages: dict[str, LanguageCalibration]
    bins: int = 15

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            languages={
                language: LanguageCalibration(**calibration)
                for language, calibration in payload["languages"].items()
            },
            bins=int(payload.get("bins", 15)),
        )


def fit_calibration_profile(
    probabilities: np.ndarray,
    truth: np.ndarray,
    languages: Sequence[str] | np.ndarray,
    *,
    bins: int = 15,
) -> CalibrationProfile:
    values = _validated_probabilities(probabilities)
    labels = np.asarray(truth, dtype=np.int64)
    language_values = np.asarray(languages, dtype=str)
    if labels.shape != (len(values),) or language_values.shape != (len(values),):
        raise ValueError("truth and languages must align with probabilities")
    results: dict[str, LanguageCalibration] = {}
    for language in sorted(set(language_values.tolist())):
        active = language_values == language
        before = calibration_metrics(values[active], labels[active], bins=bins)
        candidate_temperature = fit_temperature(values[active], labels[active])
        candidate_probabilities = apply_temperature(
            values[active],
            candidate_temperature,
        )
        candidate_after = calibration_metrics(
            candidate_probabilities,
            labels[active],
            bins=bins,
        )
        relative_ece_reduction = (
            (before["ece"] - candidate_after["ece"]) / before["ece"]
            if before["ece"] > 0
            else 0.0
        )
        enabled = (
            relative_ece_reduction >= 0.10
            and candidate_after["nll"] <= before["nll"] + 1e-9
        )
        temperature = candidate_temperature if enabled else 1.0
        calibrated = (
            candidate_probabilities if enabled else values[active]
        )
        after = (
            candidate_after
            if enabled
            else dict(before)
        )
        results[language] = LanguageCalibration(
            temperature=float(temperature),
            threshold=select_uncertainty_threshold(calibrated, labels[active]),
            enabled=enabled,
            before=before,
            after=after,
        )
    return CalibrationProfile(languages=results, bins=bins)
