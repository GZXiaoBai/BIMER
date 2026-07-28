from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .labels import EMOTION_LABELS, EmotionLabel, normalize_emotion

DatasetSplit = Literal["train", "validation", "val", "dev", "test"]
LanguageCode = Literal["zh", "en"]


@dataclass(frozen=True, slots=True)
class UtteranceRecord:
    dataset: str
    split: DatasetSplit | str
    dialogue_id: str
    utterance_id: int
    text: str
    emotion: EmotionLabel | str
    language: LanguageCode
    start_seconds: float
    end_seconds: float
    context_id: str | None = None
    speaker_id: str | None = None
    video_path: Path | str | None = None
    audio_path: Path | str | None = None
    text_source: Literal["human", "whisper"] = "human"
    asr_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.utterance_id < 0:
            raise ValueError("utterance_id must be non-negative")
        if self.language not in {"zh", "en"}:
            raise ValueError("language must be 'zh' or 'en'")
        if self.text_source not in {"human", "whisper"}:
            raise ValueError("text_source must be 'human' or 'whisper'")
        if self.asr_confidence is not None and not 0.0 <= self.asr_confidence <= 1.0:
            raise ValueError("asr_confidence must be within [0, 1]")
        context_id = self.context_id
        if context_id is None:
            match = (
                re.fullmatch(r"(G\d{5}_\d+)_\d+", self.dialogue_id)
                if self.dataset.lower() == "emotiontalk"
                else None
            )
            context_id = match.group(1) if match else self.dialogue_id
        if not context_id:
            raise ValueError("context_id must not be empty")
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(
            self,
            "emotion",
            normalize_emotion(str(self.emotion), dataset=self.dataset),
        )
        if self.video_path is not None:
            object.__setattr__(self, "video_path", Path(self.video_path))
        if self.audio_path is not None:
            object.__setattr__(self, "audio_path", Path(self.audio_path))

    @property
    def sample_id(self) -> str:
        return f"{self.dataset}:{self.split}:{self.dialogue_id}:{self.utterance_id}"

    @property
    def effective_context_id(self) -> str:
        return self.context_id or self.dialogue_id


@dataclass(frozen=True, slots=True)
class AnalysisSegment:
    start_seconds: float
    end_seconds: float
    text: str
    emotion: EmotionLabel | str
    probabilities: dict[str, float]
    modality_gates: dict[str, float]
    modality_available: dict[str, bool] = field(default_factory=dict)
    modality_quality: dict[str, dict[str, float]] = field(default_factory=dict)
    quality_warnings: tuple[str, ...] = field(default_factory=tuple)
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    confidence_status: str = "confident"
    calibration_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("segment end_seconds must be greater than start_seconds")
        object.__setattr__(self, "emotion", normalize_emotion(str(self.emotion)))
        if not self.modality_available:
            object.__setattr__(
                self,
                "modality_available",
                {
                    name: float(self.modality_gates.get(name, 0.0)) > 0.0
                    for name in ("text", "audio", "vision")
                },
            )
        if not self.raw_probabilities:
            object.__setattr__(
                self,
                "raw_probabilities",
                dict(self.probabilities),
            )
        if self.confidence_status not in {"confident", "uncertain"}:
            raise ValueError("confidence_status must be confident or uncertain")
        if self.calibration_temperature <= 0:
            raise ValueError("calibration_temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "text": self.text,
            "emotion": self.emotion,
            "probabilities": dict(self.probabilities),
            "raw_probabilities": dict(self.raw_probabilities),
            "confidence_status": self.confidence_status,
            "calibration_temperature": self.calibration_temperature,
            "modality_gates": dict(self.modality_gates),
            "modality_available": dict(self.modality_available),
            "modality_quality": {
                name: dict(values) for name, values in self.modality_quality.items()
            },
            "quality_warnings": list(self.quality_warnings),
        }


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    language: LanguageCode
    segments: tuple[AnalysisSegment, ...] = field(default_factory=tuple)
    model_version: str = "v2"
    runtime_profile: dict[str, float] = field(default_factory=dict)

    @property
    def global_distribution(self) -> dict[str, float]:
        if not self.segments:
            return {label: 0.0 for label in EMOTION_LABELS}
        return {
            label: sum(float(segment.probabilities.get(label, 0.0)) for segment in self.segments)
            / len(self.segments)
            for label in EMOTION_LABELS
        }

    @property
    def label_distribution(self) -> dict[str, float]:
        counts = {label: 0 for label in EMOTION_LABELS}
        for segment in self.segments:
            counts[normalize_emotion(str(segment.emotion))] += 1
        total = len(self.segments)
        if total == 0:
            return {label: 0.0 for label in EMOTION_LABELS}
        return {label: count / total for label, count in counts.items()}

    @property
    def transition_points(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(1, len(self.segments))
            if self.segments[index - 1].emotion != self.segments[index].emotion
        )

    @property
    def transition_events(self) -> tuple[dict[str, object], ...]:
        events: list[dict[str, object]] = []
        for index in self.transition_points:
            previous = self.segments[index - 1]
            current = self.segments[index]
            events.append(
                {
                    "segment_index": index,
                    "time_seconds": current.start_seconds,
                    "from_emotion": str(previous.emotion),
                    "to_emotion": str(current.emotion),
                    "confidence": float(current.probabilities.get(str(current.emotion), 0.0)),
                }
            )
        return tuple(events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
            "global_distribution": self.global_distribution,
            "label_distribution": self.label_distribution,
            "transition_points": list(self.transition_points),
            "transition_events": list(self.transition_events),
            "model_version": self.model_version,
            "runtime_profile": dict(self.runtime_profile),
        }
