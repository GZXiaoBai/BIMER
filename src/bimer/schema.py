from __future__ import annotations

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
    speaker_id: str | None = None
    video_path: Path | str | None = None
    audio_path: Path | str | None = None

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.utterance_id < 0:
            raise ValueError("utterance_id must be non-negative")
        if self.language not in {"zh", "en"}:
            raise ValueError("language must be 'zh' or 'en'")
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


@dataclass(frozen=True, slots=True)
class AnalysisSegment:
    start_seconds: float
    end_seconds: float
    text: str
    emotion: EmotionLabel | str
    probabilities: dict[str, float]
    modality_gates: dict[str, float]

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("segment end_seconds must be greater than start_seconds")
        object.__setattr__(self, "emotion", normalize_emotion(str(self.emotion)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "text": self.text,
            "emotion": self.emotion,
            "probabilities": dict(self.probabilities),
            "modality_gates": dict(self.modality_gates),
        }


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    language: LanguageCode
    segments: tuple[AnalysisSegment, ...] = field(default_factory=tuple)

    @property
    def global_distribution(self) -> dict[str, float]:
        counts = {label: 0 for label in EMOTION_LABELS}
        for segment in self.segments:
            counts[str(segment.emotion)] += 1
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
            "global_distribution": self.global_distribution,
            "transition_points": list(self.transition_points),
        }

