from typing import Final, Literal

EmotionLabel = Literal[
    "neutral",
    "joy",
    "sadness",
    "anger",
    "surprise",
    "fear",
    "disgust",
]

EMOTION_LABELS: Final[tuple[EmotionLabel, ...]] = (
    "neutral",
    "joy",
    "sadness",
    "anger",
    "surprise",
    "fear",
    "disgust",
)

_ALIASES: Final[dict[str, EmotionLabel]] = {
    "neutral": "neutral",
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "surprised": "surprise",
    "surprise": "surprise",
    "fearful": "fear",
    "fear": "fear",
    "disgusted": "disgust",
    "disgust": "disgust",
}


def normalize_emotion(label: str, *, dataset: str | None = None) -> EmotionLabel:
    """Map source-specific labels onto the public seven-emotion schema."""

    normalized = label.strip().lower()
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        source = f" for dataset {dataset!r}" if dataset else ""
        raise ValueError(f"Unknown emotion label {label!r}{source}") from exc


def emotion_index(label: str) -> int:
    normalized = normalize_emotion(label)
    return EMOTION_LABELS.index(normalized)
