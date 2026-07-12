from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

import numpy as np
import torch
from torch import nn

from .feature_extractors import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    VisionFeatureExtractor,
    YuNetFaceCropper,
)
from .labels import EMOTION_LABELS
from .schema import AnalysisResult, AnalysisSegment

RequestedLanguage = Literal["auto", "zh", "en"]


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    modality_mask: np.ndarray

    def __post_init__(self) -> None:
        rows = self.text.shape[0]
        if not all(array.shape[0] == rows for array in (self.audio, self.vision, self.modality_mask)):
            raise ValueError("feature bundle arrays must share a row count")
        if self.modality_mask.shape != (rows, 3):
            raise ValueError("modality_mask must have shape [segments, 3]")


class Transcriber(Protocol):
    def transcribe(
        self, video_path: Path, language: RequestedLanguage
    ) -> tuple[Literal["zh", "en"], Sequence[TranscriptSegment]]: ...


class FeaturePipeline(Protocol):
    def extract(
        self, video_path: Path, segments: Sequence[TranscriptSegment]
    ) -> FeatureBundle: ...


def _split_long_segment(
    segment: TranscriptSegment, maximum_seconds: float
) -> list[TranscriptSegment]:
    parts = max(1, math.ceil(segment.duration / maximum_seconds))
    text_parts = [
        value.strip()
        for value in re.split(r"(?<=[.!?。！？])\s*", segment.text)
        if value.strip()
    ]
    if len(text_parts) != parts:
        words = segment.text.split()
        if len(words) >= parts:
            text_parts = [
                " ".join(chunk.tolist())
                for chunk in np.array_split(np.asarray(words, dtype=str), parts)
            ]
        else:
            text_parts = [segment.text] * parts
    boundaries = np.linspace(segment.start_seconds, segment.end_seconds, parts + 1)
    return [
        TranscriptSegment(float(boundaries[index]), float(boundaries[index + 1]), text_parts[index])
        for index in range(parts)
    ]


def normalize_transcript_segments(
    segments: Sequence[TranscriptSegment],
    *,
    minimum_seconds: float = 1.0,
    maximum_seconds: float = 15.0,
) -> list[TranscriptSegment]:
    if minimum_seconds <= 0 or maximum_seconds <= minimum_seconds:
        raise ValueError("segment duration bounds are invalid")
    clean = sorted(
        (
            segment
            for segment in segments
            if segment.text.strip() and segment.end_seconds > segment.start_seconds
        ),
        key=lambda segment: segment.start_seconds,
    )
    merged: list[TranscriptSegment] = []
    index = 0
    while index < len(clean):
        segment = clean[index]
        if segment.duration < minimum_seconds and index + 1 < len(clean):
            following = clean[index + 1]
            merged.append(
                TranscriptSegment(
                    segment.start_seconds,
                    following.end_seconds,
                    f"{segment.text.strip()} {following.text.strip()}".strip(),
                )
            )
            index += 2
            continue
        if segment.duration < minimum_seconds and merged:
            previous = merged.pop()
            merged.append(
                TranscriptSegment(
                    previous.start_seconds,
                    segment.end_seconds,
                    f"{previous.text.strip()} {segment.text.strip()}".strip(),
                )
            )
        else:
            merged.append(segment)
        index += 1

    normalized: list[TranscriptSegment] = []
    for segment in merged:
        normalized.extend(_split_long_segment(segment, maximum_seconds))
    return normalized


def validate_video_input(
    video_path: Path | str,
    *,
    max_bytes: int = 500 * 1024 * 1024,
    max_duration_seconds: float = 180.0,
) -> None:
    path = Path(video_path)
    if path.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError("Only MP4 or MOV video files are supported")
    if not path.exists():
        raise ValueError(f"Video does not exist: {path}")
    if path.stat().st_size > max_bytes:
        raise ValueError("Video exceeds the 500 MB limit")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    duration = float(metadata.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        raise ValueError("Video has no measurable duration")
    if duration > max_duration_seconds:
        raise ValueError("Video exceeds the 3 minute limit")
    if not any(stream.get("codec_type") == "audio" for stream in metadata.get("streams", [])):
        raise ValueError("Video has no audio stream")


class FasterWhisperTranscriber:
    def __init__(self, model_size: str = "small", *, device: str = "cpu") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install bimer[inference] to use Whisper") from exc
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self, video_path: Path, language: RequestedLanguage
    ) -> tuple[Literal["zh", "en"], list[TranscriptSegment]]:
        selected = None if language == "auto" else language
        raw_segments, info = self.model.transcribe(
            str(video_path),
            language=selected,
            vad_filter=True,
            word_timestamps=False,
        )
        detected: Literal["zh", "en"] = "zh" if str(info.language).startswith("zh") else "en"
        return detected, [
            TranscriptSegment(float(segment.start), float(segment.end), segment.text.strip())
            for segment in raw_segments
            if segment.text.strip()
        ]


def _extract_waveform(video_path: Path, segment: TranscriptSegment) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(segment.start_seconds),
            "-to",
            str(segment.end_seconds),
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _extract_video_clip(video_path: Path, segment: TranscriptSegment, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(segment.start_seconds),
            "-to",
            str(segment.end_seconds),
            "-i",
            str(video_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-y",
            str(output),
        ],
        check=True,
    )


class PretrainedFeaturePipeline:
    def __init__(
        self,
        *,
        text_extractor: TextFeatureExtractor,
        audio_extractor: AudioFeatureExtractor,
        vision_extractor: VisionFeatureExtractor,
        face_cropper: YuNetFaceCropper,
    ) -> None:
        self.text_extractor = text_extractor
        self.audio_extractor = audio_extractor
        self.vision_extractor = vision_extractor
        self.face_cropper = face_cropper

    def extract(
        self, video_path: Path, segments: Sequence[TranscriptSegment]
    ) -> FeatureBundle:
        text = self.text_extractor.encode([segment.text for segment in segments])
        waveforms = [_extract_waveform(video_path, segment) for segment in segments]
        audio_available = np.array([waveform.size > 0 for waveform in waveforms], dtype=np.bool_)
        safe_waveforms = [waveform if waveform.size else np.zeros(160, np.float32) for waveform in waveforms]
        audio = self.audio_extractor.encode(safe_waveforms)

        vision_rows: list[np.ndarray] = []
        vision_available: list[bool] = []
        with tempfile.TemporaryDirectory(prefix="bimer-clips-") as directory:
            root = Path(directory)
            for index, segment in enumerate(segments):
                clip = root / f"segment-{index:04d}.mp4"
                _extract_video_clip(video_path, segment, clip)
                feature, available = self.vision_extractor.encode_video(
                    clip, face_cropper=self.face_cropper
                )
                vision_rows.append(feature[0])
                vision_available.append(available)
        vision = np.stack(vision_rows).astype(np.float32)
        modality_mask = np.stack(
            (
                np.ones(len(segments), dtype=np.bool_),
                audio_available,
                np.asarray(vision_available, dtype=np.bool_),
            ),
            axis=-1,
        )
        return FeatureBundle(text=text, audio=audio, vision=vision, modality_mask=modality_mask)


class DialogueAnalyzer:
    def __init__(
        self,
        *,
        transcriber: Transcriber,
        feature_pipeline: FeaturePipeline,
        model: nn.Module,
        device: torch.device | None = None,
        validator: Callable[[Path], None] = validate_video_input,
    ) -> None:
        self.transcriber = transcriber
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.device = device or torch.device("cpu")
        self.validator = validator
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def analyze(
        self, video_path: Path | str, language: RequestedLanguage = "auto"
    ) -> AnalysisResult:
        path = Path(video_path)
        self.validator(path)
        detected_language, raw_segments = self.transcriber.transcribe(path, language)
        segments = normalize_transcript_segments(raw_segments)
        if not segments:
            raise ValueError("No valid speech segments were detected")
        return self._analyze_segments(path, detected_language, segments)

    def transcribe(
        self, video_path: Path | str, language: RequestedLanguage = "auto"
    ) -> tuple[Literal["zh", "en"], list[TranscriptSegment]]:
        path = Path(video_path)
        self.validator(path)
        detected_language, raw_segments = self.transcriber.transcribe(path, language)
        segments = normalize_transcript_segments(raw_segments)
        if not segments:
            raise ValueError("No valid speech segments were detected")
        return detected_language, segments

    @torch.inference_mode()
    def analyze_segments(
        self,
        video_path: Path | str,
        *,
        detected_language: Literal["zh", "en"],
        segments: Sequence[TranscriptSegment],
    ) -> AnalysisResult:
        path = Path(video_path)
        self.validator(path)
        normalized = normalize_transcript_segments(segments)
        if not normalized:
            raise ValueError("No valid speech segments were provided")
        return self._analyze_segments(path, detected_language, normalized)

    def _analyze_segments(
        self,
        path: Path,
        detected_language: Literal["zh", "en"],
        segments: Sequence[TranscriptSegment],
    ) -> AnalysisResult:
        features = self.feature_pipeline.extract(path, segments)
        attention = torch.ones(1, len(segments), dtype=torch.bool, device=self.device)
        output = self.model(
            text_features=torch.from_numpy(features.text).unsqueeze(0).to(self.device),
            audio_features=torch.from_numpy(features.audio).unsqueeze(0).to(self.device),
            vision_features=torch.from_numpy(features.vision).unsqueeze(0).to(self.device),
            modality_mask=torch.from_numpy(features.modality_mask).unsqueeze(0).to(self.device),
            attention_mask=attention,
            language_ids=torch.tensor(
                [0 if detected_language == "en" else 1], dtype=torch.long, device=self.device
            ),
        )
        probabilities = torch.softmax(output.logits[0], dim=-1).cpu().numpy()
        gates = output.gates[0].cpu().numpy()
        results: list[AnalysisSegment] = []
        for index, segment in enumerate(segments):
            label_index = int(probabilities[index].argmax())
            results.append(
                AnalysisSegment(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=segment.text,
                    emotion=EMOTION_LABELS[label_index],
                    probabilities={
                        label: float(probabilities[index, position])
                        for position, label in enumerate(EMOTION_LABELS)
                    },
                    modality_gates={
                        name: float(gates[index, position])
                        for position, name in enumerate(("text", "audio", "vision"))
                    },
                )
            )
        return AnalysisResult(language=detected_language, segments=tuple(results))


_DEFAULT_ANALYZER: DialogueAnalyzer | None = None


def configure_default_analyzer(analyzer: DialogueAnalyzer) -> None:
    global _DEFAULT_ANALYZER
    _DEFAULT_ANALYZER = analyzer


def analyze_dialogue(
    video_path: Path,
    language: RequestedLanguage = "auto",
) -> AnalysisResult:
    if _DEFAULT_ANALYZER is None:
        raise RuntimeError("No default analyzer is configured")
    return _DEFAULT_ANALYZER.analyze(video_path, language)
