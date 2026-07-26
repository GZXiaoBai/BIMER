from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

import numpy as np
import torch
from torch import nn

from .calibration import CalibrationProfile, apply_temperature
from .feature_extractors import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    VisionFeatureExtractor,
    YuNetFaceCropper,
    prepare_audio_waveforms,
)
from .labels import EMOTION_LABELS
from .quality import MODALITY_QUALITY_NAMES, audio_quality, text_quality
from .runtime_cache import RuntimeFeatureCache
from .schema import AnalysisResult, AnalysisSegment

RequestedLanguage = Literal["auto", "zh", "en"]


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str
    asr_confidence: float | None = None

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    modality_mask: np.ndarray
    modality_quality: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = self.text.shape[0]
        if not all(
            array.shape[0] == rows for array in (self.audio, self.vision, self.modality_mask)
        ):
            raise ValueError("feature bundle arrays must share a row count")
        if self.modality_mask.shape != (rows, 3):
            raise ValueError("modality_mask must have shape [segments, 3]")
        if self.modality_quality is None:
            object.__setattr__(
                self,
                "modality_quality",
                np.repeat(self.modality_mask.astype(np.float32)[..., None], 4, axis=-1),
            )
        if np.asarray(self.modality_quality).shape != (rows, 3, 4):
            raise ValueError("modality_quality must have shape [segments, 3, 4]")


class Transcriber(Protocol):
    def transcribe(
        self, video_path: Path, language: RequestedLanguage
    ) -> tuple[Literal["zh", "en"], Sequence[TranscriptSegment]]: ...


class FeaturePipeline(Protocol):
    def extract(self, video_path: Path, segments: Sequence[TranscriptSegment]) -> FeatureBundle: ...


def _split_long_segment(
    segment: TranscriptSegment, maximum_seconds: float
) -> list[TranscriptSegment]:
    parts = max(1, math.ceil(segment.duration / maximum_seconds))
    text_parts = [
        value.strip() for value in re.split(r"(?<=[.!?。！？])\s*", segment.text) if value.strip()
    ]
    if len(text_parts) != parts:
        words = segment.text.split()
        uses_words = len(words) >= parts
        units = words if uses_words else list(segment.text)
        text_parts = [
            (" " if uses_words else "").join(chunk.tolist()).strip()
            for chunk in np.array_split(np.asarray(units, dtype=str), parts)
        ]
    boundaries = np.linspace(segment.start_seconds, segment.end_seconds, parts + 1)
    return [
        TranscriptSegment(
            float(boundaries[index]),
            float(boundaries[index + 1]),
            text_parts[index],
            segment.asr_confidence,
        )
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
                    _mean_confidence(segment, following),
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
                    _mean_confidence(previous, segment),
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
            TranscriptSegment(
                float(segment.start),
                float(segment.end),
                segment.text.strip(),
                float(np.clip(np.exp(float(getattr(segment, "avg_logprob", -20.0))), 0.0, 1.0)),
            )
            for segment in raw_segments
            if segment.text.strip()
        ]


def _mean_confidence(*segments: TranscriptSegment) -> float | None:
    values = [segment.asr_confidence for segment in segments if segment.asr_confidence is not None]
    return float(np.mean(values)) if values else None


def _extract_full_waveform(video_path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
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


def _slice_waveform(
    waveform: np.ndarray,
    segment: TranscriptSegment,
    *,
    sampling_rate: int = 16000,
) -> np.ndarray:
    start = max(0, int(round(segment.start_seconds * sampling_rate)))
    end = min(len(waveform), int(round(segment.end_seconds * sampling_rate)))
    return waveform[start:end].copy()


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
        cache: RuntimeFeatureCache | None = None,
        encoder_versions: dict[str, str] | None = None,
    ) -> None:
        self.text_extractor = text_extractor
        self.audio_extractor = audio_extractor
        self.vision_extractor = vision_extractor
        self.face_cropper = face_cropper
        self.cache = cache
        self.encoder_versions = encoder_versions or {
            "text": type(text_extractor).__name__,
            "audio": type(audio_extractor).__name__,
            "vision": type(vision_extractor).__name__,
        }
        self.last_runtime_profile: dict[str, float] = {}
        self._digest_cache: dict[Path, tuple[int, int, str]] = {}

    def _video_sha256(self, video_path: Path) -> str:
        stat = video_path.stat()
        cached = self._digest_cache.get(video_path.resolve())
        identity = (stat.st_size, stat.st_mtime_ns)
        if cached is not None and cached[:2] == identity:
            return cached[2]
        digest = RuntimeFeatureCache.file_sha256(video_path)
        self._digest_cache[video_path.resolve()] = (*identity, digest)
        return digest

    def _load_or_compute(
        self,
        namespace: str,
        payload: dict[str, object],
        compute: Callable[[], dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        if self.cache is None:
            return compute()
        key = self.cache.key(namespace, payload)
        cached = self.cache.load(key)
        if cached is not None:
            return cached
        arrays = compute()
        self.cache.store(key, arrays)
        return arrays

    def extract(self, video_path: Path, segments: Sequence[TranscriptSegment]) -> FeatureBundle:
        timestamps = [
            [float(segment.start_seconds), float(segment.end_seconds)] for segment in segments
        ]
        common = {
            "video_sha256": self._video_sha256(video_path) if self.cache else "",
            "timestamps": timestamps,
        }
        self.last_runtime_profile = {}

        started = time.perf_counter()
        text_arrays = self._load_or_compute(
            "text",
            {
                **common,
                "texts": [segment.text for segment in segments],
                "asr_confidence": [segment.asr_confidence for segment in segments],
                "encoder": self.encoder_versions["text"],
            },
            lambda: {
                "features": self.text_extractor.encode(
                    [segment.text for segment in segments]
                ).astype(np.float32),
                "quality": np.stack(
                    [
                        text_quality(
                            segment.text,
                            source=("whisper" if segment.asr_confidence is not None else "human"),
                            asr_confidence=segment.asr_confidence,
                        )
                        for segment in segments
                    ]
                ).astype(np.float32),
            },
        )
        text = text_arrays["features"].astype(np.float32)
        text_quality_rows = text_arrays["quality"].astype(np.float32)
        self.last_runtime_profile["text"] = time.perf_counter() - started

        def compute_audio() -> dict[str, np.ndarray]:
            full_waveform = _extract_full_waveform(video_path)
            waveforms = [_slice_waveform(full_waveform, segment) for segment in segments]
            safe_waveforms, available = prepare_audio_waveforms(waveforms)
            quality_rows = np.stack([audio_quality(waveform) for waveform in waveforms]).astype(
                np.float32
            )
            if not np.any(quality_rows[:, 2] > 0.0):
                raise ValueError("No valid speech audio was detected")
            return {
                "features": self.audio_extractor.encode(safe_waveforms).astype(np.float32),
                "available": available.astype(np.bool_),
                "quality": quality_rows,
            }

        started = time.perf_counter()
        audio_arrays = self._load_or_compute(
            "audio",
            {**common, "encoder": self.encoder_versions["audio"]},
            compute_audio,
        )
        audio = audio_arrays["features"].astype(np.float32)
        audio_available = audio_arrays["available"].astype(np.bool_)
        audio_quality_rows = audio_arrays["quality"].astype(np.float32)
        self.last_runtime_profile["audio"] = time.perf_counter() - started

        def compute_vision() -> dict[str, np.ndarray]:
            vision_rows: list[np.ndarray] = []
            vision_available: list[bool] = []
            vision_quality_rows: list[np.ndarray] = []
            with tempfile.TemporaryDirectory(prefix="bimer-clips-") as directory:
                root = Path(directory)
                for index, segment in enumerate(segments):
                    segment_encoder = getattr(
                        self.vision_extractor,
                        "encode_video_segment_with_quality",
                        None,
                    )
                    if segment_encoder is not None:
                        feature, available, quality = segment_encoder(
                            video_path,
                            start_seconds=segment.start_seconds,
                            end_seconds=segment.end_seconds,
                            face_cropper=self.face_cropper,
                        )
                    else:
                        clip = root / f"segment-{index:04d}.mp4"
                        _extract_video_clip(video_path, segment, clip)
                        quality_encoder = getattr(
                            self.vision_extractor,
                            "encode_video_with_quality",
                            None,
                        )
                        if quality_encoder is None:
                            feature, available = self.vision_extractor.encode_video(
                                clip,
                                face_cropper=self.face_cropper,
                            )
                            quality = np.full(
                                4,
                                float(available),
                                dtype=np.float32,
                            )
                        else:
                            feature, available, quality = quality_encoder(
                                clip,
                                face_cropper=self.face_cropper,
                            )
                    vision_rows.append(feature[0])
                    vision_available.append(available)
                    vision_quality_rows.append(np.asarray(quality, dtype=np.float32))
            return {
                "features": np.stack(vision_rows).astype(np.float32),
                "available": np.asarray(vision_available, dtype=np.bool_),
                "quality": np.stack(vision_quality_rows).astype(np.float32),
            }

        started = time.perf_counter()
        vision_arrays = self._load_or_compute(
            "vision",
            {**common, "encoder": self.encoder_versions["vision"]},
            compute_vision,
        )
        vision = vision_arrays["features"].astype(np.float32)
        vision_available = vision_arrays["available"].astype(np.bool_)
        vision_quality_rows = vision_arrays["quality"].astype(np.float32)
        self.last_runtime_profile["vision"] = time.perf_counter() - started
        modality_mask = np.stack(
            (
                np.ones(len(segments), dtype=np.bool_),
                audio_available,
                vision_available,
            ),
            axis=-1,
        )
        modality_quality = np.stack(
            (text_quality_rows, audio_quality_rows, vision_quality_rows),
            axis=1,
        ).astype(np.float32)
        return FeatureBundle(
            text=text,
            audio=audio,
            vision=vision,
            modality_mask=modality_mask,
            modality_quality=modality_quality,
        )


class DialogueAnalyzer:
    def __init__(
        self,
        *,
        transcriber: Transcriber,
        feature_pipeline: FeaturePipeline,
        model: nn.Module,
        device: torch.device | None = None,
        validator: Callable[[Path], None] = validate_video_input,
        calibration_profile: CalibrationProfile | None = None,
        model_version: str = "v2",
    ) -> None:
        self.transcriber = transcriber
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.device = device or torch.device("cpu")
        self.validator = validator
        self.calibration_profile = calibration_profile
        self.model_version = model_version
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def analyze(
        self, video_path: Path | str, language: RequestedLanguage = "auto"
    ) -> AnalysisResult:
        path = Path(video_path)
        self.validator(path)
        transcription_started = time.perf_counter()
        detected_language, raw_segments = self.transcriber.transcribe(path, language)
        transcription_seconds = time.perf_counter() - transcription_started
        segments = normalize_transcript_segments(raw_segments)
        if not segments:
            raise ValueError("No valid speech segments were detected")
        return self._analyze_segments(
            path,
            detected_language,
            segments,
            initial_runtime={"transcription": transcription_seconds},
        )

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
        return self._analyze_segments(
            path,
            detected_language,
            normalized,
            initial_runtime={"transcription": 0.0},
        )

    def _analyze_segments(
        self,
        path: Path,
        detected_language: Literal["zh", "en"],
        segments: Sequence[TranscriptSegment],
        initial_runtime: dict[str, float] | None = None,
    ) -> AnalysisResult:
        feature_started = time.perf_counter()
        features = self.feature_pipeline.extract(path, segments)
        feature_seconds = time.perf_counter() - feature_started
        runtime_profile = dict(initial_runtime or {})
        runtime_profile.update(getattr(self.feature_pipeline, "last_runtime_profile", {}))
        if not getattr(self.feature_pipeline, "last_runtime_profile", None):
            runtime_profile["feature_extraction"] = feature_seconds
        fusion_started = time.perf_counter()
        count = len(segments)
        probability_sum = np.zeros((count, len(EMOTION_LABELS)), dtype=np.float64)
        gate_sum = np.zeros((count, 3), dtype=np.float64)
        appearances = np.zeros(count, dtype=np.float64)
        for start in range(0, count, 24):
            end = min(start + 32, count)
            length = end - start
            attention = torch.ones(1, length, dtype=torch.bool, device=self.device)
            output = self.model(
                text_features=torch.from_numpy(features.text[start:end])
                .unsqueeze(0)
                .to(self.device),
                audio_features=torch.from_numpy(features.audio[start:end])
                .unsqueeze(0)
                .to(self.device),
                vision_features=torch.from_numpy(features.vision[start:end])
                .unsqueeze(0)
                .to(self.device),
                modality_mask=torch.from_numpy(features.modality_mask[start:end])
                .unsqueeze(0)
                .to(self.device),
                modality_quality=torch.from_numpy(np.asarray(features.modality_quality)[start:end])
                .unsqueeze(0)
                .to(self.device),
                attention_mask=attention,
                language_ids=torch.tensor(
                    [0 if detected_language == "en" else 1],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
            probability_sum[start:end] += torch.softmax(output.logits[0], dim=-1).cpu().numpy()
            gate_sum[start:end] += output.gates[0].cpu().numpy()
            appearances[start:end] += 1.0
            if end == count:
                break
        probabilities = probability_sum / appearances[:, None]
        gates = gate_sum / appearances[:, None]
        runtime_profile["fusion"] = time.perf_counter() - fusion_started
        raw_probabilities = probabilities.copy()
        temperature = 1.0
        threshold = 0.50
        if (
            self.calibration_profile is not None
            and detected_language in self.calibration_profile.languages
        ):
            calibration = self.calibration_profile.languages[detected_language]
            temperature = calibration.temperature
            threshold = calibration.threshold
            probabilities = apply_temperature(probabilities, temperature)
        results: list[AnalysisSegment] = []
        for index, segment in enumerate(segments):
            label_index = int(probabilities[index].argmax())
            available = {
                name: bool(features.modality_mask[index, position])
                for position, name in enumerate(("text", "audio", "vision"))
            }
            quality = {
                name: {
                    field: float(
                        np.asarray(features.modality_quality)[index, position, field_index]
                    )
                    for field_index, field in enumerate(MODALITY_QUALITY_NAMES[name])
                }
                for position, name in enumerate(("text", "audio", "vision"))
            }
            warnings = tuple(
                f"{name}_unavailable" for name, value in available.items() if not value
            )
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
                    raw_probabilities={
                        label: float(raw_probabilities[index, position])
                        for position, label in enumerate(EMOTION_LABELS)
                    },
                    confidence_status=(
                        "confident"
                        if float(probabilities[index].max()) >= threshold
                        else "uncertain"
                    ),
                    calibration_temperature=float(temperature),
                    modality_gates={
                        name: float(gates[index, position])
                        for position, name in enumerate(("text", "audio", "vision"))
                    },
                    modality_available=available,
                    modality_quality=quality,
                    quality_warnings=warnings,
                )
            )
        return AnalysisResult(
            language=detected_language,
            segments=tuple(results),
            model_version=self.model_version,
            runtime_profile=runtime_profile,
        )


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
