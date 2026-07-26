from pathlib import Path

import numpy as np
import pytest
import torch
import bimer.inference as inference

from bimer.baselines import EarlyFusionMLP
from bimer.model import FusionOutput
from bimer.inference import (
    DialogueAnalyzer,
    FeatureBundle,
    TranscriptSegment,
    normalize_transcript_segments,
    validate_video_input,
)
from bimer.calibration import (
    CalibrationProfile,
    LanguageCalibration,
)
from bimer.runtime_cache import RuntimeFeatureCache


def test_normalize_segments_merges_short_and_splits_long_segments():
    segments = [
        TranscriptSegment(0.0, 0.4, "Hi"),
        TranscriptSegment(0.4, 2.0, "there"),
        TranscriptSegment(2.0, 22.0, "First sentence. Second sentence."),
    ]
    normalized = normalize_transcript_segments(segments, minimum_seconds=1.0, maximum_seconds=15.0)
    assert normalized[0].start_seconds == 0.0
    assert normalized[0].end_seconds == 2.0
    assert normalized[0].text == "Hi there"
    assert all(segment.duration <= 15.0 for segment in normalized)


def test_chinese_long_segment_is_partitioned_without_duplicate_text():
    text = "这是一个没有标点但是需要按照字符切分的很长中文句子用于验证"
    parts = normalize_transcript_segments(
        [TranscriptSegment(0.0, 31.0, text)], maximum_seconds=15.0
    )

    assert len(parts) == 3
    assert "".join(part.text for part in parts) == text
    assert len({part.text for part in parts}) == 3


def test_validate_video_rejects_bad_extension_before_probing(tmp_path):
    path = tmp_path / "dialogue.txt"
    path.write_text("not video", encoding="utf-8")
    with pytest.raises(ValueError, match="MP4 or MOV"):
        validate_video_input(path)


class _Transcriber:
    def transcribe(self, video_path: Path, language: str):
        assert video_path.name == "sample.mp4"
        return "zh", [
            TranscriptSegment(0.0, 2.0, "你好"),
            TranscriptSegment(2.0, 4.0, "太好了"),
        ]


class _Features:
    last_texts = None

    def extract(self, video_path: Path, segments):
        self.last_texts = [segment.text for segment in segments]
        count = len(segments)
        return FeatureBundle(
            text=np.ones((count, 4), dtype=np.float32),
            audio=np.ones((count, 6), dtype=np.float32),
            vision=np.ones((count, 5), dtype=np.float32),
            modality_mask=np.ones((count, 3), dtype=np.bool_),
        )


def test_dialogue_analyzer_returns_public_analysis_result(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"placeholder")
    model = EarlyFusionMLP((4, 6, 5), hidden_dim=8, num_classes=7)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    analyzer = DialogueAnalyzer(
        transcriber=_Transcriber(),
        feature_pipeline=_Features(),
        model=model,
        device=torch.device("cpu"),
        validator=lambda _: None,
    )
    result = analyzer.analyze(video, language="auto")
    assert result.language == "zh"
    assert len(result.segments) == 2
    assert result.segments[0].emotion == "neutral"
    assert set(result.segments[0].probabilities) == {
        "neutral",
        "joy",
        "sadness",
        "anger",
        "surprise",
        "fear",
        "disgust",
    }
    assert result.segments[0].modality_available == {
        "text": True, "audio": True, "vision": True
    }
    assert set(result.segments[0].modality_quality) == {"text", "audio", "vision"}
    assert result.segments[0].raw_probabilities == result.segments[0].probabilities
    assert result.model_version == "v2"
    assert set(result.runtime_profile) >= {"transcription", "fusion"}


def test_dialogue_analyzer_applies_language_calibration_and_uncertainty(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"placeholder")
    model = EarlyFusionMLP((4, 6, 5), hidden_dim=8, num_classes=7)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    profile = CalibrationProfile(
        languages={
            "zh": LanguageCalibration(
                temperature=2.0,
                threshold=0.8,
                enabled=True,
                before={},
                after={},
            )
        }
    )
    analyzer = DialogueAnalyzer(
        transcriber=_Transcriber(),
        feature_pipeline=_Features(),
        model=model,
        calibration_profile=profile,
        model_version="v3_ranked",
        validator=lambda _: None,
    )

    result = analyzer.analyze(video)

    assert result.model_version == "v3_ranked"
    assert result.segments[0].calibration_temperature == 2.0
    assert result.segments[0].confidence_status == "uncertain"


def test_dialogue_analyzer_rejects_silent_transcription(tmp_path):
    class Silent:
        def transcribe(self, video_path: Path, language: str):
            return "en", []

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"placeholder")
    analyzer = DialogueAnalyzer(
        transcriber=Silent(),
        feature_pipeline=_Features(),
        model=EarlyFusionMLP((4, 6, 5), hidden_dim=8),
        validator=lambda _: None,
    )
    with pytest.raises(ValueError, match="No valid speech"):
        analyzer.analyze(video)


def test_reanalysis_uses_user_edited_transcript(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"placeholder")
    features = _Features()
    analyzer = DialogueAnalyzer(
        transcriber=_Transcriber(),
        feature_pipeline=features,
        model=EarlyFusionMLP((4, 6, 5), hidden_dim=8),
        validator=lambda _: None,
    )
    analyzer.analyze_segments(
        video,
        detected_language="zh",
        segments=[TranscriptSegment(0.0, 2.0, "人工修改文本")],
    )
    assert features.last_texts == ["人工修改文本"]


def test_long_dialogue_inference_uses_32_sentence_overlapping_windows(tmp_path):
    class LongFeatures:
        def extract(self, _video_path, segments):
            count = len(segments)
            return FeatureBundle(
                text=np.ones((count, 4), dtype=np.float32),
                audio=np.ones((count, 6), dtype=np.float32),
                vision=np.ones((count, 5), dtype=np.float32),
                modality_mask=np.ones((count, 3), dtype=np.bool_),
            )

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lengths = []

        def forward(self, *, text_features, modality_mask, **_inputs):
            self.lengths.append(text_features.shape[1])
            logits = torch.zeros(*text_features.shape[:2], 7)
            gates = modality_mask.float() / modality_mask.sum(dim=-1, keepdim=True)
            return FusionOutput(logits=logits, gates=gates)

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"placeholder")
    model = RecordingModel()
    analyzer = DialogueAnalyzer(
        transcriber=_Transcriber(),
        feature_pipeline=LongFeatures(),
        model=model,
        validator=lambda _: None,
    )
    segments = [
        TranscriptSegment(float(index), float(index + 1), f"line {index}")
        for index in range(70)
    ]

    result = analyzer.analyze_segments(
        video, detected_language="en", segments=segments
    )

    assert len(result.segments) == 70
    assert model.lengths == [32, 32, 22]


def test_pretrained_pipeline_decodes_audio_once_and_emits_quality(monkeypatch):
    calls = []
    waveform = np.full(4 * 16000, 0.1, dtype=np.float32)
    monkeypatch.setattr(
        inference,
        "_extract_full_waveform",
        lambda path: calls.append(path) or waveform,
    )

    class Text:
        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    class Audio:
        def __init__(self):
            self.lengths = []

        def encode(self, waveforms):
            self.lengths = [len(value) for value in waveforms]
            return np.ones((len(waveforms), 6), dtype=np.float32)

    class Vision:
        def encode_video_segment_with_quality(
            self, _path, *, start_seconds, end_seconds, face_cropper
        ):
            assert face_cropper is not None
            return (
                np.ones((1, 5), dtype=np.float32),
                True,
                np.array([1.0, 1.0, 1.0, 0.25], dtype=np.float32),
            )

    audio = Audio()
    pipeline = inference.PretrainedFeaturePipeline(
        text_extractor=Text(),
        audio_extractor=audio,
        vision_extractor=Vision(),
        face_cropper=object(),
    )
    path = Path("sample.mp4")
    bundle = pipeline.extract(
        path,
        [
            TranscriptSegment(0.0, 2.0, "hello", asr_confidence=0.8),
            TranscriptSegment(2.0, 4.0, "world", asr_confidence=0.9),
        ],
    )

    assert calls == [path]
    assert audio.lengths == [32000, 32000]
    assert bundle.modality_quality.shape == (2, 3, 4)
    assert bundle.modality_quality[0, 0, :2].tolist() == pytest.approx([0.0, 0.8])
    assert bundle.modality_quality[0, 2].tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 0.25]
    )


def test_pretrained_pipeline_rejects_silent_audio(monkeypatch):
    monkeypatch.setattr(
        inference,
        "_extract_full_waveform",
        lambda _path: np.zeros(32000, dtype=np.float32),
    )

    class Encoder:
        def encode(self, values):
            return np.ones((len(values), 4), dtype=np.float32)

    pipeline = inference.PretrainedFeaturePipeline(
        text_extractor=Encoder(),
        audio_extractor=Encoder(),
        vision_extractor=object(),
        face_cropper=object(),
    )

    with pytest.raises(ValueError, match="No valid speech audio"):
        pipeline.extract(Path("silent.mp4"), [TranscriptSegment(0.0, 2.0, "text")])


def test_pretrained_pipeline_text_edit_reuses_cached_audio_and_vision(
    tmp_path, monkeypatch
):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    calls = {"text": 0, "audio_decode": 0, "audio": 0, "vision": 0}
    monkeypatch.setattr(
        inference,
        "_extract_full_waveform",
        lambda _path: calls.__setitem__("audio_decode", calls["audio_decode"] + 1)
        or np.full(32000, 0.1, dtype=np.float32),
    )

    class Text:
        def encode(self, values):
            calls["text"] += 1
            return np.ones((len(values), 4), dtype=np.float32)

    class Audio:
        def encode(self, values):
            calls["audio"] += 1
            return np.ones((len(values), 6), dtype=np.float32)

    class Vision:
        def encode_video_segment_with_quality(self, *_args, **_kwargs):
            calls["vision"] += 1
            return (
                np.ones((1, 5), dtype=np.float32),
                True,
                np.ones(4, dtype=np.float32),
            )

    pipeline = inference.PretrainedFeaturePipeline(
        text_extractor=Text(),
        audio_extractor=Audio(),
        vision_extractor=Vision(),
        face_cropper=object(),
        cache=RuntimeFeatureCache(tmp_path / "cache"),
        encoder_versions={"text": "t1", "audio": "a1", "vision": "v1"},
    )
    pipeline.extract(video, [TranscriptSegment(0.0, 2.0, "old")])
    pipeline.extract(video, [TranscriptSegment(0.0, 2.0, "new")])

    assert calls == {"text": 2, "audio_decode": 1, "audio": 1, "vision": 1}
    assert set(pipeline.last_runtime_profile) == {"text", "audio", "vision"}
