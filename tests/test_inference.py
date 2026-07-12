from pathlib import Path

import numpy as np
import pytest
import torch

from bimer.baselines import EarlyFusionMLP
from bimer.inference import (
    DialogueAnalyzer,
    FeatureBundle,
    TranscriptSegment,
    normalize_transcript_segments,
    validate_video_input,
)


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
