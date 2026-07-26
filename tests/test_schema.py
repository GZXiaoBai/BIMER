from pathlib import Path

import pytest

from bimer.schema import AnalysisResult, AnalysisSegment, UtteranceRecord


def test_utterance_record_has_stable_sample_id_and_validated_times():
    record = UtteranceRecord(
        dataset="meld",
        split="train",
        dialogue_id="6",
        utterance_id=1,
        text="You liked it?",
        emotion="joy",
        language="en",
        start_seconds=1.5,
        end_seconds=3.0,
        video_path=Path("dia6_utt1.mp4"),
    )
    assert record.sample_id == "meld:train:6:1"


def test_context_id_can_differ_without_changing_sample_identity():
    record = UtteranceRecord(
        dataset="emotiontalk",
        split="train",
        dialogue_id="G00006_58_07",
        context_id="G00006_58",
        utterance_id=6,
        text="哇，真的吗？",
        emotion="surprise",
        language="zh",
        start_seconds=0.0,
        end_seconds=1.3,
    )

    assert record.context_id == "G00006_58"
    assert record.effective_context_id == "G00006_58"
    assert record.sample_id == "emotiontalk:train:G00006_58_07:6"


def test_official_emotiontalk_context_id_is_inferred_for_legacy_manifests():
    record = UtteranceRecord(
        dataset="emotiontalk",
        split="train",
        dialogue_id="G00006_58_12",
        utterance_id=1,
        text="开场",
        emotion="neutral",
        language="zh",
        start_seconds=0.0,
        end_seconds=1.0,
    )

    assert record.context_id == "G00006_58"


def test_utterance_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="end_seconds"):
        UtteranceRecord(
            dataset="meld",
            split="train",
            dialogue_id="6",
            utterance_id=1,
            text="bad",
            emotion="neutral",
            language="en",
            start_seconds=2.0,
            end_seconds=2.0,
        )


def test_analysis_result_serializes_probabilities_and_gates():
    segment = AnalysisSegment(
        start_seconds=0.0,
        end_seconds=2.0,
        text="你好",
        emotion="joy",
        probabilities={"neutral": 0.1, "joy": 0.9},
        modality_gates={"text": 0.5, "audio": 0.3, "vision": 0.2},
        modality_available={"text": True, "audio": True, "vision": False},
        modality_quality={"vision": {"face_ratio": 0.1}},
        quality_warnings=("vision_unavailable",),
    )
    result = AnalysisResult(language="zh", segments=(segment,))
    payload = result.to_dict()
    assert payload["language"] == "zh"
    assert payload["segments"][0]["emotion"] == "joy"
    assert payload["global_distribution"]["joy"] == pytest.approx(0.9)
    assert payload["label_distribution"]["joy"] == pytest.approx(1.0)
    assert payload["segments"][0]["modality_available"]["vision"] is False


def test_analysis_result_exposes_timestamped_transition_events():
    first = AnalysisSegment(0.0, 1.0, "hi", "neutral", {"neutral": 0.8}, {"text": 1.0})
    second = AnalysisSegment(1.0, 2.0, "great", "joy", {"joy": 0.9}, {"text": 1.0})

    event = AnalysisResult(language="en", segments=(first, second)).to_dict()["transition_events"][
        0
    ]

    assert event == {
        "segment_index": 1,
        "time_seconds": 1.0,
        "from_emotion": "neutral",
        "to_emotion": "joy",
        "confidence": 0.9,
    }
