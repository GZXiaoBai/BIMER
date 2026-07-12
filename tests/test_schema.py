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
    )
    result = AnalysisResult(language="zh", segments=(segment,))
    payload = result.to_dict()
    assert payload["language"] == "zh"
    assert payload["segments"][0]["emotion"] == "joy"
    assert payload["global_distribution"]["joy"] == pytest.approx(1.0)

