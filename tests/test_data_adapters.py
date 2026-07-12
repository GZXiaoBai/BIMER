import json

import pandas as pd
import pytest

from bimer.data_adapters import (
    EXPECTED_SPLIT_COUNTS,
    check_official_split_counts,
    load_emotiontalk_manifest,
    load_meld_csv,
    parse_timestamp,
)


def test_parse_timestamp_supports_meld_comma_milliseconds():
    assert parse_timestamp("00:01:02,500") == pytest.approx(62.5)


def test_loads_meld_csv_into_unified_records(tmp_path):
    csv_path = tmp_path / "train_sent_emo.csv"
    pd.DataFrame(
        [
            {
                "Utterance": "You liked it?",
                "Speaker": "Monica",
                "Emotion": "joy",
                "Dialogue_ID": 6,
                "Utterance_ID": 1,
                "StartTime": "00:00:01,000",
                "EndTime": "00:00:03,500",
            }
        ]
    ).to_csv(csv_path, index=False)
    records = load_meld_csv(csv_path, media_root=tmp_path / "videos", split="train")
    assert records[0].sample_id == "meld:train:6:1"
    assert records[0].speaker_id == "Monica"
    assert records[0].video_path.name == "dia6_utt1.mp4"
    assert records[0].end_seconds == pytest.approx(3.5)


def test_loads_emotiontalk_json_manifest_and_maps_labels(tmp_path):
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "dialogue_id": "G01_001",
                    "utterance_id": 4,
                    "content": "太好了",
                    "emotion_result": "Happy",
                    "speaker_id": "G01",
                    "file_name": "G01_001_004.mp4",
                    "start_seconds": 2.0,
                    "end_seconds": 4.5,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    records = load_emotiontalk_manifest(
        manifest, media_root=tmp_path / "videos", split="train"
    )
    assert records[0].emotion == "joy"
    assert records[0].language == "zh"
    assert records[0].video_path.name == "G01_001_004.mp4"


def test_official_count_check_reports_exact_mismatches():
    assert EXPECTED_SPLIT_COUNTS["meld"]["train"] == 9989
    with pytest.raises(ValueError, match="expected 9989, found 1"):
        check_official_split_counts("meld", {"train": 1})

