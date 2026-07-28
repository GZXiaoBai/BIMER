import json

import pandas as pd
import pytest

import bimer.data_adapters as data_adapters
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


def test_loads_meld_csv_repairs_equal_timestamps_from_media_duration(tmp_path):
    csv_path = tmp_path / "test_sent_emo.csv"
    media_root = tmp_path / "videos"
    media_root.mkdir()
    media_path = media_root / "dia155_utt3.mp4"
    media_path.touch()
    pd.DataFrame(
        [
            {
                "Utterance": "Oh my God.",
                "Speaker": "Phoebe",
                "Emotion": "surprise",
                "Dialogue_ID": 155,
                "Utterance_ID": 3,
                "StartTime": "0:12:32,632",
                "EndTime": "0:12:32,632",
            }
        ]
    ).to_csv(csv_path, index=False)

    records = load_meld_csv(
        csv_path,
        media_root=media_root,
        split="test",
        duration_probe=lambda path: 1.75 if path == media_path else 0.0,
    )

    assert records[0].start_seconds == pytest.approx(752.632)
    assert records[0].end_seconds == pytest.approx(754.382)


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
    records = load_emotiontalk_manifest(manifest, media_root=tmp_path / "videos", split="train")
    assert records[0].emotion == "joy"
    assert records[0].language == "zh"
    assert records[0].video_path.name == "G01_001_004.mp4"


def test_official_count_check_reports_exact_mismatches():
    assert EXPECTED_SPLIT_COUNTS["meld"]["train"] == 9989
    with pytest.raises(ValueError, match="expected 9989, found 1"):
        check_official_split_counts("meld", {"train": 1})


def test_loads_emotiontalk_official_csv_with_published_group_splits(tmp_path):
    labels = tmp_path / "mm.csv"
    transcripts = tmp_path / "transcription.csv"
    relative_video = "G00001/G00001_01/G00001_01_07/G00001_01_07_003.mp4"
    labels.write_text(
        f"file_name,emotion\n{relative_video},happy\n",
        encoding="utf-8",
    )
    transcripts.write_text(
        "name,emotion,chinese\n"
        "G00001/G00001_01/G00001_01_07/G00001_01_07_003,neutral,"
        "[over/]这是[/over]重叠。\n",
        encoding="utf-8",
    )
    video_path = tmp_path / "extracted" / "Multimodal" / relative_video
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    records = data_adapters.load_emotiontalk_official_csv(
        labels,
        transcripts,
        media_root=tmp_path / "extracted",
        duration_probe=lambda _: 2.5,
        duration_workers=2,
    )

    assert len(records) == 1
    assert records[0].split == "validation"
    assert records[0].dialogue_id == "G00001_01_07"
    assert records[0].context_id == "G00001_01"
    assert records[0].speaker_id == "07"
    assert records[0].utterance_id == 3
    assert records[0].text == "这是重叠。"
    assert records[0].emotion == "joy"
    assert records[0].video_path == video_path
    assert records[0].end_seconds == pytest.approx(2.5)


def test_emotiontalk_official_csv_rejects_missing_transcription(tmp_path):
    labels = tmp_path / "mm.csv"
    transcripts = tmp_path / "transcription.csv"
    labels.write_text(
        "file_name,emotion\nG00003/G00003_01_09/G00003_01_09_001.mp4,neutral\n",
        encoding="utf-8",
    )
    transcripts.write_text("name,emotion,chinese\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing transcriptions"):
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_supports_jsonl_wrappers_and_duration_variants(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "one.mp4").touch()
    manifest = tmp_path / "items.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "file_name": "one.mp4",
                        "content": "文本",
                        "emotion_result": {"happy": 0.8, "sad": 0.2},
                        "duration_seconds": 1.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records = load_emotiontalk_manifest(manifest, media_root=media, split="train")
    assert records[0].emotion == "joy"
    assert records[0].end_seconds == pytest.approx(1.5)

    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "file_name": "one.mp4",
                "content": "文本",
                "emotion_result": ["sad", "sad", "happy"],
                "paragraphs": {"start": "0:00:01", "end": "0:00:03"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_emotiontalk_manifest(jsonl, media_root=media, split="train")
    assert records[0].emotion == "sadness"
    assert (records[0].start_seconds, records[0].end_seconds) == (1.0, 3.0)


def test_manifest_rejects_unusable_payloads_and_missing_duration(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert load_emotiontalk_manifest(empty, media_root=tmp_path, split="train") == []

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"unknown": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="list of records"):
        load_emotiontalk_manifest(wrapped, media_root=tmp_path, split="train")

    invalid_emotion = tmp_path / "emotion.json"
    invalid_emotion.write_text(
        json.dumps(
            [
                {
                    "file_name": "missing.mp4",
                    "emotion_result": {},
                    "duration_seconds": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="derive emotion"):
        load_emotiontalk_manifest(
            invalid_emotion,
            media_root=tmp_path,
            split="train",
        )

    missing_duration = tmp_path / "duration.json"
    missing_duration.write_text(
        json.dumps([{"file_name": "missing.mp4", "emotion_result": "neutral"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing duration"):
        load_emotiontalk_manifest(
            missing_duration,
            media_root=tmp_path,
            split="train",
        )


def test_meld_and_official_emotiontalk_validate_malformed_inputs(tmp_path):
    meld = tmp_path / "meld.csv"
    pd.DataFrame([{"Utterance": "missing fields"}]).to_csv(meld, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_meld_csv(meld, media_root=tmp_path, split="train")

    labels = tmp_path / "labels.csv"
    transcripts = tmp_path / "transcripts.csv"
    labels.write_text("bad,emotion\nx,neutral\n", encoding="utf-8")
    transcripts.write_text("name,chinese\nx,文本\n", encoding="utf-8")
    with pytest.raises(ValueError, match="labels CSV missing"):
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
        )

    labels.write_text("file_name,emotion\nx.mp4,neutral\n", encoding="utf-8")
    transcripts.write_text("bad,chinese\nx,文本\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transcriptions CSV missing"):
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
        )


def test_official_emotiontalk_rejects_duplicates_workers_and_bad_media(tmp_path):
    labels = tmp_path / "labels.csv"
    transcripts = tmp_path / "transcripts.csv"
    name = "G00001/G00001_01/G00001_01_07/G00001_01_07_003"
    labels.write_text(f"file_name,emotion\n{name}.mp4,happy\n", encoding="utf-8")
    transcripts.write_text(
        f"name,chinese\n{name},文本\n{name},重复\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate transcriptions"):
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
        )

    transcripts.write_text(f"name,chinese\n{name},文本\n", encoding="utf-8")
    labels.write_text(
        f"file_name,emotion\n{name}.mp4,happy\n{name}.mp4,sad\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate label"):
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
        )

    labels.write_text("file_name,emotion\n", encoding="utf-8")
    assert (
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
        )
        == []
    )

    labels.write_text(f"file_name,emotion\n{name}.mp4,happy\n", encoding="utf-8")
    video = tmp_path / f"{name}.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    with pytest.raises(ValueError, match="duration_workers"):
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
            duration_workers=0,
        )

    video.unlink()
    with pytest.raises(ValueError, match="locate EmotionTalk media"):
        data_adapters.load_emotiontalk_official_csv(
            labels,
            transcripts,
            media_root=tmp_path,
        )


def test_unknown_dataset_count_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        check_official_split_counts("unknown", {})
