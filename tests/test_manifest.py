from pathlib import Path

from bimer.manifest import read_manifest, write_manifest
from bimer.schema import UtteranceRecord


def test_manifest_round_trip_preserves_paths_and_labels(tmp_path):
    records = [
        UtteranceRecord(
            dataset="emotiontalk",
            split="validation",
            dialogue_id="G01",
            context_id="G01_scene",
            utterance_id=2,
            text="你好",
            emotion="happy",
            language="zh",
            start_seconds=0.0,
            end_seconds=2.0,
            video_path=Path("video.mp4"),
            text_source="whisper",
            asr_confidence=0.75,
        )
    ]
    output = write_manifest(records, tmp_path / "manifest.jsonl")
    loaded = read_manifest(output)
    assert loaded == records


def test_manifest_append_keeps_existing_rows(tmp_path):
    first = UtteranceRecord(
        dataset="meld",
        split="test",
        dialogue_id="d1",
        utterance_id=0,
        text="first",
        emotion="neutral",
        language="en",
        start_seconds=0.0,
        end_seconds=1.0,
    )
    second = UtteranceRecord(
        dataset="meld",
        split="test",
        dialogue_id="d1",
        utterance_id=1,
        text="second",
        emotion="joy",
        language="en",
        start_seconds=1.0,
        end_seconds=2.0,
    )
    path = write_manifest([first], tmp_path / "manifest.jsonl")

    write_manifest([second], path, append=True)

    assert read_manifest(path) == [first, second]
