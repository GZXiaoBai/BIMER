from pathlib import Path

from bimer.manifest import read_manifest, write_manifest
from bimer.schema import UtteranceRecord


def test_manifest_round_trip_preserves_paths_and_labels(tmp_path):
    records = [
        UtteranceRecord(
            dataset="emotiontalk",
            split="validation",
            dialogue_id="G01",
            utterance_id=2,
            text="你好",
            emotion="happy",
            language="zh",
            start_seconds=0.0,
            end_seconds=2.0,
            video_path=Path("video.mp4"),
        )
    ]
    output = write_manifest(records, tmp_path / "manifest.jsonl")
    loaded = read_manifest(output)
    assert loaded == records
