from bimer.schema import UtteranceRecord
from bimer.validation import validate_dataset_records


def _record(split: str, utterance_id: int, video_path: str):
    return UtteranceRecord(
        dataset="meld",
        split=split,
        dialogue_id="d1",
        utterance_id=utterance_id,
        text="line",
        emotion="neutral",
        language="en",
        start_seconds=0.0,
        end_seconds=1.0,
        video_path=video_path,
    )


def test_validation_reports_duplicate_media_across_splits():
    report = validate_dataset_records(
        [_record("train", 0, "same.mp4"), _record("test", 1, "same.mp4")]
    )
    assert report.is_valid is False
    assert report.cross_split_media == ("same.mp4",)


def test_validation_accepts_unique_records_and_counts_labels():
    report = validate_dataset_records(
        [_record("train", 0, "a.mp4"), _record("test", 1, "b.mp4")]
    )
    assert report.is_valid is True
    assert report.split_counts == {"train": 1, "test": 1}
    assert report.label_counts["neutral"] == 2
