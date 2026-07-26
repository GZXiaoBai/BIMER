import numpy as np

from bimer.schema import UtteranceRecord
from bimer.windowing import make_context_windows, merge_window_probabilities


def _records(count: int):
    return [
        UtteranceRecord(
            dataset="meld",
            split="train",
            dialogue_id="d1",
            utterance_id=index,
            text=f"line {index}",
            emotion="neutral",
            language="en",
            start_seconds=float(index),
            end_seconds=float(index + 1),
        )
        for index in range(count)
    ]


def test_context_windows_use_fixed_overlap_without_dropping_utterances():
    windows = make_context_windows(_records(70), max_length=32, overlap=8)
    assert [len(window.records) for window in windows] == [32, 32, 22]
    assert windows[1].records[0].utterance_id == 24
    assert windows[2].records[0].utterance_id == 48
    assert windows[-1].records[-1].utterance_id == 69


def test_context_windows_merge_emotiontalk_speaker_tracks_by_context_id():
    records = [
        UtteranceRecord(
            dataset="emotiontalk",
            split="train",
            dialogue_id=f"G00006_58_{speaker}",
            context_id="G00006_58",
            utterance_id=utterance_id,
            text=f"line {utterance_id}",
            emotion="neutral",
            language="zh",
            start_seconds=0.0,
            end_seconds=1.0,
        )
        for utterance_id, speaker in ((1, "12"), (2, "07"), (3, "12"))
    ]

    windows = make_context_windows(records, max_length=32, overlap=8)

    assert len(windows) == 1
    assert [record.utterance_id for record in windows[0].records] == [1, 2, 3]


def test_merges_overlapping_probabilities_by_mean():
    windows = make_context_windows(_records(3), max_length=2, overlap=1)
    predictions = [
        np.array([[1.0, 0.0], [0.8, 0.2]]),
        np.array([[0.4, 0.6], [0.0, 1.0]]),
    ]
    merged = merge_window_probabilities(windows, predictions)
    assert np.allclose(merged["meld:train:d1:1"], [0.6, 0.4])
    assert np.allclose(merged["meld:train:d1:2"], [0.0, 1.0])
