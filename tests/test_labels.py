import pytest

from bimer.labels import EMOTION_LABELS, emotion_index, normalize_emotion


def test_normalizes_meld_and_emotiontalk_labels_to_one_schema():
    assert normalize_emotion("Joy", dataset="meld") == "joy"
    assert normalize_emotion("happy", dataset="emotiontalk") == "joy"
    assert normalize_emotion("Disgusted", dataset="emotiontalk") == "disgust"
    assert normalize_emotion(" sadness ", dataset="meld") == "sadness"


def test_emotion_indices_follow_the_public_label_order():
    assert EMOTION_LABELS == (
        "neutral",
        "joy",
        "sadness",
        "anger",
        "surprise",
        "fear",
        "disgust",
    )
    assert emotion_index("fear") == 5


def test_rejects_unknown_dataset_label():
    with pytest.raises(ValueError, match="Unknown emotion label"):
        normalize_emotion("confused", dataset="meld")
