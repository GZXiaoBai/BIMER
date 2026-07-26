import numpy as np

from bimer.quality import audio_quality, text_quality, vision_quality


def test_text_quality_distinguishes_human_and_asr_sources():
    human = text_quality("这是人工文本", source="human")
    asr = text_quality("这是转写文本", source="whisper", asr_confidence=0.7)

    np.testing.assert_allclose(human[:2], [1.0, 1.0])
    np.testing.assert_allclose(asr[:2], [0.0, 0.7])
    assert np.all((human >= 0.0) & (human <= 1.0))


def test_audio_quality_marks_silence_as_no_voiced_audio():
    silence = audio_quality(np.zeros(16000, dtype=np.float32))
    speech_like = audio_quality(np.full(16000, 0.1, dtype=np.float32))

    assert silence[1] == 0.0
    assert silence[2] == 0.0
    assert speech_like[0] > 0.0
    assert speech_like[1] > silence[1]
    assert speech_like[2] == 1.0


def test_vision_quality_tracks_face_ratio_and_bbox_stability():
    detected = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.bool_)
    bboxes = [(0.2, 0.2, 0.4, 0.4)] * 4 + [None] * 4

    quality = vision_quality(detected, bboxes, expected_frames=8)

    assert quality[0] == 0.5
    assert quality[1] == 1.0
    assert quality[2] == 1.0
    assert quality[3] == 0.16
