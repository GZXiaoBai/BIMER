from pathlib import Path

from bimer.asr_manifest import replace_text_with_asr
from bimer.inference import TranscriptSegment
from bimer.schema import UtteranceRecord


class Transcriber:
    def transcribe(self, video_path: Path, language: str):
        return language, [TranscriptSegment(0.0, 1.0, "ASR"), TranscriptSegment(1.0, 2.0, "text")]


def test_asr_manifest_preserves_labels_and_replaces_only_text(tmp_path):
    source = UtteranceRecord(
        dataset="meld",
        split="test",
        dialogue_id="d1",
        utterance_id=0,
        text="human text",
        emotion="joy",
        language="en",
        start_seconds=0.0,
        end_seconds=2.0,
        video_path=tmp_path / "clip.mp4",
    )
    replaced = replace_text_with_asr([source], Transcriber())
    assert replaced[0].text == "ASR text"
    assert replaced[0].emotion == source.emotion
    assert replaced[0].sample_id == source.sample_id
