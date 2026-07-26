import json
from dataclasses import replace
from pathlib import Path

import pytest

import bimer.asr_manifest as asr_manifest
from bimer.asr_manifest import replace_text_with_asr
from bimer.inference import TranscriptSegment
from bimer.manifest import read_manifest, write_manifest
from bimer.schema import UtteranceRecord


class Transcriber:
    def transcribe(self, video_path: Path, language: str):
        return language, [
            TranscriptSegment(0.0, 1.0, "ASR", 0.8),
            TranscriptSegment(1.0, 2.0, "text", 0.6),
        ]


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
    assert replaced[0].text_source == "whisper"
    assert replaced[0].asr_confidence == pytest.approx(0.7)


def _source(utterance_id: int, tmp_path: Path) -> UtteranceRecord:
    return UtteranceRecord(
        dataset="meld",
        split="test",
        dialogue_id="d1",
        utterance_id=utterance_id,
        text=f"human {utterance_id}",
        emotion="neutral",
        language="en",
        start_seconds=float(utterance_id),
        end_seconds=float(utterance_id + 1),
        video_path=tmp_path / f"{utterance_id}.mp4",
    )


def test_incremental_asr_manifest_resumes_verified_prefix(tmp_path):
    records = [_source(0, tmp_path), _source(1, tmp_path)]
    output = tmp_path / "asr.jsonl"
    write_manifest([replace(records[0], text="already done")], output)

    class CountingTranscriber:
        def __init__(self):
            self.calls = []

        def transcribe(self, video_path: Path, language: str):
            self.calls.append((video_path, language))
            return language, [TranscriptSegment(0.0, 1.0, "new text")]

    transcriber = CountingTranscriber()
    result = asr_manifest.write_asr_manifest_incrementally(records, transcriber, output)

    assert transcriber.calls == [(records[1].video_path, "en")]
    assert [record.text for record in result] == ["already done", "new text"]
    assert read_manifest(output) == result


def test_incremental_asr_manifest_persists_completed_rows_on_failure(tmp_path):
    records = [_source(0, tmp_path), _source(1, tmp_path)]
    output = tmp_path / "asr.jsonl"

    class FailingTranscriber:
        def transcribe(self, video_path: Path, language: str):
            if video_path == records[1].video_path:
                raise RuntimeError("broken media")
            return language, [TranscriptSegment(0.0, 1.0, "saved")]

    with pytest.raises(RuntimeError, match="broken media"):
        asr_manifest.write_asr_manifest_incrementally(records, FailingTranscriber(), output)

    saved = read_manifest(output)
    assert len(saved) == 1
    assert saved[0].sample_id == records[0].sample_id
    assert saved[0].text == "saved"


def test_incremental_asr_manifest_keeps_original_and_logs_media_failure(tmp_path):
    records = [_source(0, tmp_path), _source(1, tmp_path)]
    output = tmp_path / "asr.jsonl"
    errors = tmp_path / "asr-errors.jsonl"

    class FailingTranscriber:
        def transcribe(self, video_path: Path, language: str):
            if video_path == records[1].video_path:
                raise RuntimeError("broken media")
            return language, [TranscriptSegment(0.0, 1.0, "whisper text")]

    result = asr_manifest.write_asr_manifest_incrementally(
        records,
        FailingTranscriber(),
        output,
        keep_original_on_error=True,
        error_path=errors,
    )

    assert [record.text for record in result] == ["whisper text", "human 1"]
    assert read_manifest(output) == result
    error_rows = [json.loads(line) for line in errors.read_text().splitlines()]
    assert error_rows == [
        {
            "dataset": "meld",
            "error": "broken media",
            "error_type": "RuntimeError",
            "sample_id": records[1].sample_id,
            "split": "test",
        }
    ]
