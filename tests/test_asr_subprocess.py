from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import bimer.asr_subprocess as asr_subprocess
from bimer.asr_subprocess import ASRWorkerError, SubprocessWhisperTranscriber


def test_subprocess_transcriber_uses_json_protocol_without_importing_whisper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "sample video.mp4"
    video.write_bytes(b"video")
    payload = {
        "language": "zh",
        "segments": [
            {
                "start_seconds": 0.0,
                "end_seconds": 1.5,
                "text": "你好",
                "asr_confidence": 0.8,
            }
        ],
    }
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(asr_subprocess.subprocess, "run", fake_run)
    transcriber = SubprocessWhisperTranscriber(
        "/private/models/whisper",
        device="cpu",
        python_executable="/private/python",
        timeout_seconds=30,
    )

    language, segments = transcriber.transcribe(video, "auto")

    assert language == "zh"
    assert segments[0].text == "你好"
    assert segments[0].asr_confidence == pytest.approx(0.8)
    command, kwargs = calls[0]
    assert command[:3] == ["/private/python", "-m", "bimer.asr_worker"]
    assert command[-2:] == ["--language", "auto"]
    assert kwargs["timeout"] == 30
    assert kwargs["check"] is False


def test_subprocess_transcriber_surfaces_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        asr_subprocess.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="model could not be loaded",
        ),
    )

    with pytest.raises(ASRWorkerError, match="model could not be loaded"):
        SubprocessWhisperTranscriber("small").transcribe(
            tmp_path / "sample.mp4",
            "en",
        )


@pytest.mark.parametrize(
    "stdout",
    [
        "not-json",
        '{"language":"fr","segments":[]}',
        '{"language":"en","segments":[{"start_seconds":2,"end_seconds":1,"text":"bad"}]}',
    ],
)
def test_subprocess_transcriber_rejects_invalid_worker_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
) -> None:
    monkeypatch.setattr(
        asr_subprocess.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=stdout,
            stderr="",
        ),
    )

    with pytest.raises(ASRWorkerError, match="invalid"):
        SubprocessWhisperTranscriber("small").transcribe(
            tmp_path / "sample.mp4",
            "auto",
        )


def test_subprocess_transcriber_converts_timeout_to_domain_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["python"], 3)

    monkeypatch.setattr(asr_subprocess.subprocess, "run", timeout)

    with pytest.raises(ASRWorkerError, match="timed out"):
        SubprocessWhisperTranscriber("small", timeout_seconds=3).transcribe(
            tmp_path / "sample.mp4",
            "auto",
        )
