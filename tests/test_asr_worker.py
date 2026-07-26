from __future__ import annotations

import json
from pathlib import Path

import bimer.asr_worker as asr_worker


def test_worker_main_emits_machine_readable_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        asr_worker,
        "run_worker",
        lambda **_kwargs: {
            "language": "en",
            "segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "text": "hello",
                    "asr_confidence": 0.75,
                },
            ],
        },
    )

    exit_code = asr_worker.main(
        [
            "--model",
            "small",
            "--video",
            str(video),
            "--language",
            "auto",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["segments"][0]["text"] == "hello"


def test_worker_main_converts_model_error_to_nonzero_exit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def fail(**_kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(asr_worker, "run_worker", fail)

    exit_code = asr_worker.main(
        [
            "--model",
            "small",
            "--video",
            str(tmp_path / "sample.mp4"),
        ]
    )

    assert exit_code == 2
    assert "unavailable" in capsys.readouterr().err
