from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _lock_module():
    script = Path(__file__).parents[1] / "scripts" / "lock_external_video_plan.py"
    spec = importlib.util.spec_from_file_location("lock_external_video_plan", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_media_requires_an_audio_stream(monkeypatch: pytest.MonkeyPatch):
    module = _lock_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout='{"format":{"duration":"35.0"},"streams":[{"codec_type":"video"}]}'
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="audio"):
        module._probe_media(Path("silent.mp4"))


def test_probe_media_returns_duration_when_audio_exists(monkeypatch: pytest.MonkeyPatch):
    module = _lock_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=(
                '{"format":{"duration":"35.0"},"streams":'
                '[{"codec_type":"video"},{"codec_type":"audio"}]}'
            )
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert module._probe_media(Path("valid.mp4")) == 35.0
