from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import bimer.runtime as runtime_module
from bimer.inference import TranscriptSegment
from bimer.runtime import RuntimeSession, SegmentEdit

ROOT = Path(__file__).resolve().parents[1]


class FakeAnalyzer:
    def __init__(self) -> None:
        self.feature_pipeline = SimpleNamespace(cache=SimpleNamespace(clear=lambda: 3))
        self.analyze_calls: list[tuple[Path, str]] = []
        self.segment_calls: list[tuple[Path, str, tuple[TranscriptSegment, ...]]] = []

    def analyze(self, path: Path, language: str):
        self.analyze_calls.append((path, language))
        return SimpleNamespace(language="zh" if language == "auto" else language)

    def transcribe(self, path: Path, language: str):
        detected = "zh" if language == "auto" else language
        return detected, [TranscriptSegment(0.0, 1.0, "原文本", 0.8)]

    def analyze_segments(
        self,
        path: Path,
        *,
        detected_language: str,
        segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...],
    ):
        self.segment_calls.append((path, detected_language, tuple(segments)))
        return SimpleNamespace(language=detected_language)


def test_runtime_session_reanalyzes_last_transcription_with_edits() -> None:
    analyzer = FakeAnalyzer()
    session = RuntimeSession(analyzer)
    video = Path("dialogue.mp4")

    detected, _segments = session.transcribe(video, "auto")
    result = session.reanalyze([SegmentEdit(0.0, 1.0, "人工修改")])

    assert detected == "zh"
    assert result.language == "zh"
    path, language, segments = analyzer.segment_calls[-1]
    assert path == video
    assert language == "zh"
    assert segments == (TranscriptSegment(0.0, 1.0, "人工修改", None),)


def test_runtime_session_requires_context_for_reanalysis() -> None:
    session = RuntimeSession(FakeAnalyzer())

    with pytest.raises(RuntimeError, match="no previous video"):
        session.reanalyze([SegmentEdit(0.0, 1.0, "edit")])


def test_runtime_session_close_is_idempotent_and_blocks_future_work() -> None:
    session = RuntimeSession(FakeAnalyzer())

    session.close()
    session.close()

    with pytest.raises(RuntimeError, match="closed"):
        session.analyze(Path("dialogue.mp4"), "en")


def test_runtime_session_exposes_cache_clear_and_deployment_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace()
    expected = SimpleNamespace(ok=True)
    monkeypatch.setattr(runtime_module, "verify_deployment", lambda *_args, **_kwargs: expected)
    session = RuntimeSession(
        FakeAnalyzer(),
        manifest=manifest,
        artifact_root=Path("/tmp/artifacts"),
        offline=True,
    )

    assert session.clear_cache() == 3
    assert session.verify() is expected


def test_runtime_adapters_use_the_session_builder() -> None:
    cli = (ROOT / "src" / "bimer" / "cli_commands.py").read_text(encoding="utf-8")
    acceptance = (ROOT / "scripts" / "m2_acceptance.py").read_text(encoding="utf-8")

    assert "build_runtime_session(" in cli
    assert "build_runtime_session(" in acceptance
