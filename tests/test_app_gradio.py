from __future__ import annotations

from pathlib import Path

from bimer.app import create_app
from bimer.inference import TranscriptSegment
from bimer.labels import EMOTION_LABELS
from bimer.schema import AnalysisResult, AnalysisSegment


def _result(text: str = "人工修改") -> AnalysisResult:
    probabilities = {label: (1.0 if label == "neutral" else 0.0) for label in EMOTION_LABELS}
    return AnalysisResult(
        language="zh",
        segments=(
            AnalysisSegment(
                start_seconds=0.0,
                end_seconds=1.0,
                text=text,
                emotion="neutral",
                probabilities=probabilities,
                raw_probabilities=probabilities,
                modality_gates={"text": 0.6, "audio": 0.3, "vision": 0.1},
                modality_available={"text": True, "audio": True, "vision": False},
                modality_quality={
                    "text": {"source": 0.0},
                    "audio": {"energy": 0.7},
                    "vision": {"face_ratio": 0.0},
                },
                quality_warnings=("vision_unavailable",),
            ),
        ),
        model_version="v2_quality_lagf",
        runtime_profile={"fusion": 0.01},
    )


def test_real_gradio_app_wires_transcription_analysis_exports_and_cache(
    tmp_path: Path,
) -> None:
    class Cache:
        def clear(self) -> int:
            return 4

    class Analyzer:
        feature_pipeline = type("Pipeline", (), {"cache": Cache()})()

        def transcribe(self, _video: Path, _language: str):
            return "zh", [TranscriptSegment(0.0, 1.0, "原文本", 0.8)]

        def analyze_segments(
            self,
            _video: Path,
            *,
            detected_language: str,
            segments: list[TranscriptSegment],
        ) -> AnalysisResult:
            assert detected_language == "zh"
            return _result(segments[0].text)

    progress_events: list[tuple[float, str]] = []

    def progress(value: float, *, desc: str) -> None:
        progress_events.append((value, desc))

    app = create_app(Analyzer(), export_root=tmp_path)
    try:
        transcription = app.fns[0].fn
        analysis = app.fns[1].fn
        clear_cache = app.fns[2].fn

        rows, language = transcription("dialogue.mp4", "auto", progress)
        outputs = analysis(
            "dialogue.mp4",
            language,
            [[rows[0][0], rows[0][1], "人工修改"]],
            progress,
        )

        assert language == "zh"
        assert outputs[1][0][2] == "人工修改"
        assert Path(outputs[7]).is_file()
        assert Path(outputs[8]).is_file()
        assert Path(outputs[9]).is_file()
        assert clear_cache() == "已清除 4 个缓存文件"
        assert progress_events[-1] == (1.0, "分析完成")
    finally:
        app.close()
