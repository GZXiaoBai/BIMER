from bimer.app import (
    analysis_rows,
    distribution_figure,
    modality_quality_figure,
    result_summary_markdown,
    timeline_figure,
    timeline_html,
    transcript_rows,
)
from bimer.inference import TranscriptSegment
from bimer.schema import AnalysisResult, AnalysisSegment


def _result():
    return AnalysisResult(
        language="en",
        segments=(
            AnalysisSegment(
                0.0,
                1.5,
                "Hello",
                "joy",
                {"neutral": 0.1, "joy": 0.9},
                {"text": 0.5, "audio": 0.3, "vision": 0.2},
            ),
            AnalysisSegment(
                1.5,
                3.0,
                "No",
                "anger",
                {"anger": 0.8, "neutral": 0.2},
                {"text": 0.4, "audio": 0.4, "vision": 0.2},
            ),
        ),
    )


def test_ui_helpers_create_editable_transcript_and_analysis_rows():
    rows = transcript_rows([TranscriptSegment(0.0, 1.0, "hello")])
    assert rows == [[0.0, 1.0, "hello"]]
    analysis = analysis_rows(_result())
    assert analysis[0][3] == "joy"
    assert analysis[0][4] == 0.9
    assert analysis[0][8:11] == [True, True, True]
    assert len(analysis[0]) == 16
    assert analysis[0][-1] == "confident"


def test_timeline_figure_marks_each_segment():
    figure = timeline_figure(_result())
    assert len(figure.axes) == 1
    assert len(figure.axes[0].patches) == 2


def test_distribution_and_quality_figures_are_renderable():
    assert len(distribution_figure(_result()).axes) == 1
    assert len(modality_quality_figure(_result()).axes) == 1


def test_timeline_html_can_seek_the_uploaded_video():
    html = timeline_html(_result())
    assert "currentTime=0.0" in html
    assert "currentTime=1.5" in html
    assert "dialogue-video" in html


def test_result_summary_surfaces_model_runtime_uncertainty_and_quality_warnings():
    result = AnalysisResult(
        language="zh",
        model_version="v2_quality_lagf",
        runtime_profile={
            "text": 1.2,
            "audio": 2.3,
            "vision": 3.4,
            "fusion": 0.5,
            "export": 0.1,
        },
        segments=(
            AnalysisSegment(
                0.0,
                1.0,
                "你好",
                "neutral",
                {"neutral": 0.4},
                {"text": 1.0},
                confidence_status="uncertain",
                quality_warnings=("vision_unavailable",),
            ),
        ),
    )

    summary = result_summary_markdown(result)

    assert "v2_quality_lagf" in summary
    assert "中文" in summary
    assert "不确定 1 句" in summary
    assert "vision_unavailable" in summary
    assert "总计 7.50s" in summary
