import json

from bimer.export import export_analysis_csv, export_analysis_figure, export_analysis_json
from bimer.schema import AnalysisResult, AnalysisSegment


def _result():
    return AnalysisResult(
        language="en",
        segments=(
            AnalysisSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="Great",
                emotion="joy",
                probabilities={"joy": 0.8, "neutral": 0.2},
                modality_gates={"text": 0.5, "audio": 0.3, "vision": 0.2},
            ),
        ),
    )


def test_exports_json_and_flat_csv(tmp_path):
    json_path = export_analysis_json(_result(), tmp_path / "result.json")
    csv_path = export_analysis_csv(_result(), tmp_path / "result.csv")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["segments"][0]["emotion"] == "joy"
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "gate_text" in csv_text
    assert "available_vision" in csv_text
    assert "quality_vision_face_ratio" in csv_text
    assert "confidence_status" in csv_text
    assert "raw_probability_joy" in csv_text
    assert payload["model_version"] == "v2"
    assert "runtime_profile" in payload
    assert "Great" in csv_text
    figure_path = export_analysis_figure(_result(), tmp_path / "result.png")
    assert figure_path.read_bytes().startswith(b"\x89PNG")
