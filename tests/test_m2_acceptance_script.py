from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m2_acceptance_runner_enforces_time_memory_swap_and_exports():
    python_text = (ROOT / "scripts" / "m2_acceptance.py").read_text(
        encoding="utf-8"
    )
    shell_text = (ROOT / "scripts" / "run_m2_acceptance.sh").read_text(
        encoding="utf-8"
    )

    assert "<= 120" in python_text
    assert "<= 15" in python_text
    assert "english_no_face_disables_vision" in python_text
    assert "export_analysis_json" in python_text
    assert "export_analysis_csv" in python_text
    assert "export_analysis_figure" in python_text
    assert "wrong_format_error" in python_text
    assert "oversized_file_error" in python_text
    assert "silent_video_error" in python_text
    assert "/usr/bin/time -l" in shell_text
    assert "6979321856" in shell_text
    assert "swap_before" in shell_text and "swap_after" in shell_text
