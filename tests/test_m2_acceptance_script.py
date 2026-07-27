import importlib.util
import subprocess
import sys
from pathlib import Path

from bimer.schema import AnalysisResult, AnalysisSegment

ROOT = Path(__file__).resolve().parents[1]


def test_m2_acceptance_uses_the_frozen_deployment_manifest_interface():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "m2_acceptance.py"), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--deployment" in result.stdout
    assert "--artifact-root" in result.stdout
    assert "--allow-partial" in result.stdout
    assert "--preserve-runtime-cache" in result.stdout
    assert "--checkpoint" not in result.stdout


def test_m2_acceptance_runner_enforces_time_memory_swap_and_exports():
    python_text = (ROOT / "scripts" / "m2_acceptance.py").read_text(encoding="utf-8")
    shell_text = (ROOT / "scripts" / "run_m2_acceptance.sh").read_text(encoding="utf-8")

    assert "<= 120" in python_text
    assert "<= 15" in python_text
    assert "chinese_content_is_chinese" in python_text
    assert "chinese_face_enables_vision" in python_text
    assert "english_no_face_disables_vision" in python_text
    assert "export_analysis_json" in python_text
    assert "export_analysis_csv" in python_text
    assert "export_analysis_figure" in python_text
    assert "wrong_format_error" in python_text
    assert "oversized_file_error" in python_text
    assert "silent_video_error" in python_text
    assert '"complete": complete' in python_text
    assert "authorized 30-60 second Chinese face video" in python_text
    assert "/usr/bin/time -l" in shell_text
    assert "6979321856" in shell_text
    assert "swap_before" in shell_text and "swap_after" in shell_text
    assert "process_swaps" in shell_text
    assert "peak memory footprint" in shell_text
    assert 'PYTHON="${BIMER_PYTHON:-$ROOT/.venv/bin/python}"' in shell_text


def test_m2_text_edit_reuses_timestamps_from_first_analysis():
    script = ROOT / "scripts" / "m2_acceptance.py"
    specification = importlib.util.spec_from_file_location("m2_acceptance", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    result = AnalysisResult(
        language="zh",
        segments=(
            AnalysisSegment(
                start_seconds=1.25,
                end_seconds=4.75,
                text="原始文本",
                emotion="neutral",
                probabilities={"neutral": 1.0},
                modality_gates={"text": 1.0, "audio": 0.0, "vision": 0.0},
            ),
        ),
    )

    segments = module.transcript_segments_from_result(result)

    assert [(segment.start_seconds, segment.end_seconds, segment.text) for segment in segments] == [
        (1.25, 4.75, "原始文本")
    ]


def test_m2_chinese_content_ratio_rejects_english_only_sample():
    script = ROOT / "scripts" / "m2_acceptance.py"
    specification = importlib.util.spec_from_file_location("m2_acceptance", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    chinese = AnalysisResult(
        language="zh",
        segments=(
            AnalysisSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="这是中文对话样例",
                emotion="neutral",
                probabilities={"neutral": 1.0},
                modality_gates={"text": 1.0, "audio": 0.0, "vision": 0.0},
            ),
        ),
    )
    english = AnalysisResult(
        language="zh",
        segments=(
            AnalysisSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="This is an English interview.",
                emotion="neutral",
                probabilities={"neutral": 1.0},
                modality_gates={"text": 1.0, "audio": 0.0, "vision": 0.0},
            ),
        ),
    )

    assert module.chinese_character_ratio(chinese) == 1.0
    assert module.chinese_character_ratio(english) == 0.0


def test_m2_chinese_face_requires_vision_for_every_segment():
    script = ROOT / "scripts" / "m2_acceptance.py"
    specification = importlib.util.spec_from_file_location("m2_acceptance", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    result = AnalysisResult(
        language="zh",
        segments=(
            AnalysisSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="第一句",
                emotion="neutral",
                probabilities={"neutral": 1.0},
                modality_gates={"text": 0.5, "audio": 0.2, "vision": 0.3},
                modality_available={"text": True, "audio": True, "vision": True},
            ),
            AnalysisSegment(
                start_seconds=2.0,
                end_seconds=4.0,
                text="第二句",
                emotion="neutral",
                probabilities={"neutral": 1.0},
                modality_gates={"text": 0.5, "audio": 0.5, "vision": 0.0},
                modality_available={"text": True, "audio": True, "vision": False},
            ),
        ),
    )

    assert not module.all_segments_have_vision(result)
