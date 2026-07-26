from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v4_autodl.sh"


def test_v4_cloud_wrapper_enforces_budget_resume_packaging_and_shutdown():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "TOTAL_GPU_MAX_SECONDS" in text
    assert "72000" in text
    assert "GPU_SECONDS_USED" in text
    assert "timeout --signal=TERM" in text
    assert "run_v4_experiments.py" in text
    assert "run_v4_lora_fallback.py" in text
    assert "summarize_v4_formal.py" in text
    assert "run_v4_exploratory_test.py" in text
    assert "sha256sum" in text
    assert "DOWNLOAD_READY" in text
    assert "shutdown -h now" in text
    assert "--exclude=" in text
