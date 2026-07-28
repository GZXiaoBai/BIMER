from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_domain_context_defines_research_and_runtime_terms() -> None:
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")

    for term in (
        "sample_id",
        "context_id",
        "模态质量",
        "确认性实验",
        "探索性实验",
        "RuntimeSession",
    ):
        assert term in context


def test_architecture_decisions_and_script_support_index_are_present() -> None:
    decisions = sorted((ROOT / "docs" / "adr").glob("*.md"))
    scripts = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert len(decisions) >= 3
    for decision in decisions:
        text = decision.read_text(encoding="utf-8")
        assert "Status: Accepted" in text
        assert "## Decision" in text
        assert "## Consequences" in text
    assert "正式支持" in scripts
    assert "研究归档" in scripts
    assert "云端历史" in scripts
