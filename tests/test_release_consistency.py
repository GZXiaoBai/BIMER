from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.1.0"


def test_public_release_metadata_and_commands_are_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")

    assert pyproject["project"]["version"] == EXPECTED_VERSION
    assert f"version: {EXPECTED_VERSION}" in citation
    assert "python3.11 -m pip install uv" in readme
    assert "uv sync --extra dev --extra inference --frozen" in makefile
    assert "uv run python -m pytest" in makefile
    assert "V4 is not deployed" in model_card


def test_v1_release_evidence_manifest_freezes_the_published_baseline() -> None:
    path = ROOT / "docs" / "releases" / "v1.0.0-evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["schema_version"] == 1
    assert evidence["release"]["tag"] == "v1.0.0"
    assert (
        evidence["release"]["commit"]
        == "57f5a86a4638f6c4b5c8c9a3458f85432e137e76"
    )
    assert len(evidence["release"]["git_archive_sha256"]) == 64
    assert (
        evidence["deployment"]["checkpoint_sha256"]
        == "41ebf8aa84b16fb01f17ea6bc3d26f5f14201c55d9692ba3210b241c8c9464bb"
    )
    assert evidence["research"]["formal_report_sha256"]
    assert evidence["research"]["robustness_report_sha256"]
