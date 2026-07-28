from __future__ import annotations

import json
from pathlib import Path

import pytest

from bimer.v5_protocol import freeze_v5_selection
from scripts.run_v5_exploratory_test import run_v5_exploratory_test


def test_v5_exploratory_test_requires_formal_completion_and_is_single_use(
    tmp_path: Path,
) -> None:
    selection = freeze_v5_selection(
        tmp_path / "selection.json",
        selected_candidate="beta_005",
        candidate_config={"asr_consistency_weight": 0.05},
        evidence={"validation_only": True, "test_set_used": False},
    )
    formal = tmp_path / "FORMAL_COMPLETE"
    formal.touch()
    calls: list[dict[str, object]] = []

    def evaluator(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"test": {"meld": {}, "emotiontalk": {}}}))
        return output

    report = run_v5_exploratory_test(
        selection_path=selection,
        formal_complete_marker=formal,
        checkpoint_path=tmp_path / "best.pt",
        output_directory=tmp_path / "test",
        clean_manifest=tmp_path / "all.jsonl",
        clean_features=tmp_path / "features",
        conditions={"whisper": (tmp_path / "whisper.jsonl", tmp_path / "whisper-features")},
        evaluator=evaluator,
    )

    assert report.is_file()
    assert len(calls) == 2
    assert all(call["evaluation_role"] == "test" for call in calls)
    with pytest.raises(RuntimeError, match="already been evaluated"):
        run_v5_exploratory_test(
            selection_path=selection,
            formal_complete_marker=formal,
            checkpoint_path=tmp_path / "best.pt",
            output_directory=tmp_path / "test",
            clean_manifest=tmp_path / "all.jsonl",
            clean_features=tmp_path / "features",
            evaluator=evaluator,
        )


def test_v5_exploratory_test_rejects_missing_formal_marker(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="formal V5 training"):
        run_v5_exploratory_test(
            selection_path=tmp_path / "selection.json",
            formal_complete_marker=tmp_path / "missing",
            checkpoint_path=tmp_path / "best.pt",
            output_directory=tmp_path / "test",
            clean_manifest=tmp_path / "all.jsonl",
            clean_features=tmp_path / "features",
            evaluator=lambda **_kwargs: None,
        )
