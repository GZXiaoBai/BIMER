from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bimer.experiment_protocol import (
    ExperimentProtocolRunner,
    ProtocolSpec,
    run_guarded_exploratory_test,
)


def _config(**changes):
    payload = {
        "protocol_stage": "standard",
        "seed": 42,
        "evaluate_test": True,
    }
    payload.update(changes)
    return SimpleNamespace(**payload)


def test_protocol_spec_enforces_screen_and_formal_test_boundaries() -> None:
    ProtocolSpec.from_config(_config(protocol_stage="standard"))
    ProtocolSpec.from_config(_config(protocol_stage="v3_formal", seed=123, evaluate_test=False))

    with pytest.raises(ValueError, match="restricted to seed 42"):
        ProtocolSpec.from_config(_config(protocol_stage="v5_screen", seed=123, evaluate_test=False))
    with pytest.raises(ValueError, match="must use --skip-test"):
        ProtocolSpec.from_config(_config(protocol_stage="v5_screen", seed=42, evaluate_test=True))
    with pytest.raises(ValueError, match="must use --skip-test"):
        ProtocolSpec.from_config(_config(protocol_stage="v5_formal", seed=42, evaluate_test=True))


def test_protocol_runner_writes_success_and_resumes_existing_result(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    status = tmp_path / "status.json"
    calls: list[str] = []
    runner = ExperimentProtocolRunner(
        ProtocolSpec.from_config(_config()),
        status_path=status,
        result_path=result,
    )

    first = runner.run(lambda: calls.append("run") or result)
    result.write_text("{}\n", encoding="utf-8")
    second = runner.run(lambda: calls.append("rerun") or result)

    assert first == result
    assert second == result
    assert calls == ["run"]
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["protocol_stage"] == "standard"


def test_protocol_runner_records_failure_before_reraising(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    runner = ExperimentProtocolRunner(
        ProtocolSpec.from_config(_config(protocol_stage="v4_formal", evaluate_test=False)),
        status_path=status,
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"


def test_generic_exploratory_test_guard_allows_only_one_frozen_run(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"state": "frozen", "version": "v5"}),
        encoding="utf-8",
    )
    marker = tmp_path / "TEST_EVALUATED"

    assert (
        run_guarded_exploratory_test(
            version="v5",
            selection_path=selection,
            marker_path=marker,
            evaluator=lambda: "result",
        )
        == "result"
    )
    with pytest.raises(RuntimeError, match="already been evaluated"):
        run_guarded_exploratory_test(
            version="v5",
            selection_path=selection,
            marker_path=marker,
            evaluator=lambda: "again",
        )
