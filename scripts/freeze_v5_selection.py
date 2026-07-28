#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimer.v5_protocol import freeze_v5_selection


def freeze_from_decision(decision_path: Path, output_path: Path) -> Path:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selected = decision.get("selected")
    if decision.get("decision") != "pass_v5" or not isinstance(selected, str):
        raise ValueError("V5 screen did not pass; selection cannot be frozen")
    if (
        decision.get("evidence_scope") != "validation_only"
        or decision.get("test_set_used") is not False
    ):
        raise ValueError("V5 screen evidence must be validation-only")
    candidate_configs = decision.get("candidate_configs")
    if not isinstance(candidate_configs, dict) or selected not in candidate_configs:
        raise ValueError("selected V5 candidate configuration is missing")
    return freeze_v5_selection(
        output_path,
        selected_candidate=selected,
        candidate_config=candidate_configs[selected],
        evidence={
            "validation_only": True,
            "test_set_used": False,
            "decision_report": str(decision_path.resolve()),
            "diagnostics": decision.get("diagnostics", {}).get(selected, {}),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(freeze_from_decision(args.decision, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
