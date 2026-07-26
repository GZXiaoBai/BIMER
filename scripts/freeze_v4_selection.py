#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimer.v4_protocol import freeze_v4_selection


def freeze_from_decision(decision_path: Path, output_path: Path) -> Path:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") not in {"pass_v4a", "pass_lora"} or not decision.get("selected"):
        raise ValueError(
            "V4 selection did not pass: neither V4-A nor the conditional LoRA fallback"
        )
    selected = str(decision["selected"])
    candidate_configs = decision.get("candidate_configs", {})
    if selected not in candidate_configs:
        raise ValueError("selected V4 candidate has no frozen configuration")
    return freeze_v4_selection(
        output_path,
        selected_candidate=selected,
        candidate_config=candidate_configs[selected],
        evidence={
            "validation_only": True,
            "test_set_used": False,
            "screen_decision": decision,
            "selection_route": decision["decision"],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = freeze_from_decision(Path(args.decision), Path(args.output))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
