from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def select_deployment_model(
    *,
    frozen_selection: Mapping[str, object],
    external_report: Mapping[str, object],
    m2_report: Mapping[str, object],
) -> dict[str, object]:
    validation_passed = (
        frozen_selection.get("state") == "frozen"
        and frozen_selection.get("version") == "v3"
        and float(frozen_selection.get("gate_ranking_weight", 0.0)) > 0
    )
    external_passed = bool(
        external_report.get("v3_acceptance", {}).get("accepted", False)
    )
    m2_passed = bool(m2_report.get("passed", False))
    checks = {
        "validation_ranking_passed": validation_passed,
        "external_acceptance_passed": external_passed,
        "m2_acceptance_passed": m2_passed,
    }
    use_v3 = all(checks.values())
    return {
        "state": "frozen",
        "deployed_model": "v3_ranked" if use_v3 else "v2_quality_lagf",
        "checks": checks,
        "fallback_used": not use_v3,
    }


def write_deployment_selection(
    selection: Mapping[str, object],
    output_path: Path | str,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(selection), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
