#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimer.system_acceptance import evaluate_system_swap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a post-reboot M2 resource report.",
    )
    parser.add_argument("--resource-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    resource = json.loads(args.resource_report.read_text(encoding="utf-8"))
    result = evaluate_system_swap(
        resource["system_swap_before"],
        resource["system_swap_after"],
    )
    result["process_swaps_zero"] = int(resource["process_swaps"]) == 0
    result["passed"] = bool(result["passed"] and result["process_swaps_zero"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
