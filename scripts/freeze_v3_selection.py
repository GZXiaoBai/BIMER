#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimer.v3_protocol import freeze_v3_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-loss", required=True)
    parser.add_argument("--gate-ranking-weight", type=float, required=True)
    parser.add_argument("--loss-decision", required=True)
    parser.add_argument("--ranking-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    loss_evidence = json.loads(Path(args.loss_decision).read_text(encoding="utf-8"))
    ranking_evidence = json.loads(Path(args.ranking_decision).read_text(encoding="utf-8"))
    if args.classification_loss != loss_evidence.get("selected"):
        raise SystemExit("classification loss is not the validation-screen selection")
    if (
        float(args.gate_ranking_weight) != float(ranking_evidence.get("selected", 0.0))
        or float(args.gate_ranking_weight) <= 0
    ):
        raise SystemExit("gate ranking weight did not pass the validation-only screen")
    freeze_v3_selection(
        args.output,
        classification_loss=args.classification_loss,
        gate_ranking_weight=args.gate_ranking_weight,
        evidence={
            "validation_only": True,
            "test_set_used": False,
            "loss_decision": loss_evidence,
            "ranking_decision": ranking_evidence,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
