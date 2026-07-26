#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = []
    for seed in (42, 123, 2026):
        result = args.formal_root / "quality_lagf" / "joint" / f"seed-{seed}" / "results.json"
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("test") or payload.get("evaluation_datasets"):
            raise SystemExit("system checkpoint selection must not use test output")
        if payload["config"].get("protocol_stage") != "v3_formal":
            raise SystemExit("candidate is not a frozen V3 formal run")
        validation = payload["validation"]
        score = (
            sum(float(validation[dataset]["weighted_f1"]) for dataset in ("meld", "emotiontalk"))
            / 2
        )
        checkpoint = result.parent / "best.pt"
        candidates.append((score, -seed, seed, checkpoint))
    score, _, seed, checkpoint = max(candidates)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selection = {
        "state": "frozen",
        "selection_scope": "validation_only",
        "test_set_used": False,
        "variant": "v3_ranked",
        "seed": seed,
        "bilingual_validation_weighted_f1": score,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
