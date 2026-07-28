#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bimer.external_annotation_pack import (
    ANNOTATION_COLUMNS,
    audit_annotation_readiness,
    prepare_adjudication_rows,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check two independent human annotation files and prepare adjudication."
    )
    parser.add_argument(
        "--master",
        type=Path,
        help="Locked segment CSV; defaults to 00-segments-and-asr.csv beside annotator one.",
    )
    parser.add_argument("--annotator-one", required=True, type=Path)
    parser.add_argument("--annotator-two", required=True, type=Path)
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--confirm-independent",
        action="store_true",
        help="Attest that two different human annotators completed the files independently.",
    )
    args = parser.parse_args()

    master_path = args.master or args.annotator_one.with_name("00-segments-and-asr.csv")
    first = _rows(args.annotator_one)
    second = _rows(args.annotator_two)
    report = audit_annotation_readiness(
        _rows(master_path),
        first,
        second,
        independence_confirmed=args.confirm_independent,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.report)
    if not report["agreement_calculated"]:
        return 2

    rows, _ = prepare_adjudication_rows(first, second)
    args.adjudication.parent.mkdir(parents=True, exist_ok=True)
    with args.adjudication.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return 1 if report["requires_reannotation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
