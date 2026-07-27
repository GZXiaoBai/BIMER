#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bimer.external_annotation_pack import (
    ANNOTATION_COLUMNS,
    prepare_adjudication_rows,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check two independent human annotation files and prepare adjudication."
    )
    parser.add_argument("--annotator-one", required=True, type=Path)
    parser.add_argument("--annotator-two", required=True, type=Path)
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    rows, report = prepare_adjudication_rows(
        _rows(args.annotator_one),
        _rows(args.annotator_two),
    )
    args.adjudication.parent.mkdir(parents=True, exist_ok=True)
    with args.adjudication.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.report)
    return 0 if not report["requires_reannotation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
