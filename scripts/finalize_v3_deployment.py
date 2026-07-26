#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bimer.deployment_selection import (
    select_deployment_model,
    write_deployment_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--external-report", required=True, type=Path)
    parser.add_argument("--m2-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select_deployment_model(
        frozen_selection=json.loads(args.selection.read_text(encoding="utf-8")),
        external_report=json.loads(args.external_report.read_text(encoding="utf-8")),
        m2_report=json.loads(args.m2_report.read_text(encoding="utf-8")),
    )
    write_deployment_selection(result, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
