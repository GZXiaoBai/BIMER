#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bimer.manifest import read_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [
        record
        for record in read_manifest(args.manifest)
        if (
            (record.dataset == "meld" and str(record.split) == "dev")
            or (
                record.dataset == "emotiontalk"
                and str(record.split) == "validation"
            )
        )
    ]
    if not records:
        raise SystemExit("source manifest contains no validation records")
    write_manifest(records, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
