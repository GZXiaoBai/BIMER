#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bimer.repository_policy import inspect_public_tree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--maximum-bytes", type=int, default=10 * 1024**2)
    args = parser.parse_args()
    root = args.root.resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked = [Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value]
    violations = inspect_public_tree(
        root=root,
        tracked_paths=tracked,
        maximum_bytes=args.maximum_bytes,
    )
    print(
        json.dumps(
            {
                "ok": not violations,
                "tracked_files": len(tracked),
                "violations": [
                    {
                        "code": violation.code,
                        "path": violation.path.as_posix(),
                        "detail": violation.detail,
                    }
                    for violation in violations
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
