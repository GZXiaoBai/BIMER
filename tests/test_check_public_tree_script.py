from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_tree_script_checks_the_actual_git_index(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    safe = tmp_path / "src" / "module.py"
    safe.parent.mkdir()
    safe.write_text("value = 42\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src/module.py"], check=True)

    safe_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_public_tree.py"),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert safe_result.returncode == 0
    assert json.loads(safe_result.stdout)["ok"] is True

    private = tmp_path / "data" / "processed" / "all.jsonl"
    private.parent.mkdir(parents=True)
    private.write_text("{}\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "-f", "data/processed/all.jsonl"],
        check=True,
    )
    blocked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_public_tree.py"),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert blocked.returncode == 1
    report = json.loads(blocked.stdout)
    assert report["violations"][0]["code"] == "forbidden-path"
