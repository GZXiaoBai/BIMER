from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from bimer.external_annotation_pack import (
    ANNOTATION_COLUMNS,
    AnnotationSegment,
    build_annotation_rows,
)


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_external_annotations.py",
            "--master",
            str(tmp_path / "master.csv"),
            "--annotator-one",
            str(tmp_path / "one.csv"),
            "--annotator-two",
            str(tmp_path / "two.csv"),
            "--adjudication",
            str(tmp_path / "adjudication.csv"),
            "--report",
            str(tmp_path / "report.json"),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )


def test_check_script_reports_blank_human_annotations_as_blocked(tmp_path: Path) -> None:
    rows = build_annotation_rows({"en-normal-01": [AnnotationSegment(0.0, 3.0, "Hello", 0.9)]})
    for name in ("master.csv", "one.csv", "two.csv"):
        _write(tmp_path / name, rows)

    result = _run(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert result.returncode == 2
    assert report["status"] == "blocked_human_annotation"
    assert report["agreement_calculated"] is False
    assert not (tmp_path / "adjudication.csv").exists()


def test_check_script_requires_then_records_independence_attestation(tmp_path: Path) -> None:
    master = build_annotation_rows(
        {
            "en-normal-01": [
                AnnotationSegment(0.0, 3.0, "Hello", 0.9),
                AnnotationSegment(3.0, 6.0, "Goodbye", 0.8),
            ]
        }
    )
    _write(tmp_path / "master.csv", master)
    _write(
        tmp_path / "one.csv",
        [{**master[0], "label": "joy"}, {**master[1], "label": "sadness"}],
    )
    _write(
        tmp_path / "two.csv",
        [{**master[0], "label": "joy"}, {**master[1], "label": "sadness"}],
    )

    awaiting = _run(tmp_path)
    awaiting_report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    confirmed = _run(tmp_path, "--confirm-independent")
    confirmed_report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert awaiting.returncode == 2
    assert awaiting_report["status"] == "awaiting_independence_attestation"
    assert confirmed.returncode == 0
    assert confirmed_report["status"] == "ready_for_adjudication"
    assert confirmed_report["cohen_kappa"] == 1.0
    assert (tmp_path / "adjudication.csv").is_file()
