#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from bimer.v5_protocol import select_v5_candidate


def _condition_path(condition_root: Path, name: str) -> Path:
    candidates = (
        ("video_drop_50.json", "video_50.json") if name == "video_drop_50" else (f"{name}.json",)
    )
    for candidate in candidates:
        path = condition_root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing validation condition {name}; tried "
        + ", ".join(str(condition_root / candidate) for candidate in candidates)
    )


def _conditions(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    condition_root = result_path.parent / "validation_conditions"
    return {
        "clean": result["validation"],
        **{
            name: json.loads(_condition_path(condition_root, name).read_text(encoding="utf-8"))[
                "validation"
            ]
            for name in ("whisper", "audio_10db", "video_drop_50")
        },
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def summarize_v5_screen(
    *,
    baseline_path: Path,
    candidate_paths: Mapping[str, Path],
    candidate_betas: Mapping[str, float],
    output_path: Path,
) -> Path:
    candidates = {
        name: {
            "beta": candidate_betas[name],
            "conditions": _conditions(path),
        }
        for name, path in candidate_paths.items()
    }
    decision = select_v5_candidate(
        baseline=_conditions(baseline_path),
        candidates=candidates,
    )
    payload = {
        **asdict(decision),
        "version": "v5",
        "evidence_scope": "validation_only",
        "test_set_used": False,
        "baseline": str(baseline_path),
        "candidates": {name: str(path) for name, path in candidate_paths.items()},
        "candidate_configs": {
            name: {"asr_consistency_weight": float(candidate_betas[name])}
            for name in candidate_paths
        },
    }
    _atomic_json(output_path, payload)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths: dict[str, Path] = {}
    betas: dict[str, float] = {}
    for value in args.candidate:
        name, beta_text, path = value.split("=", 2)
        paths[name] = Path(path)
        betas[name] = float(beta_text)
    summarize_v5_screen(
        baseline_path=Path(args.baseline),
        candidate_paths=paths,
        candidate_betas=betas,
        output_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
