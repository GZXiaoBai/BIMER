#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping

from bimer.experiment import evaluate_checkpoint
from bimer.v5_protocol import run_guarded_v5_test

Evaluator = Callable[..., Path]


def run_v5_exploratory_test(
    *,
    selection_path: Path,
    formal_complete_marker: Path,
    checkpoint_path: Path,
    output_directory: Path,
    clean_manifest: Path,
    clean_features: Path,
    conditions: Mapping[str, tuple[Path, Path]] | None = None,
    evaluator: Evaluator = evaluate_checkpoint,
    device: str = "auto",
    bootstrap_iterations: int = 2000,
) -> Path:
    if not formal_complete_marker.is_file():
        raise RuntimeError("formal V5 training is not marked complete")
    output_directory.mkdir(parents=True, exist_ok=True)
    marker = output_directory / "TEST_EVALUATED"

    def operation() -> Path:
        outputs: dict[str, str] = {}
        clean_output = evaluator(
            manifest_path=clean_manifest,
            feature_root=clean_features,
            checkpoint_path=checkpoint_path,
            output_path=output_directory / "clean.json",
            condition_name="clean",
            bootstrap_iterations=bootstrap_iterations,
            device_name=device,
            evaluation_role="test",
        )
        outputs["clean"] = str(clean_output)
        for name, (manifest, features) in (conditions or {}).items():
            result = evaluator(
                manifest_path=manifest,
                feature_root=features,
                checkpoint_path=checkpoint_path,
                output_path=output_directory / f"{name}.json",
                condition_name=name,
                bootstrap_iterations=bootstrap_iterations,
                device_name=device,
                evaluation_role="test",
            )
            outputs[name] = str(result)
        report = output_directory / "exploratory-test-index.json"
        report.write_text(
            json.dumps(
                {
                    "version": "v5",
                    "scope": "post_hoc_exploratory",
                    "selection": str(selection_path.resolve()),
                    "checkpoint": str(checkpoint_path.resolve()),
                    "outputs": outputs,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return report

    return run_guarded_v5_test(selection_path, marker, operation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--formal-complete-marker", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    conditions: dict[str, tuple[Path, Path]] = {}
    for value in args.condition:
        name, manifest, features = value.split("=", 2)
        conditions[name] = (Path(manifest), Path(features))
    print(
        run_v5_exploratory_test(
            selection_path=args.selection,
            formal_complete_marker=args.formal_complete_marker,
            checkpoint_path=args.checkpoint,
            output_directory=args.output,
            clean_manifest=args.manifest,
            clean_features=args.features,
            conditions=conditions,
            device=args.device,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
