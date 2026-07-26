#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LEARNING_RATES = (1e-4, 2e-4)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _tag(learning_rate: float) -> str:
    return f"lr_{int(round(learning_rate * 1_000_000)):03d}"


def _commands(
    args: argparse.Namespace,
    candidate_config: dict[str, object],
) -> list[tuple[str, list[str], Path]]:
    commands: list[tuple[str, list[str], Path]] = []
    for learning_rate in LEARNING_RATES:
        tag = _tag(learning_rate)
        root = args.output / tag
        adapter_root = root / "text-adaptation"
        adapted_features = root / "features"
        fusion_output = root / "fusion"
        train_command = [
            sys.executable,
            str(ROOT / "scripts" / "train_v4_text_lora.py"),
            "--manifest",
            str(args.manifest),
            "--base-model",
            args.base_model,
            "--output",
            str(adapter_root),
            "--learning-rate",
            str(learning_rate),
            "--device",
            args.device,
        ]
        extract_command = [
            sys.executable,
            str(ROOT / "scripts" / "extract_v4_lora_text_features.py"),
            "--manifest",
            str(args.manifest),
            "--source-features",
            str(args.source_features),
            "--output-features",
            str(adapted_features),
            "--base-model",
            args.base_model,
            "--adapter",
            str(adapter_root / "adapter"),
            "--device",
            args.device,
        ]
        fusion_command = [
            sys.executable,
            "-m",
            "bimer.cli",
            "train",
            "--manifest",
            str(args.manifest),
            "--features",
            str(adapted_features),
            "--output",
            str(fusion_output),
            "--model",
            str(candidate_config["model"]),
            "--training-scope",
            "joint",
            "--seed",
            "42",
            "--learning-rate",
            "0.0001",
            "--prototype-loss-weight",
            str(candidate_config["prototype_loss_weight"]),
            "--prototype-temperature",
            "0.07",
            "--device",
            args.device,
            "--no-language",
            "--skip-test",
            "--v4-screen",
        ]
        if not bool(candidate_config["use_adaptive_context_gate"]):
            fusion_command.append("--no-adaptive-context-gate")
        if args.local_files_only:
            train_command.append("--local-files-only")
            extract_command.append("--local-files-only")
        commands.extend(
            (
                (f"{tag}-adapter", train_command, adapter_root / "result.json"),
                (
                    f"{tag}-features",
                    extract_command,
                    adapted_features / "TEXT_FEATURES_READY.json",
                ),
                (
                    f"{tag}-fusion",
                    fusion_command,
                    fusion_output
                    / str(candidate_config["model"])
                    / "joint"
                    / "seed-42"
                    / "results.json",
                ),
            )
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if decision.get("decision") == "pass_v4a":
        print("SKIP_LORA: V4-A already passed its validation gates")
        return 0
    if decision.get("decision") != "trigger_lora":
        raise ValueError("screen decision must be pass_v4a or trigger_lora")
    best_candidate = str(decision["best_candidate"])
    try:
        candidate_config = decision["candidate_configs"][best_candidate]
    except (KeyError, TypeError) as exc:
        raise ValueError("screen decision is missing the best V4-A configuration") from exc
    print(f"TRIGGER_LORA: reusing {best_candidate}")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(ROOT / "src")
        + os.pathsep
        + environment.get(
            "PYTHONPATH",
            "",
        )
    )
    statuses: dict[str, object] = {}
    for name, command, completion_path in _commands(args, candidate_config):
        if args.dry_run:
            print("RUN " + shlex.join(command))
            continue
        if not completion_path.is_file():
            _atomic_json(
                args.output / "_status" / f"{name}.json",
                {"status": "running", "command": command},
            )
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
        if not completion_path.is_file():
            raise RuntimeError(f"LoRA stage did not produce {completion_path}")
        if name.endswith("-fusion"):
            from bimer.experiment import evaluate_checkpoint

            run_root = completion_path.parent
            checkpoint = run_root / "best.pt"
            condition_root = run_root / "validation_conditions"
            feature_root = completion_path.parents[4] / "features"
            for modality in ("text", "audio", "vision"):
                condition = condition_root / f"missing_{modality}.json"
                if condition.is_file():
                    continue
                evaluate_checkpoint(
                    manifest_path=args.manifest,
                    feature_root=feature_root,
                    checkpoint_path=checkpoint,
                    output_path=condition,
                    missing_modality=modality,
                    bootstrap_iterations=2000,
                    device_name=args.device,
                    evaluation_role="validation",
                )
        statuses[name] = {
            "status": "complete",
            "output": str(completion_path),
        }
        _atomic_json(args.output / "_status" / f"{name}.json", statuses[name])
    if not args.dry_run:
        _atomic_json(
            args.output / "LORA_SCREEN_READY.json",
            {
                "status": "complete",
                "evidence_scope": "validation_only",
                "test_set_used": False,
                "best_v4a_structure": best_candidate,
                "candidate_config": candidate_config,
                "stages": statuses,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
