#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    variant: str
    seed: int
    classification_loss: str
    ranking_weight: float
    output: Path
    screen: bool

    @property
    def result_path(self) -> Path:
        return self.output / "quality_lagf" / "joint" / f"seed-{self.seed}" / "results.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.result_path.parent / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment-v3.toml")
    parser.add_argument(
        "--stage",
        choices=["loss-screen", "ranking-screen", "formal", "test"],
        required=True,
    )
    parser.add_argument("--manifest")
    parser.add_argument("--features")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--selection", default="configs/experiment-v3-selection.json")
    parser.add_argument("--classification-loss")
    parser.add_argument(
        "--validation-manifest",
        default="artifacts/features/v3-validation/manifests/validation-clean.jsonl",
    )
    parser.add_argument(
        "--whisper-validation-manifest",
        default="artifacts/features/v3-validation/manifests/validation-whisper.jsonl",
    )
    parser.add_argument(
        "--validation-audio-features",
        default="artifacts/features/v3-validation/audio-10db",
    )
    parser.add_argument(
        "--validation-video-features",
        default="artifacts/features/v3-validation/video-50",
    )
    parser.add_argument(
        "--validation-whisper-features",
        default="artifacts/features/v3-validation/whisper",
    )
    parser.add_argument("--augmentation-manifest", action="append", default=[])
    parser.add_argument("--augmentation-features", action="append", default=[])
    parser.add_argument("--augmentation-modality", action="append", default=[])
    parser.add_argument("--augmentation-severity", action="append", type=float, default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_selection(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "frozen" or payload.get("version") != "v3":
        raise SystemExit("V3 selection is not frozen")
    return payload


def build_jobs(config: dict, args: argparse.Namespace, output: Path) -> list[Job]:
    if args.stage == "loss-screen":
        return [
            Job(
                variant=f"loss-{loss}",
                seed=42,
                classification_loss=loss,
                ranking_weight=0.0,
                output=output / "screen" / "loss" / loss,
                screen=True,
            )
            for loss in config["classification_losses"]
        ]
    if args.stage == "ranking-screen":
        if args.classification_loss not in config["classification_losses"]:
            raise SystemExit(
                "ranking-screen requires --classification-loss from the completed loss screen"
            )
        return [
            Job(
                variant=f"rank-{weight:g}",
                seed=42,
                classification_loss=args.classification_loss,
                ranking_weight=float(weight),
                output=output / "screen" / "ranking" / f"lambda-{weight:g}",
                screen=True,
            )
            for weight in config["gate_ranking_weights"]
        ]
    selection = _load_selection(ROOT / args.selection)
    if args.stage == "formal":
        return [
            Job(
                variant=variant,
                seed=int(seed),
                classification_loss=selection["classification_loss"],
                ranking_weight=(
                    0.0 if variant == "v3_loss_only" else float(selection["gate_ranking_weight"])
                ),
                output=output / "formal" / variant,
                screen=False,
            )
            for variant in ("v3_loss_only", "v3_ranked")
            for seed in config["seeds"]
        ]
    return []


def _command(
    job: Job,
    *,
    config: dict,
    manifest: str,
    features: str,
    device: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bimer.cli",
        "train",
        "--manifest",
        manifest,
        "--features",
        features,
        "--output",
        str(job.output),
        "--model",
        "quality_lagf",
        "--training-scope",
        "joint",
        "--seed",
        str(job.seed),
        "--learning-rate",
        str(config["learning_rate"]),
        "--classification-loss",
        job.classification_loss,
        "--corrupted-classification-weight",
        str(config["corrupted_classification_weight"]),
        "--gate-ranking-weight",
        str(job.ranking_weight),
        "--gate-ranking-margin",
        str(config["gate_ranking_margin"]),
        "--device",
        device,
        "--skip-test",
    ]
    if job.screen:
        command.append("--v3-screen")
    else:
        command.append("--v3-formal")
    for value in args.augmentation_manifest:
        command.extend(("--augmentation-manifest", value))
    for value in args.augmentation_features:
        command.extend(("--augmentation-features", value))
    for value in args.augmentation_modality:
        command.extend(("--augmentation-modality", value))
    for value in args.augmentation_severity:
        command.extend(("--augmentation-severity", str(value)))
    return command


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _evaluate_validation_views(
    job: Job,
    *,
    args: argparse.Namespace,
    device: str,
) -> None:
    from bimer.experiment import evaluate_checkpoint

    condition_root = job.result_path.parent / "validation_conditions"
    for condition, manifest, features in (
        (
            "audio_10db",
            args.validation_manifest,
            args.validation_audio_features,
        ),
        (
            "video_50",
            args.validation_manifest,
            args.validation_video_features,
        ),
        (
            "whisper",
            args.whisper_validation_manifest,
            args.validation_whisper_features,
        ),
    ):
        output_path = condition_root / f"{condition}.json"
        if output_path.is_file():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if set(payload.get("validation", {})) == {"meld", "emotiontalk"}:
                continue
        evaluate_checkpoint(
            manifest_path=manifest,
            feature_root=features,
            checkpoint_path=job.checkpoint_path,
            output_path=output_path,
            condition_name=condition,
            evaluation_role="validation",
            device_name=device,
        )


def _run_test_once(
    *,
    config: dict,
    args: argparse.Namespace,
    manifest: str,
    features: str,
    output: Path,
    device: str,
) -> int:
    from bimer.experiment import evaluate_checkpoint
    from bimer.v3_protocol import run_guarded_v3_test

    selection_path = ROOT / args.selection
    selection = _load_selection(selection_path)
    marker = output / "exploratory-test" / "TEST_EVALUATED"
    system_checkpoint_selection = output / "formal" / "v3-system-checkpoint.json"
    if not args.dry_run and not system_checkpoint_selection.is_file():
        raise SystemExit("V3 system checkpoint is not frozen from validation-only formal results")

    def evaluate_all():
        paths = []
        for variant in ("v3_loss_only", "v3_ranked"):
            for seed in config["seeds"]:
                checkpoint = (
                    output
                    / "formal"
                    / variant
                    / "quality_lagf"
                    / "joint"
                    / f"seed-{seed}"
                    / "best.pt"
                )
                result = output / "exploratory-test" / f"{variant}-seed-{seed}.json"
                paths.append(
                    str(
                        evaluate_checkpoint(
                            manifest_path=manifest,
                            feature_root=features,
                            checkpoint_path=checkpoint,
                            output_path=result,
                            device_name=device,
                        )
                    )
                )
        return paths

    if args.dry_run:
        print(
            "RUN guarded-v3-test "
            + shlex.join(
                [
                    "--selection",
                    str(selection_path),
                    "--marker",
                    str(marker),
                    "--classification-loss",
                    selection["classification_loss"],
                    "--lambda",
                    str(selection["gate_ranking_weight"]),
                    "--system-checkpoint-selection",
                    str(system_checkpoint_selection),
                ]
            )
        )
        return 0
    run_guarded_v3_test(selection_path, marker, evaluate_all)
    return 0


def main() -> int:
    args = parse_args()
    with (ROOT / args.config).open("rb") as stream:
        config = tomllib.load(stream)
    defaults = config["defaults"]
    manifest = args.manifest or defaults["manifest"]
    features = args.features or defaults["features"]
    output = Path(args.output or defaults["output"])
    device = args.device or defaults["device"]
    if not any(
        (
            args.augmentation_manifest,
            args.augmentation_features,
            args.augmentation_modality,
            args.augmentation_severity,
        )
    ):
        augmentation_defaults = config["augmentations"]
        args.augmentation_manifest = list(augmentation_defaults["manifests"])
        args.augmentation_features = list(augmentation_defaults["features"])
        args.augmentation_modality = list(augmentation_defaults["modalities"])
        args.augmentation_severity = list(augmentation_defaults["severities"])
    lengths = {
        len(args.augmentation_manifest),
        len(args.augmentation_features),
        len(args.augmentation_modality),
        len(args.augmentation_severity),
    }
    if len(lengths) != 1:
        raise SystemExit(
            "paired augmentation manifests, features, modalities and severities must align"
        )
    if args.stage == "test":
        return _run_test_once(
            config=config,
            args=args,
            manifest=manifest,
            features=features,
            output=output,
            device=device,
        )
    jobs = build_jobs(config, args, output)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(ROOT / "src")
        + os.pathsep
        + environment.get(
            "PYTHONPATH",
            "",
        )
    )
    for job in jobs:
        command = _command(
            job,
            config=config,
            manifest=manifest,
            features=features,
            device=device,
            args=args,
        )
        print("RUN " + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        if job.result_path.exists():
            existing = json.loads(job.result_path.read_text(encoding="utf-8"))
            if existing.get("test") or existing.get("evaluation_datasets"):
                raise RuntimeError("V3 training result illegally contains test output")
            existing_config = existing.get("config", {})
            expected = {
                "seed": job.seed,
                "classification_loss": job.classification_loss,
                "gate_ranking_weight": job.ranking_weight,
                "protocol_stage": "v3_screen" if job.screen else "v3_formal",
                "evaluate_test": False,
            }
            mismatches = {
                name: (existing_config.get(name), value)
                for name, value in expected.items()
                if existing_config.get(name) != value
            }
            if mismatches:
                raise RuntimeError(f"existing V3 result conflicts with requested job: {mismatches}")
            if set(existing.get("validation", {})) != {"meld", "emotiontalk"}:
                raise RuntimeError("existing V3 result lacks bilingual validation metrics")
            if not job.checkpoint_path.is_file():
                raise RuntimeError("existing V3 result has no checkpoint")
            if job.screen:
                _evaluate_validation_views(job, args=args, device=device)
            print(f"SKIP verified validation-only result: {job.result_path}")
            continue
        started = time.time()
        try:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            if job.screen:
                _evaluate_validation_views(job, args=args, device=device)
            digest = hashlib.sha256(job.result_path.read_bytes()).hexdigest()
            _write_status(
                output / "_status" / f"{job.variant}-seed-{job.seed}.done.json",
                {
                    "status": "complete",
                    "elapsed_seconds": time.time() - started,
                    "sha256": digest,
                    "result": str(job.result_path),
                    "command": command,
                },
            )
        except BaseException as error:
            _write_status(
                output / "_status" / f"{job.variant}-seed-{job.seed}.failed.json",
                {
                    "status": "failed",
                    "elapsed_seconds": time.time() - started,
                    "error": f"{type(error).__name__}: {error}",
                    "command": command,
                },
            )
            raise
    if args.stage == "formal" and not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "select_v3_system_checkpoint.py"),
                "--formal-root",
                str(output / "formal" / "v3_ranked"),
                "--output",
                str(output / "formal" / "v3-system-checkpoint.json"),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
