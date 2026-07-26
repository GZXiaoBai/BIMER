#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    variant: str
    model: str
    seed: int
    prototype_weight: float
    adaptive_context: bool
    output: Path
    screen: bool

    @property
    def result_path(self) -> Path:
        return self.output / self.model / "joint" / f"seed-{self.seed}" / "results.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.result_path.parent / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment-v4.toml")
    parser.add_argument("--stage", choices=["screen", "formal"], required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--features")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--selection", default="configs/experiment-v4-selection.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_selection(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "frozen" or payload.get("version") != "v4":
        raise SystemExit("V4 selection is not frozen")
    return payload


def _screen_jobs(config: dict, output: Path) -> list[Job]:
    jobs = [
        Job(
            variant="v2_no_language",
            model="quality_lagf",
            seed=42,
            prototype_weight=0.0,
            adaptive_context=False,
            output=output / "screen" / "v2_no_language",
            screen=True,
        ),
        Job(
            variant="context_only",
            model="adaptive_context_prototype",
            seed=42,
            prototype_weight=0.0,
            adaptive_context=True,
            output=output / "screen" / "context_only",
            screen=True,
        ),
        Job(
            variant="prototype_only",
            model="adaptive_context_prototype",
            seed=42,
            prototype_weight=float(config["prototype_only_weight"]),
            adaptive_context=False,
            output=output / "screen" / "prototype_only",
            screen=True,
        ),
    ]
    jobs.extend(
        Job(
            variant=f"combined_mu_{int(round(float(weight) * 1000)):03d}",
            model="adaptive_context_prototype",
            seed=42,
            prototype_weight=float(weight),
            adaptive_context=True,
            output=output / "screen" / f"combined_mu_{int(round(float(weight) * 1000)):03d}",
            screen=True,
        )
        for weight in config["prototype_weights"]
    )
    return jobs


def _formal_jobs(config: dict, output: Path, selection: dict) -> list[Job]:
    selected = selection["candidate_config"]
    prototype_weight = float(selected["prototype_loss_weight"])
    adaptive_context = bool(selected["use_adaptive_context_gate"])
    variants = (
        ("full", prototype_weight, adaptive_context),
        ("no_context_gate", prototype_weight, False),
        ("no_prototype", 0.0, adaptive_context),
        ("neither", 0.0, False),
    )
    return [
        Job(
            variant=variant,
            model="adaptive_context_prototype",
            seed=int(seed),
            prototype_weight=weight,
            adaptive_context=adaptive,
            output=output / "formal" / variant,
            screen=False,
        )
        for variant, weight, adaptive in variants
        for seed in config["seeds"]
    ]


def build_jobs(config: dict, args: argparse.Namespace, output: Path) -> list[Job]:
    if args.stage == "screen":
        return _screen_jobs(config, output)
    return _formal_jobs(config, output, _load_selection(Path(args.selection)))


def _command(
    job: Job,
    *,
    config: dict,
    manifest: str,
    features: str,
    device: str,
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
        job.model,
        "--training-scope",
        "joint",
        "--seed",
        str(job.seed),
        "--learning-rate",
        str(config["learning_rate"]),
        "--prototype-loss-weight",
        str(job.prototype_weight),
        "--prototype-temperature",
        "0.07",
        "--device",
        device,
        "--no-language",
        "--skip-test",
        "--v4-screen" if job.screen else "--v4-formal",
    ]
    if job.model == "adaptive_context_prototype" and not job.adaptive_context:
        command.append("--no-adaptive-context-gate")
    return command


def _is_result_complete(job: Job) -> bool:
    if not job.result_path.is_file() or not job.checkpoint_path.is_file():
        return False
    payload = json.loads(job.result_path.read_text(encoding="utf-8"))
    return set(payload.get("validation", {})) == {"meld", "emotiontalk"}


def _evaluate_missing_validation(
    job: Job,
    *,
    manifest: str,
    features: str,
    device: str,
) -> None:
    from bimer.experiment import evaluate_checkpoint

    condition_root = job.result_path.parent / "validation_conditions"
    for modality in ("text", "audio", "vision"):
        output = condition_root / f"missing_{modality}.json"
        if output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
            if set(payload.get("validation", {})) == {"meld", "emotiontalk"}:
                continue
        evaluate_checkpoint(
            manifest_path=manifest,
            feature_root=features,
            checkpoint_path=job.checkpoint_path,
            output_path=output,
            missing_modality=modality,
            bootstrap_iterations=2000,
            device_name=device,
            evaluation_role="validation",
        )


def _write_status(path: Path, payload: dict) -> None:
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
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    defaults = config["defaults"]
    manifest = args.manifest or defaults["manifest"]
    frozen_selection = _load_selection(Path(args.selection)) if args.stage == "formal" else None
    frozen_features = (
        frozen_selection["candidate_config"].get("feature_root")
        if frozen_selection is not None
        else None
    )
    features = args.features or frozen_features or defaults["features"]
    output = Path(args.output or defaults["output"])
    device = args.device or defaults["device"]
    jobs = build_jobs(config, args, output)
    status_root = output / "_status"
    for job in jobs:
        command = _command(
            job,
            config=config,
            manifest=manifest,
            features=features,
            device=device,
        )
        if args.dry_run:
            print("RUN " + shlex.join(command))
            continue
        if not _is_result_complete(job):
            started = {"status": "running", "variant": job.variant, "command": command}
            _write_status(status_root / f"{args.stage}-{job.variant}-{job.seed}.json", started)
            subprocess.run(command, cwd=ROOT, check=True)
        if job.screen:
            _evaluate_missing_validation(
                job,
                manifest=manifest,
                features=features,
                device=device,
            )
        _write_status(
            status_root / f"{args.stage}-{job.variant}-{job.seed}.json",
            {
                "status": "complete",
                "variant": job.variant,
                "result": str(job.result_path),
            },
        )
    if not args.dry_run:
        _write_status(
            status_root / f"{args.stage.upper()}_COMPLETE",
            {"status": "complete", "jobs": len(jobs)},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
