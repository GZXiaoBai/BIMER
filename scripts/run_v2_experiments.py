#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    tag: str
    output: Path
    model: str
    scope: str
    seed: int
    learning_rate: float
    flags: tuple[str, ...] = ()
    skip_test: bool = False
    use_augmentations: bool = True

    @property
    def result_path(self) -> Path:
        return self.output / self.model / self.scope / f"seed-{self.seed}" / "results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment-v2.toml")
    parser.add_argument(
        "--stage",
        choices=["audio-screen", "fusion-screen", "formal", "ablations", "all"],
        default="all",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--features")
    parser.add_argument("--quality-features")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--augmentation-manifest", action="append", default=[])
    parser.add_argument("--augmentation-features", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant", action="append", default=[])
    return parser.parse_args()


def _variant(name: str) -> tuple[str, tuple[str, ...]]:
    if name == "lagf_no_gates":
        return "lagf", ("--no-gates",)
    return name, ()


def build_jobs(config: dict, args: argparse.Namespace, output: Path) -> list[Job]:
    stages = (
        ("audio-screen", "fusion-screen", "formal", "ablations")
        if args.stage == "all"
        else (args.stage,)
    )
    jobs: list[Job] = []
    if "audio-screen" in stages:
        for scope in ("meld", "emotiontalk"):
            for rate in config["audio_learning_rates"]:
                jobs.append(
                    Job(
                        tag=f"audio-{scope}-lr-{rate}",
                        output=output / "screen" / "audio" / f"lr-{rate}",
                        model="audio",
                        scope=scope,
                        seed=42,
                        learning_rate=float(rate),
                        skip_test=True,
                        use_augmentations=False,
                    )
                )
    if "fusion-screen" in stages:
        variants = args.variant or config["formal"]["variants"]
        for name in variants:
            model, flags = _variant(name)
            for rate in config["fusion_learning_rates"]:
                jobs.append(
                    Job(
                        tag=f"{name}-lr-{rate}",
                        output=output / "screen" / name / f"lr-{rate}",
                        model=model,
                        scope="joint",
                        seed=42,
                        learning_rate=float(rate),
                        flags=flags,
                        skip_test=True,
                    )
                )
    if "formal" in stages:
        for name in (args.variant or config["formal"]["variants"]):
            model, flags = _variant(name)
            for seed in config["seeds"]:
                jobs.append(
                    Job(
                        tag=f"formal-{name}-seed-{seed}",
                        output=output / "formal" / name,
                        model=model,
                        scope="joint",
                        seed=int(seed),
                        learning_rate=float(config["formal_learning_rate"]),
                        flags=flags,
                    )
                )
    if "ablations" in stages:
        for name, flags in config["ablations"].items():
            for seed in config["seeds"]:
                jobs.append(
                    Job(
                        tag=f"ablation-{name}-seed-{seed}",
                        output=output / "ablations" / name,
                        model="quality_lagf",
                        scope="joint",
                        seed=int(seed),
                        learning_rate=float(config["formal_learning_rate"]),
                        flags=tuple(flags),
                        use_augmentations=name != "no_perturbation_training",
                    )
                )
    return jobs


def _command(
    job: Job,
    *,
    manifest: str,
    features: str,
    device: str,
    augmentation_manifests: list[str],
    augmentation_features: list[str],
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
        job.scope,
        "--seed",
        str(job.seed),
        "--learning-rate",
        str(job.learning_rate),
        "--device",
        device,
        *job.flags,
    ]
    if job.skip_test:
        command.append("--skip-test")
    if job.use_augmentations:
        for name in augmentation_manifests:
            command.extend(("--augmentation-manifest", name))
        for name in augmentation_features:
            command.extend(("--augmentation-features", name))
    return command


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _verify_existing_result(job: Job) -> bool:
    if not job.result_path.is_file():
        return False
    payload = json.loads(job.result_path.read_text(encoding="utf-8"))
    existing = payload.get("config", {})
    expected = {
        "model": job.model,
        "seed": job.seed,
        "training_scope": job.scope,
        "learning_rate": job.learning_rate,
        "evaluate_test": not job.skip_test,
    }
    mismatches = {
        name: (existing.get(name), value)
        for name, value in expected.items()
        if existing.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"existing result does not match requested job {job.tag}: {mismatches}"
        )
    if job.skip_test and (
        payload.get("test") or payload.get("evaluation_datasets")
    ):
        raise ValueError(f"validation-only result {job.result_path} contains test output")
    if not job.skip_test and not payload.get("test"):
        raise ValueError(f"formal result {job.result_path} contains no test output")
    return True


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    defaults = config["defaults"]
    manifest = args.manifest or defaults["manifest"]
    features = args.quality_features or args.features or defaults["features"]
    output = Path(args.output or defaults["output"])
    device = args.device or defaults["device"]
    if len(args.augmentation_manifest) != len(args.augmentation_features):
        raise SystemExit("augmentation manifest/feature arguments must be paired")
    jobs = build_jobs(config, args, output)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    for job in jobs:
        command = _command(
            job,
            manifest=manifest,
            features=features,
            device=device,
            augmentation_manifests=args.augmentation_manifest,
            augmentation_features=args.augmentation_features,
        )
        print("RUN " + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        if _verify_existing_result(job):
            print(f"SKIP verified result exists: {job.result_path}", flush=True)
            continue
        started = time.time()
        try:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            digest = hashlib.sha256(job.result_path.read_bytes()).hexdigest()
            _write_status(
                output / "_status" / f"{job.tag}.done.json",
                {
                    "status": "complete",
                    "elapsed_seconds": time.time() - started,
                    "result": str(job.result_path),
                    "sha256": digest,
                    "command": command,
                },
            )
        except BaseException as error:
            _write_status(
                output / "_status" / f"{job.tag}.failed.json",
                {
                    "status": "failed",
                    "elapsed_seconds": time.time() - started,
                    "error": f"{type(error).__name__}: {error}",
                    "command": command,
                },
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
