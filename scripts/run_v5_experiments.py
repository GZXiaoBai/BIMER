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


@dataclass(frozen=True, slots=True)
class Job:
    variant: str
    seed: int
    beta: float
    output: Path
    stage: str

    @property
    def result_path(self) -> Path:
        return (
            self.output
            / "asr_consistent_quality_lagf"
            / "joint"
            / f"seed-{self.seed}"
            / "results.json"
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.result_path.parent / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment-v5.toml")
    parser.add_argument("--stage", choices=["screen", "formal"], required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--features")
    parser.add_argument("--paired-manifest")
    parser.add_argument("--paired-features")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--selection", default="configs/experiment-v5-selection.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_selection(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "frozen" or payload.get("version") != "v5":
        raise SystemExit("V5 selection is not frozen")
    return payload


def build_jobs(
    config: dict[str, object],
    *,
    stage: str,
    output: Path,
    selection_path: Path,
) -> list[Job]:
    if stage == "screen":
        return [
            Job(
                variant=f"beta_{int(round(float(beta) * 100)):03d}",
                seed=42,
                beta=float(beta),
                output=output / "screen" / f"beta_{int(round(float(beta) * 100)):03d}",
                stage=stage,
            )
            for beta in config["screen_betas"]  # type: ignore[union-attr]
        ]
    selection = _load_selection(selection_path)
    candidate_config = selection.get("candidate_config")
    if not isinstance(candidate_config, dict):
        raise SystemExit("V5 frozen selection has no candidate_config")
    beta = float(candidate_config["asr_consistency_weight"])
    return [
        Job(
            variant="asr_consistent",
            seed=int(seed),
            beta=beta,
            output=output / "formal" / "asr_consistent",
            stage=stage,
        )
        for seed in config["seeds"]  # type: ignore[union-attr]
    ]


def _command(
    job: Job,
    *,
    config: dict[str, object],
    manifest: str,
    features: str,
    paired_manifest: str,
    paired_features: str,
    device: str,
) -> list[str]:
    return [
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
        "asr_consistent_quality_lagf",
        "--training-scope",
        "joint",
        "--seed",
        str(job.seed),
        "--learning-rate",
        str(config["learning_rate"]),
        "--max-epochs",
        str(config["max_epochs"]),
        "--min-epochs",
        str(config["min_epochs"]),
        "--patience",
        str(config["patience"]),
        "--augmentation-manifest",
        paired_manifest,
        "--augmentation-features",
        paired_features,
        "--augmentation-modality",
        "text",
        "--augmentation-severity",
        "1.0",
        "--corrupted-classification-weight",
        str(config["corrupted_classification_weight"]),
        "--asr-consistency-weight",
        str(job.beta),
        "--device",
        device,
        "--no-language",
        "--skip-test",
        "--v5-screen" if job.stage == "screen" else "--v5-formal",
    ]


def _complete(job: Job) -> bool:
    if not job.result_path.is_file() or not job.checkpoint_path.is_file():
        return False
    payload = json.loads(job.result_path.read_text(encoding="utf-8"))
    return set(payload.get("validation", {})) == {"meld", "emotiontalk"}


def _write_status(path: Path, payload: object) -> None:
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


def _evaluate_validation_conditions(
    job: Job,
    *,
    defaults: dict[str, object],
    device: str,
) -> None:
    from bimer.experiment import evaluate_checkpoint

    condition_root = job.result_path.parent / "validation_conditions"
    conditions = (
        (
            "whisper",
            str(defaults["whisper_validation_manifest"]),
            str(defaults["whisper_validation_features"]),
        ),
        (
            "audio_10db",
            str(defaults["validation_manifest"]),
            str(defaults["audio_validation_features"]),
        ),
        (
            "video_drop_50",
            str(defaults["validation_manifest"]),
            str(defaults["video_validation_features"]),
        ),
    )
    for name, manifest, features in conditions:
        output = condition_root / f"{name}.json"
        if output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
            if set(payload.get("validation", {})) == {"meld", "emotiontalk"}:
                continue
        evaluate_checkpoint(
            manifest_path=manifest,
            feature_root=features,
            checkpoint_path=job.checkpoint_path,
            output_path=output,
            condition_name=name,
            bootstrap_iterations=2000,
            device_name=device,
            evaluation_role="validation",
        )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    defaults = config["defaults"]
    manifest = args.manifest or defaults["manifest"]
    features = args.features or defaults["features"]
    paired_manifest = args.paired_manifest or defaults["paired_manifest"]
    paired_features = args.paired_features or defaults["paired_features"]
    output = Path(args.output or defaults["output"])
    device = args.device or defaults["device"]
    selection = Path(args.selection)
    jobs = build_jobs(
        defaults,
        stage=args.stage,
        output=output,
        selection_path=selection,
    )
    for job in jobs:
        command = _command(
            job,
            config=defaults,
            manifest=str(manifest),
            features=str(features),
            paired_manifest=str(paired_manifest),
            paired_features=str(paired_features),
            device=str(device),
        )
        if args.dry_run:
            print("RUN " + shlex.join(command))
            continue
        status_path = output / "_status" / f"{args.stage}-{job.variant}-{job.seed}.json"
        if not _complete(job):
            _write_status(
                status_path,
                {
                    "status": "running",
                    "variant": job.variant,
                    "seed": job.seed,
                    "command": command,
                },
            )
            subprocess.run(command, cwd=ROOT, check=True)
        _evaluate_validation_conditions(job, defaults=defaults, device=str(device))
        _write_status(
            status_path,
            {
                "status": "completed",
                "variant": job.variant,
                "seed": job.seed,
                "result": str(job.result_path),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
