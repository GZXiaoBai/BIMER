#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEEDS = (42, 123, 2026)
DATASETS = ("meld", "emotiontalk")


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    checkpoint_template: str


@dataclass(frozen=True)
class RobustnessCondition:
    name: str
    feature_directory: str
    missing_modalities: tuple[str, ...] = ()
    use_whisper_manifest: bool = False


MODELS = (
    ModelCandidate(
        "quality_lagf",
        "artifacts/experiments/v2/formal/quality_lagf/quality_lagf/joint/seed-{seed}/best.pt",
    ),
    ModelCandidate(
        "no_gates",
        "artifacts/experiments/v2/ablations/no_gates/quality_lagf/joint/seed-{seed}/best.pt",
    ),
)
CONDITIONS = (
    RobustnessCondition("standard", "standard"),
    RobustnessCondition("audio_snr_20db", "audio_snr_20db"),
    RobustnessCondition("audio_snr_10db", "audio_snr_10db"),
    RobustnessCondition(
        "video_frame_drop_25pct",
        "video_frame_drop_25pct",
    ),
    RobustnessCondition(
        "video_frame_drop_50pct",
        "video_frame_drop_50pct",
    ),
    RobustnessCondition(
        "whisper_text",
        "whisper_text",
        use_whisper_manifest=True,
    ),
    RobustnessCondition("missing-text", "standard", ("text",)),
    RobustnessCondition("missing-audio", "standard", ("audio",)),
    RobustnessCondition("missing-vision", "standard", ("vision",)),
    RobustnessCondition(
        "missing-audio-vision",
        "standard",
        ("audio", "vision"),
    ),
    RobustnessCondition(
        "missing-text-vision",
        "standard",
        ("text", "vision"),
    ),
    RobustnessCondition(
        "missing-text-audio",
        "standard",
        ("text", "audio"),
    ),
)
REQUIRED_PREDICTION_KEYS = {
    "sample_ids",
    "context_ids",
    "truth",
    "prediction",
    "probabilities",
    "gates",
    "modality_quality",
    "modality_available",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/root/autodl-tmp/bimer-runtime"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--whisper-manifest", type=Path)
    parser.add_argument("--base-features", type=Path)
    parser.add_argument("--robustness-features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.root.resolve()
    return {
        "root": root,
        "manifest": (
            args.manifest.resolve() if args.manifest else root / "data/processed/v2/all.jsonl"
        ),
        "whisper_manifest": (
            args.whisper_manifest.resolve()
            if args.whisper_manifest
            else root / "data/processed/v2/whisper-test.jsonl"
        ),
        "base_features": (
            args.base_features.resolve()
            if args.base_features
            else root / "artifacts/features/bilingual-v2-quality"
        ),
        "robustness_features": (
            args.robustness_features.resolve()
            if args.robustness_features
            else root / "artifacts/features/v2-robustness"
        ),
        "output": (
            args.output.resolve() if args.output else root / "artifacts/experiments/v2/robustness"
        ),
    }


def _feature_root(
    paths: dict[str, Path],
    condition: RobustnessCondition,
) -> Path:
    if condition.feature_directory == "standard":
        return paths["base_features"]
    return paths["robustness_features"] / condition.feature_directory


def _prediction_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if set(payload.get("test", {})) != set(DATASETS):
        return False
    prediction_root = path.parent / f"{path.stem}.predictions"
    for dataset in DATASETS:
        prediction_path = prediction_root / f"{dataset}.npz"
        if not prediction_path.is_file():
            return False
        try:
            with np.load(prediction_path, allow_pickle=False) as arrays:
                if not REQUIRED_PREDICTION_KEYS.issubset(arrays.files):
                    return False
        except (OSError, ValueError):
            return False
    return True


def _command(
    *,
    paths: dict[str, Path],
    model: ModelCandidate,
    condition: RobustnessCondition,
    seed: int,
    device: str,
) -> list[str]:
    manifest = paths["whisper_manifest"] if condition.use_whisper_manifest else paths["manifest"]
    checkpoint = paths["root"] / model.checkpoint_template.format(seed=seed)
    output = paths["output"] / model.name / condition.name / f"seed-{seed}.json"
    command = [
        sys.executable,
        "-m",
        "bimer.cli",
        "evaluate",
        "--manifest",
        str(manifest),
        "--features",
        str(_feature_root(paths, condition)),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        device,
    ]
    for modality in condition.missing_modalities:
        command.extend(["--missing", modality])
    if condition.name != "standard" and not condition.missing_modalities:
        command.extend(["--condition-name", condition.name])
    return command


def _validate_assets(paths: dict[str, Path]) -> None:
    for manifest in (paths["manifest"], paths["whisper_manifest"]):
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest is missing: {manifest}")
    feature_roots = {_feature_root(paths, condition) for condition in CONDITIONS}
    for feature_root in feature_roots:
        for dataset in DATASETS:
            split = feature_root / dataset / "test"
            if not split.is_dir():
                raise FileNotFoundError(f"feature split is missing: {split}")
    for model in MODELS:
        for seed in SEEDS:
            checkpoint = paths["root"] / model.checkpoint_template.format(seed=seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")


def main() -> int:
    args = parse_args()
    paths = _resolved_paths(args)
    if not args.dry_run:
        _validate_assets(paths)
    paths["output"].mkdir(parents=True, exist_ok=True)
    run_count = 0
    skip_count = 0
    for model in MODELS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                command = _command(
                    paths=paths,
                    model=model,
                    condition=condition,
                    seed=seed,
                    device=args.device,
                )
                output = Path(command[command.index("--output") + 1])
                if _prediction_complete(output):
                    print(f"SKIP model={model.name} condition={condition.name} seed={seed}")
                    skip_count += 1
                    continue
                print(
                    "RUN " + shlex.join(command),
                    flush=True,
                )
                run_count += 1
                if not args.dry_run:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(command, cwd=paths["root"], check=True)
                    if not _prediction_complete(output):
                        raise RuntimeError(f"evaluation did not produce complete output: {output}")
    if not args.dry_run:
        status = paths["output"] / "_status"
        status.mkdir(parents=True, exist_ok=True)
        (status / "EVALUATION_COMPLETE").write_text(
            f"runs={run_count} skipped={skip_count}\n",
            encoding="utf-8",
        )
    print(f"V2_ROBUSTNESS_EVALUATION runs={run_count} skipped={skip_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
