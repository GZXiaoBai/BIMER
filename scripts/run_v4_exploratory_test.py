#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bimer.experiment import evaluate_checkpoint
from bimer.labels import EMOTION_LABELS
from bimer.metrics import classification_metrics
from bimer.v4_analysis import ensemble_predictions
from bimer.v4_protocol import run_guarded_v4_test

SEEDS = (42, 123, 2026)
MISSING_CONDITIONS = {
    "missing_text": ("text",),
    "missing_audio": ("audio",),
    "missing_vision": ("vision",),
    "missing_audio_vision": ("audio", "vision"),
    "missing_text_vision": ("text", "vision"),
    "missing_text_audio": ("text", "audio"),
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


def _checkpoint(formal_root: Path, model: str, seed: int) -> Path:
    return formal_root / "full" / model / "joint" / f"seed-{seed}" / "best.pt"


def _summarize_test(output: Path, condition_names: tuple[str, ...]) -> Path:
    summary: dict[str, object] = {
        "seeds": list(SEEDS),
        "conditions": {},
    }
    ensemble_root = output / "ensemble"
    ensemble_root.mkdir(parents=True, exist_ok=True)
    for condition in condition_names:
        runs = [
            json.loads((output / condition / f"seed-{seed}.json").read_text(encoding="utf-8"))
            for seed in SEEDS
        ]
        condition_report: dict[str, object] = {"datasets": {}, "ensemble": {}}
        for dataset in ("meld", "emotiontalk"):
            metrics = {}
            for metric in ("weighted_f1", "macro_f1", "accuracy"):
                values = np.asarray(
                    [run["test"][dataset][metric] for run in runs],
                    dtype=np.float64,
                )
                metrics[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            condition_report["datasets"][dataset] = metrics
            ensemble = ensemble_predictions(
                [
                    output / condition / f"seed-{seed}.predictions" / f"{dataset}.npz"
                    for seed in SEEDS
                ]
            )
            condition_ensemble_root = ensemble_root / condition
            condition_ensemble_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(condition_ensemble_root / f"{dataset}.npz", **ensemble)
            condition_report["ensemble"][dataset] = classification_metrics(
                ensemble["truth"],
                ensemble["prediction"],
                label_names=EMOTION_LABELS,
            )
        summary["conditions"][condition] = condition_report
    summary["ensemble_probability_average"] = True
    summary["result_type"] = "v4_post_hoc_exploratory"
    summary["test_evaluated_once"] = True
    path = output / "exploratory-test-summary.json"
    _atomic_json(path, summary)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--robustness-features", type=Path)
    parser.add_argument("--whisper-manifest", type=Path)
    parser.add_argument("--lora-robustness-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("state") != "frozen" or selection.get("version") != "v4":
        raise RuntimeError("V4 selection is not frozen")
    formal = json.loads(args.formal_summary.read_text(encoding="utf-8"))
    if formal.get("test_set_used") is not False:
        raise RuntimeError("formal summary is not validation-only")
    if formal.get("formal_stable") is not True:
        raise RuntimeError("V4 formal result is not stable; official test is forbidden")
    if formal.get("seeds") != list(SEEDS):
        raise RuntimeError("formal summary does not contain the frozen three seeds")
    config = selection["candidate_config"]
    model = str(config.get("model", "adaptive_context_prototype"))
    features = Path(str(config.get("feature_root", args.features)))
    uses_lora = bool(config.get("adapter_path"))
    robustness_root = args.lora_robustness_features if uses_lora else args.robustness_features
    conditions: list[tuple[str, Path, Path, tuple[str, ...]]] = [
        ("standard", args.manifest, features, ()),
        *[(name, args.manifest, features, missing) for name, missing in MISSING_CONDITIONS.items()],
    ]
    if robustness_root is not None:
        conditions.extend(
            (
                (
                    "audio_snr_10db",
                    args.manifest,
                    robustness_root / "audio_snr_10db",
                    (),
                ),
                (
                    "video_frame_drop_50pct",
                    args.manifest,
                    robustness_root / "video_frame_drop_50pct",
                    (),
                ),
            )
        )
        if args.whisper_manifest is not None:
            conditions.append(
                (
                    "whisper_text",
                    args.whisper_manifest,
                    robustness_root / "whisper_text",
                    (),
                )
            )

    for condition, manifest, condition_features, missing_modalities in conditions:
        print(
            f"CONDITION {condition} manifest={manifest} features={condition_features} "
            f"missing={','.join(missing_modalities)}"
        )
        for seed in SEEDS:
            checkpoint = _checkpoint(args.formal_root, model, seed)
            print(f"EVALUATE seed={seed} checkpoint={checkpoint}")
    if args.dry_run:
        print("OFFICIAL_TEST_WILL_BE_CONSUMED_ON_EXECUTION")
        return 0

    checkpoints = [_checkpoint(args.formal_root, model, seed) for seed in SEEDS]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"formal checkpoints are missing: {missing}")
    for condition, manifest, condition_features, _ in conditions:
        if not manifest.is_file():
            raise FileNotFoundError(f"{condition} manifest is missing: {manifest}")
        for dataset in ("meld", "emotiontalk"):
            if not (condition_features / dataset / "test").is_dir():
                raise FileNotFoundError(
                    f"{condition} feature split is missing: {condition_features / dataset / 'test'}"
                )

    def evaluator() -> Path:
        args.output.mkdir(parents=True, exist_ok=True)
        for condition, manifest, condition_features, missing_modalities in conditions:
            for seed, checkpoint in zip(SEEDS, checkpoints, strict=True):
                evaluate_checkpoint(
                    manifest_path=manifest,
                    feature_root=condition_features,
                    checkpoint_path=checkpoint,
                    output_path=args.output / condition / f"seed-{seed}.json",
                    missing_modality=missing_modalities,
                    condition_name=condition if not missing_modalities else None,
                    bootstrap_iterations=args.bootstrap_iterations,
                    device_name=args.device,
                    evaluation_role="test",
                )
        return _summarize_test(
            args.output,
            tuple(condition for condition, *_ in conditions),
        )

    result = run_guarded_v4_test(
        args.selection,
        args.output / "TEST_EVALUATED",
        evaluator,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
