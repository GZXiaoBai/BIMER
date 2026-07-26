import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_v2_formal_results.py"
SEEDS = (42, 123, 2026)
FORMAL_VARIANTS = (
    "early_mlp",
    "early_context",
    "lagf_no_gates",
    "quality_lagf",
)
ABLATIONS = (
    "no_language",
    "no_gates",
    "no_context",
    "no_quality",
    "no_modality_dropout",
    "no_perturbation_training",
)


def _write_result(
    root: Path,
    *,
    scope: str,
    variant: str,
    seed: int,
    meld_weighted_f1: float,
    emotiontalk_weighted_f1: float,
) -> Path:
    model = "lagf" if variant == "lagf_no_gates" else "quality_lagf"
    if variant in {"early_mlp", "early_context"}:
        model = variant
    path = (
        root
        / scope
        / variant
        / model
        / "joint"
        / f"seed-{seed}"
        / "results.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": {
                    "model": model,
                    "seed": seed,
                    "learning_rate": 0.0001,
                    "training_scope": "joint",
                    "evaluate_test": True,
                },
                "evaluation_datasets": ["meld", "emotiontalk"],
                "history": {"best_epoch": 8, "best_score": 0.61},
                "test": {
                    "meld": {
                        "weighted_f1": meld_weighted_f1,
                        "macro_f1": meld_weighted_f1 - 0.1,
                        "accuracy": meld_weighted_f1 + 0.01,
                        "per_class_f1": {
                            "neutral": meld_weighted_f1 + 0.1,
                            "joy": meld_weighted_f1,
                        },
                    },
                    "emotiontalk": {
                        "weighted_f1": emotiontalk_weighted_f1,
                        "macro_f1": emotiontalk_weighted_f1 - 0.1,
                        "accuracy": emotiontalk_weighted_f1 + 0.01,
                        "per_class_f1": {
                            "neutral": emotiontalk_weighted_f1 + 0.1,
                            "joy": emotiontalk_weighted_f1,
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _build_complete_fixture(root: Path) -> None:
    seed_offsets = {42: 0.0, 123: 0.1, 2026: 0.2}
    formal_starts = {
        "early_mlp": 0.4,
        "early_context": 0.45,
        "lagf_no_gates": 0.48,
        "quality_lagf": 0.5,
    }
    for variant, start in formal_starts.items():
        for seed in SEEDS:
            meld = start + seed_offsets[seed]
            _write_result(
                root,
                scope="formal",
                variant=variant,
                seed=seed,
                meld_weighted_f1=meld,
                emotiontalk_weighted_f1=meld + 0.1,
            )
    ablation_starts = {
        "no_language": 0.45,
        "no_gates": 0.47,
        "no_context": 0.4,
        "no_quality": 0.46,
        "no_modality_dropout": 0.48,
        "no_perturbation_training": 0.44,
    }
    for variant, start in ablation_starts.items():
        for seed in SEEDS:
            meld = start + seed_offsets[seed]
            _write_result(
                root,
                scope="ablations",
                variant=variant,
                seed=seed,
                meld_weighted_f1=meld,
                emotiontalk_weighted_f1=meld + 0.1,
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_cli_writes_sample_standard_deviation_and_ablation_deltas(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "summary"
    _build_complete_fixture(results)

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(results),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    formal = _read_csv(output / "formal_summary.csv")
    quality = next(
        row
        for row in formal
        if row["variant"] == "quality_lagf"
        and row["dataset"] == "bilingual_average"
    )
    assert float(quality["weighted_f1_mean"]) == 0.65
    assert float(quality["weighted_f1_std"]) == 0.1
    ablations = _read_csv(output / "ablation_summary.csv")
    no_language = next(
        row
        for row in ablations
        if row["variant"] == "no_language"
        and row["dataset"] == "bilingual_average"
    )
    assert float(no_language["weighted_f1_mean"]) == 0.6
    assert float(no_language["weighted_f1_delta_vs_full"]) == -0.05
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation"]["formal_result_count"] == 12
    assert summary["validation"]["ablation_result_count"] == 18
    assert summary["methodology"]["standard_deviation_ddof"] == 1


def test_cli_rejects_an_incomplete_three_seed_matrix(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "summary"
    _build_complete_fixture(results)
    missing = next(
        (results / "ablations" / "no_quality").glob(
            "**/seed-2026/results.json"
        )
    )
    missing.unlink()

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(results),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert run.returncode != 0
    assert "missing results" in run.stderr.lower()
