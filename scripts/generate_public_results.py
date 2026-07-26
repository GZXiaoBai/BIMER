#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MODEL_NAMES = {
    "early_mlp": "Early MLP",
    "early_context": "Early Context",
    "lagf_no_gates": "No-gate Context",
    "quality_lagf": "V2 Quality LAGF",
}
DATASET_NAMES = {
    "meld": "MELD",
    "emotiontalk": "EmotionTalk",
    "bilingual_average": "Bilingual average",
}
ABLATION_NAMES = {
    "no_language": "Language embedding",
    "no_gates": "Reliability gates",
    "no_context": "Dialogue context",
    "no_quality": "Quality input",
    "no_modality_dropout": "Modality dropout",
    "no_perturbation_training": "Corruption training",
}
CONDITION_NAMES = {
    "standard": "Clean",
    "audio_snr_20db": "Audio 20 dB",
    "audio_snr_10db": "Audio 10 dB",
    "video_frame_drop_25pct": "Video drop 25%",
    "video_frame_drop_50pct": "Video drop 50%",
    "whisper_text": "Whisper text",
    "missing-text": "Missing text",
    "missing-audio": "Missing audio",
    "missing-vision": "Missing vision",
}


def v3_public_summary(
    loss_decision: dict[str, Any],
    ranking_decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "validation-only predeclared screening",
        "loss_screen": {
            "selected": loss_decision["selected"],
            "passed": loss_decision["passed"],
            "diagnostics": loss_decision["diagnostics"],
        },
        "ranking_screen": {
            "selected_lambda": ranking_decision["selected"],
            "passed": ranking_decision["passed"],
            "diagnostics": ranking_decision["diagnostics"],
        },
        "decision": "stop_v3_and_deploy_v2",
        "formal_v3_test_run": False,
    }


def select_public_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    conditions = list(CONDITION_NAMES)
    selected = frame[
        frame["model"].isin(["quality_lagf", "no_gates"])
        & (frame["dataset"] == "bilingual_average")
        & frame["condition"].isin(conditions)
    ].copy()
    order = {condition: index for index, condition in enumerate(conditions)}
    selected["condition_order"] = selected["condition"].map(order)
    selected = selected.sort_values(["model", "condition_order"])
    return selected[
        [
            "model",
            "condition",
            "runs",
            "weighted_f1_mean",
            "weighted_f1_std",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "accuracy_std",
            "delta_from_standard",
        ]
    ].reset_index(drop=True)


def select_public_per_class(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[(frame["scope"] == "formal") & (frame["variant"] == "quality_lagf")].copy()
    return selected.sort_values(["dataset", "label"]).reset_index(drop=True)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "BIMER publication pipeline"},
    )
    plt.close(figure)


def plot_main_results(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["dataset"].isin(DATASET_NAMES)].copy()
    variants = list(MODEL_NAMES)
    datasets = list(DATASET_NAMES)
    x = np.arange(len(datasets))
    width = 0.19
    colors = ["#94a3b8", "#60a5fa", "#a78bfa", "#10b981"]
    figure, axis = plt.subplots(figsize=(10, 5.4))
    for index, variant in enumerate(variants):
        block = selected[selected["variant"] == variant].set_index("dataset").loc[datasets]
        axis.bar(
            x + (index - 1.5) * width,
            block["weighted_f1_mean"] * 100,
            width,
            yerr=block["weighted_f1_std"] * 100,
            capsize=3,
            color=colors[index],
            label=MODEL_NAMES[variant],
        )
    axis.set_title("Formal weighted-F1 by dataset (three seeds)")
    axis.set_ylabel("weighted-F1 (%)")
    axis.set_xticks(x, [DATASET_NAMES[value] for value in datasets])
    axis.set_ylim(55, 64)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    figure.tight_layout()
    _save(figure, output)


def plot_ablation(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["dataset"] == "bilingual_average"].copy()
    variants = list(ABLATION_NAMES)
    selected = selected.set_index("variant").loc[variants]
    contribution = -selected["weighted_f1_delta_vs_full"].to_numpy() * 100
    colors = ["#10b981" if value > 0 else "#fb7185" for value in contribution]
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    positions = np.arange(len(variants))
    axis.barh(positions, contribution, color=colors)
    axis.axvline(0, color="#334155", linewidth=1)
    axis.set_yticks(positions, [ABLATION_NAMES[value] for value in variants])
    axis.invert_yaxis()
    axis.set_xlabel("Complete model minus ablation (weighted-F1 points)")
    axis.set_title("Bilingual ablation effects")
    for position, value in zip(positions, contribution, strict=True):
        offset = 0.03 if value >= 0 else 0.015
        axis.text(
            value + offset,
            position,
            f"{value:+.3f}",
            va="center",
            ha="left",
        )
    figure.tight_layout()
    _save(figure, output)


def plot_robustness(frame: pd.DataFrame, output: Path) -> None:
    conditions = [
        "standard",
        "audio_snr_20db",
        "audio_snr_10db",
        "video_frame_drop_25pct",
        "video_frame_drop_50pct",
        "whisper_text",
        "missing-audio",
        "missing-vision",
        "missing-text",
    ]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for model, color, marker in [
        ("quality_lagf", "#10b981", "o"),
        ("no_gates", "#7c3aed", "s"),
    ]:
        block = frame[(frame["model"] == model) & frame["condition"].isin(conditions)]
        block = block.set_index("condition").loc[conditions]
        label = "V2 Quality LAGF" if model == "quality_lagf" else "No-gate Context"
        axis.errorbar(
            np.arange(len(conditions)),
            block["weighted_f1_mean"] * 100,
            yerr=block["weighted_f1_std"] * 100,
            marker=marker,
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )
    axis.set_title("Bilingual robustness by input condition")
    axis.set_ylabel("weighted-F1 (%)")
    axis.set_xticks(
        np.arange(len(conditions)),
        [CONDITION_NAMES[value] for value in conditions],
        rotation=24,
        ha="right",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output)


def plot_per_class(frame: pd.DataFrame, output: Path) -> None:
    labels = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.5, 5.3))
    for index, (dataset, color) in enumerate([("meld", "#60a5fa"), ("emotiontalk", "#f59e0b")]):
        block = frame[frame["dataset"] == dataset].set_index("label").loc[labels]
        axis.bar(
            x + (index - 0.5) * width,
            block["f1_mean"] * 100,
            width,
            yerr=block["f1_std"] * 100,
            capsize=3,
            color=color,
            label=DATASET_NAMES[dataset],
        )
    axis.set_title("V2 Quality LAGF per-class F1")
    axis.set_ylabel("F1 (%)")
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylim(0, 82)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output)


def export_public_results(
    *,
    analysis_root: Path,
    v3_screen_root: Path,
    results_root: Path,
    figures_root: Path,
) -> None:
    formal_root = analysis_root / "v2-formal-ablations"
    robustness_root = analysis_root / "v2-robustness"
    results_root.mkdir(parents=True, exist_ok=True)

    formal = pd.read_csv(formal_root / "formal_summary.csv")
    ablations = pd.read_csv(formal_root / "ablation_summary.csv")
    bootstrap = pd.read_csv(formal_root / "paired_cluster_bootstrap.csv")
    per_class = select_public_per_class(pd.read_csv(formal_root / "per_class_f1_summary.csv"))
    robustness = select_public_robustness(
        pd.read_csv(robustness_root / "model-condition-summary.csv")
    )

    formal.to_csv(results_root / "v2_model_summary.csv", index=False)
    ablations.to_csv(results_root / "v2_ablation_summary.csv", index=False)
    bootstrap.to_csv(results_root / "v2_bootstrap_summary.csv", index=False)
    per_class.to_csv(results_root / "v2_per_class_summary.csv", index=False)
    robustness.to_csv(results_root / "v2_robustness_summary.csv", index=False)

    data_audit = json.loads((analysis_root / "v2-data-audit.json").read_text())
    data_audit.pop("manifest", None)
    (results_root / "data_audit.json").write_text(
        json.dumps(data_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loss_decision = json.loads((v3_screen_root / "loss-decision.json").read_text())
    ranking_decision = json.loads((v3_screen_root / "ranking-decision.json").read_text())
    (results_root / "v3_screening_summary.json").write_text(
        json.dumps(
            v3_public_summary(loss_decision, ranking_decision),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _style()
    plot_main_results(formal, figures_root / "main_results.png")
    plot_ablation(ablations, figures_root / "ablation_effects.png")
    plot_robustness(robustness, figures_root / "robustness_comparison.png")
    plot_per_class(per_class, figures_root / "per_class_f1.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("artifacts/analysis"))
    parser.add_argument(
        "--v3-screen-root",
        type=Path,
        default=Path("artifacts/experiments/v3/screen"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--figures-root", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    export_public_results(
        analysis_root=args.analysis_root,
        v3_screen_root=args.v3_screen_root,
        results_root=args.results_root,
        figures_root=args.figures_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
