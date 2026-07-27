#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bimer.calibration import apply_temperature, fit_calibration_profile


def _load(directory: Path):
    rows = []
    for dataset, default_language in (("meld", "en"), ("emotiontalk", "zh")):
        path = directory / f"{dataset}.npz"
        with np.load(path, allow_pickle=False) as payload:
            truth = payload["truth"].astype(np.int64)
            rows.append(
                (
                    payload["probabilities"].astype(np.float64),
                    truth,
                    (
                        payload["languages"].astype(str)
                        if "languages" in payload.files
                        else np.full(len(truth), default_language)
                    ),
                )
            )
    return tuple(np.concatenate(values) for values in zip(*rows, strict=True))


def _reliability_points(probabilities, truth, bins=15):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    points = []
    for lower, upper in zip(
        np.linspace(0, 1, bins + 1)[:-1],
        np.linspace(0, 1, bins + 1)[1:],
        strict=True,
    ):
        active = (confidence >= lower) & (confidence <= upper)
        if active.any():
            points.append((float(confidence[active].mean()), float(correct[active].mean())))
    return np.asarray(points, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    probabilities, truth, languages = _load(args.predictions)
    profile = fit_calibration_profile(probabilities, truth, languages)
    profile.save(args.profile)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                **profile.to_dict(),
                "selection_rules": {
                    "ece_relative_reduction": 0.10,
                    "nll_may_worsen": False,
                    "minimum_coverage": 0.70,
                    "minimum_selective_accuracy_gain": 0.03,
                },
                "source": str(args.predictions),
                "split": "validation",
                "test_set_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, language in zip(axes, ("en", "zh"), strict=True):
        active = languages == language
        raw = _reliability_points(probabilities[active], truth[active])
        calibrated_values = apply_temperature(
            probabilities[active],
            profile.languages[language].temperature,
        )
        calibrated = _reliability_points(calibrated_values, truth[active])
        axis.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
        if len(raw):
            axis.plot(raw[:, 0], raw[:, 1], "o-", label="raw")
        if len(calibrated):
            axis.plot(
                calibrated[:, 0],
                calibrated[:, 1],
                "o-",
                label="calibrated",
            )
        axis.set_title(language)
        axis.set_xlabel("confidence")
        axis.set_ylabel("accuracy")
        axis.legend()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=160)
    plt.close(figure)
    print(args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
