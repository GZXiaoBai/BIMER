#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

SEEDS = (42, 123, 2026)
MODELS = ("text", "audio", "vision")
DATASETS = ("meld", "emotiontalk")
METRICS = ("weighted_f1", "macro_f1", "accuracy")


def _predicted_class_count(confusion_matrix: list[list[int]]) -> int:
    class_count = len(confusion_matrix)
    return sum(
        sum(int(confusion_matrix[row][column]) for row in range(class_count)) > 0
        for column in range(class_count)
    )


def summarize_unimodal_results(
    root: Path,
    *,
    models: tuple[str, ...] = MODELS,
    datasets: tuple[str, ...] = DATASETS,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        for dataset in datasets:
            payloads = [
                json.loads(
                    (root / model / model / dataset / f"seed-{seed}" / "results.json").read_text(
                        encoding="utf-8"
                    )
                )
                for seed in seeds
            ]
            learning_rates = {float(payload["config"]["learning_rate"]) for payload in payloads}
            if len(learning_rates) != 1:
                raise ValueError(f"{model}/{dataset} uses inconsistent learning rates")
            row: dict[str, object] = {
                "model": model,
                "dataset": dataset,
                "seeds": "|".join(str(seed) for seed in seeds),
                "learning_rate": learning_rates.pop(),
            }
            for metric in METRICS:
                values = [float(payload["test"][dataset][metric]) for payload in payloads]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
            row["predicted_class_count_min"] = min(
                _predicted_class_count(payload["test"][dataset]["confusion_matrix"])
                for payload in payloads
            )
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize_unimodal_results(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
