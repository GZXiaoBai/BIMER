#!/usr/bin/env python3
import argparse

from bimer.prediction_comparison import compare_prediction_archives


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

print(
    compare_prediction_archives(
        args.baseline,
        args.candidate,
        args.output,
        iterations=args.iterations,
        seed=args.seed,
    )
)
