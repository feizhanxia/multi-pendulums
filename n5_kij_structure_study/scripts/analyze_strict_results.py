#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.feature_analysis import run_analysis


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Analyze large-batch strict-search outputs.")
    parser.add_argument("--json_path", type=Path, default=root / "results" / "data" / "n5_strict_results.json")
    parser.add_argument("--out_root", type=Path, default=root / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(args.json_path, args.out_root)
    counts = result["summary"]["counts"]
    print(f"json: {args.json_path}")
    print(f"analysis_out: {args.out_root}")
    print(f"counts: {counts}")
    if result["feature_scores"]:
        name, score = result["feature_scores"][0]
        print(f"top_feature: {name} ({score:.3f})")


if __name__ == "__main__":
    main()
