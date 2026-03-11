from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: Dict[str, Any] = {}
            for k, v in row.items():
                if k in {"amp"}:
                    rec[k] = json.loads(v)
                elif k in {"amp_fft"}:
                    rec[k] = json.loads(v)
                elif k in {
                    "N",
                    "drive_index",
                    "seed",
                    "target_idx",
                    "target_nd_idx",
                    "target_fft_idx",
                    "target_fft_nd_idx",
                }:
                    rec[k] = int(float(v))
                elif k in {
                    "gamma",
                    "K",
                    "F",
                    "Omega",
                    "w0",
                    "detune_eps",
                    "discard_ratio",
                    "t_total",
                    "dt",
                    "amp_max",
                    "amp_second",
                    "amp_nd_max",
                    "amp_nd_second",
                    "selectivity",
                    "selectivity_nd",
                    "amp_fft_max",
                    "amp_fft_second",
                    "amp_fft_nd_max",
                    "amp_fft_nd_second",
                    "selectivity_fft",
                    "selectivity_fft_nd",
                }:
                    rec[k] = float(v)
                else:
                    rec[k] = v
            records.append(rec)
    return records


def plot_heatmap(records: List[Dict[str, Any]], metric: str, out_path: Path) -> None:
    omegas = sorted({r["Omega"] for r in records})
    ks = sorted({r["K"] for r in records})
    grid = np.full((len(ks), len(omegas)), np.nan)
    for r in records:
        i = ks.index(r["K"])
        j = omegas.index(r["Omega"])
        grid[i, j] = r[metric]

    plt.figure(figsize=(6, 4))
    im = plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(omegas), max(omegas), min(ks), max(ks)],
        cmap="viridis",
    )
    plt.colorbar(im, label=metric)
    plt.xlabel("Omega")
    plt.ylabel("K")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_amplitude_bar(record: Dict[str, Any], out_path: Path) -> None:
    amps = record["amp"]
    plt.figure(figsize=(4, 3))
    plt.bar(range(len(amps)), amps, color="tab:blue")
    plt.xlabel("node")
    plt.ylabel("RMS amplitude")
    plt.title(f"Omega={record['Omega']:.3f}, K={record['K']:.3f}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze and plot scan summaries.")
    p.add_argument("--summary_path", type=Path, default=Path("data/summary/grid.csv"))
    p.add_argument("--metric", type=str, default="selectivity", help="Metric for heatmap.")
    p.add_argument("--top", type=int, default=3, help="Number of top runs to bar-plot.")
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_summary(args.summary_path)
    if not records:
        raise ValueError("No records loaded from summary.")

    plot_heatmap(records, args.metric, args.out_dir / f"heatmap_{args.metric}.png")

    top_records = sorted(records, key=lambda r: r.get(args.metric, 0), reverse=True)[: args.top]
    for idx, rec in enumerate(top_records):
        plot_amplitude_bar(rec, args.out_dir / f"top{idx+1}_Omega{rec['Omega']:.3f}_K{rec['K']:.3f}.png")

    print(f"plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
