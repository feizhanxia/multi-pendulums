from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: Dict[str, Any] = {}
            for k, v in row.items():
                if k in {"amp_fft", "K"}:
                    rec[k] = json.loads(v)
                elif k in {"N", "drive_index", "seed", "target_fft_nd_idx"}:
                    rec[k] = int(float(v))
                elif k in {
                    "gamma",
                    "F",
                    "Omega",
                    "w0",
                    "discard_ratio",
                    "t_total",
                    "dt",
                    "amp_fft_nd_max",
                    "amp_fft_nd_second",
                    "selectivity_fft_nd",
                }:
                    rec[k] = float(v)
                else:
                    rec[k] = v
            records.append(rec)
    return records


def _window_scores(
    omegas: np.ndarray,
    winners: np.ndarray,
    selectivity: np.ndarray,
    target_node: int,
    window_size: int,
) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []
    for start in range(0, len(omegas) - window_size + 1):
        end = start + window_size - 1
        win_ratio = float(np.mean(winners[start : end + 1] == target_node))
        avg_sel = float(np.mean(selectivity[start : end + 1]))
        score = win_ratio * avg_sel
        scores.append(
            {
                "start": start,
                "end": end,
                "omega_start": float(omegas[start]),
                "omega_end": float(omegas[end]),
                "win_ratio": win_ratio,
                "avg_selectivity": avg_sel,
                "score": score,
            }
        )
    return scores


def find_best_band_pair(
    omegas: np.ndarray,
    winners: np.ndarray,
    selectivity: np.ndarray,
    node_a: int,
    node_b: int,
    window_size: int,
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    scores_a = _window_scores(omegas, winners, selectivity, node_a, window_size)
    scores_b = _window_scores(omegas, winners, selectivity, node_b, window_size)

    best_pair = None
    best_sum = -1.0
    for wa in scores_a:
        for wb in scores_b:
            non_overlap = wa["end"] < wb["start"] or wb["end"] < wa["start"]
            if not non_overlap:
                continue
            score_sum = np.sqrt(wa["win_ratio"] * wb["win_ratio"]) * (
                (wa["avg_selectivity"] + wb["avg_selectivity"]) / 2.0
            )
            if score_sum > best_sum:
                best_sum = score_sum
                best_pair = (wa, wb)

    if best_pair is not None:
        return best_pair[0], best_pair[1], True

    # Fallback: allow overlap if no non-overlap pair exists.
    top_a = max(scores_a, key=lambda x: x["score"])
    top_b = max(scores_b, key=lambda x: x["score"])
    return top_a, top_b, False


def plot_winners(
    omegas: np.ndarray,
    winners: np.ndarray,
    selectivity: np.ndarray,
    node_a: int,
    node_b: int,
    out_path: Path,
) -> None:
    colors = np.where(winners == node_a, "tab:blue", "tab:orange")
    plt.figure(figsize=(7, 3.5))
    plt.scatter(omegas, selectivity, c=colors, s=12, alpha=0.8)
    plt.xlabel("Omega")
    plt.ylabel("selectivity (main-frequency top-1/top-2)")
    plt.title(f"Winner: node {node_a} (blue) vs node {node_b} (orange)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze K_ij scan and auto-find two frequency bands.")
    p.add_argument("--summary_path", type=Path, default=Path("data/summary/scan_kij.csv"))
    p.add_argument("--node_a", type=int, default=1)
    p.add_argument("--node_b", type=int, default=2)
    p.add_argument("--window_size", type=int, default=15)
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_summary(args.summary_path)
    if not records:
        raise ValueError("No records loaded from summary.")

    records = sorted(records, key=lambda r: r["Omega"])
    omegas = np.array([r["Omega"] for r in records], dtype=float)
    amps = np.array([r["amp_fft"] for r in records], dtype=float)

    if amps.shape[1] <= max(args.node_a, args.node_b):
        raise ValueError("node_a/node_b out of range for amp array.")

    amp_a = amps[:, args.node_a]
    amp_b = amps[:, args.node_b]
    winners = np.where(amp_a >= amp_b, args.node_a, args.node_b)
    selectivity = np.maximum(amp_a, amp_b) / (np.minimum(amp_a, amp_b) + 1e-12)

    win_a, win_b, non_overlap = find_best_band_pair(
        omegas, winners, selectivity, args.node_a, args.node_b, args.window_size
    )

    report = {
        "summary_path": str(args.summary_path),
        "node_a": args.node_a,
        "node_b": args.node_b,
        "window_size": args.window_size,
        "band_a": win_a,
        "band_b": win_b,
        "non_overlap": non_overlap,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "band_report_kij.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_path = out_dir / "winner_selectivity_kij.png"
    plot_winners(omegas, winners, selectivity, args.node_a, args.node_b, plot_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report saved to {report_path}")
    print(f"plot saved to {plot_path}")


if __name__ == "__main__":
    main()
