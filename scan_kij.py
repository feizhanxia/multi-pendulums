from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from model_kij import ParamsKij
from simulate_kij import run_simulation


def _frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    v = start
    while v <= stop + 1e-12:
        vals.append(round(v, 10))
        v += step
    return vals


def _parse_kij(values: list[float], n: int) -> np.ndarray:
    if len(values) != n * n:
        raise ValueError(f"--K requires {n*n} values (row-major).")
    K = np.array(values, dtype=float).reshape(n, n)
    np.fill_diagonal(K, 0.0)
    return K


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan Omega for fixed K_ij (N=3).")
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.08)
    p.add_argument("--F", type=float, default=0.1)
    p.add_argument("--w0", type=float, default=1.0)
    p.add_argument("--K", type=float, nargs="+", required=True, help="K_ij row-major (N*N values).")
    p.add_argument("--t_total", type=float, default=1200.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--discard_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drive_index", type=int, default=0)
    p.add_argument("--omega_range", type=float, nargs=3, default=[0.6, 1.4, 0.01], metavar=("START", "STOP", "STEP"))
    p.add_argument("--summary_path", type=Path, default=Path("data/summary/scan_kij.csv"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    K = _parse_kij(args.K, args.N)
    omega_vals = _frange(*args.omega_range)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for idx, omega in enumerate(omega_vals):
        params = ParamsKij(
            N=args.N,
            gamma=args.gamma,
            F=args.F,
            Omega=omega,
            w0=args.w0,
            K=K,
            discard_ratio=args.discard_ratio,
            t_total=args.t_total,
            dt=args.dt,
            seed=args.seed + idx,
            drive_index=args.drive_index,
        )
        records.append(run_simulation(params))

    header_needed = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        if header_needed:
            writer.writeheader()
        for rec in records:
            serializable = {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                for k, v in rec.items()
            }
            writer.writerow(serializable)

    print(f"completed {len(records)} runs; summary -> {summary_path}")


if __name__ == "__main__":
    main()
