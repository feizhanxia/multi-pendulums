from __future__ import annotations

import argparse
import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from model import Params
from simulate import run_simulation


def build_grid(omega_range: Tuple[float, float, float], k_range: Tuple[float, float, float]) -> List[Tuple[float, float]]:
    omega_vals = _frange(*omega_range)
    k_vals = _frange(*k_range)
    return list(itertools.product(omega_vals, k_vals))


def _frange(start: float, stop: float, step: float):
    vals = []
    v = start
    while v <= stop + 1e-12:
        vals.append(round(v, 10))
        v += step
    return vals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter grid scan for driven coupled pendulums.")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.08)
    p.add_argument("--F", type=float, default=0.1)
    p.add_argument("--w0", type=float, default=1.0)
    p.add_argument("--detune_eps", type=float, default=0.0, help="Small frequency gradient to break degeneracy.")
    p.add_argument(
        "--detune_target",
        type=int,
        default=-1,
        help="-1 for gradient detune; >=0 to detune a single node.",
    )
    p.add_argument("--t_total", type=float, default=1200.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--discard_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drive_index", type=int, default=0)
    p.add_argument("--omega_range", type=float, nargs=3, default=[0.5, 1.5, 0.02], metavar=("START", "STOP", "STEP"))
    p.add_argument("--k_range", type=float, nargs=3, default=[0.0, 1.0, 0.05], metavar=("START", "STOP", "STEP"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--raw_dir", type=Path, default=Path("data/raw/grid"))
    p.add_argument("--summary_path", type=Path, default=Path("data/summary/grid.csv"))
    return p.parse_args(argv)


def run_task_for_scan(payload: Dict[str, object]) -> Dict[str, object]:
    """Top-level helper for multiprocessing pickling."""
    params = Params(
        N=int(payload["N"]),
        gamma=float(payload["gamma"]),
        K=float(payload["K"]),
        F=float(payload["F"]),
        Omega=float(payload["Omega"]),
        w0=float(payload["w0"]),
        detune_eps=float(payload["detune_eps"]),
        detune_target=int(payload["detune_target"]),
        t_total=float(payload["t_total"]),
        dt=float(payload["dt"]),
        discard_ratio=float(payload["discard_ratio"]),
        seed=int(payload["seed"]),
        drive_index=int(payload["drive_index"]),
    )
    raw_path = payload.get("raw_path")
    return run_simulation(params, save_raw=bool(payload["save_raw"]), raw_path=Path(raw_path) if raw_path else None)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    grid = build_grid(tuple(args.omega_range), tuple(args.k_range))
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, object]] = []
    for idx, (omega, k_val) in enumerate(grid):
        raw_path = None
        if args.save_raw:
            raw_path = (
                Path(args.raw_dir) / f"raw_N{args.N}_K{k_val:.4f}_O{omega:.4f}_seed{args.seed+idx}.npz"
            )
        tasks.append(
            {
                "N": args.N,
                "gamma": args.gamma,
                "F": args.F,
                "w0": args.w0,
                "detune_eps": args.detune_eps,
                "detune_target": args.detune_target,
                "t_total": args.t_total,
                "dt": args.dt,
                "discard_ratio": args.discard_ratio,
                "seed": args.seed + idx,
                "drive_index": args.drive_index,
                "Omega": omega,
                "K": k_val,
                "save_raw": args.save_raw,
                "raw_path": str(raw_path) if raw_path else None,
            }
        )

    if args.workers == 1:
        records = [run_task_for_scan(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_idx = {ex.submit(run_task_for_scan, t): idx for idx, t in enumerate(tasks)}
            records = []
            for fut in as_completed(fut_to_idx):
                records.append(fut.result())

    # Write all records
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
