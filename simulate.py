from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from model import Params, initial_conditions, pendulum_ode, rms_amplitude


def run_simulation(
    params: Params,
    save_raw: bool = False,
    raw_path: Path | None = None,
    plot_path: Path | None = None,
) -> Dict[str, Any]:
    # Ensure t_eval within [0, t_total]
    t_eval = np.arange(0.0, params.t_total, params.dt)
    if t_eval[-1] < params.t_total:
        t_eval = np.append(t_eval, params.t_total)
    y0 = initial_conditions(params)

    sol = solve_ivp(
        lambda t, y: pendulum_ode(t, y, params),
        (0.0, params.t_total),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )

    theta = sol.y[: params.N].T  # shape (time, N)
    discard_idx = int(len(t_eval) * params.discard_ratio)
    theta_tail = theta[discard_idx:]
    t_tail = t_eval[discard_idx:]

    amp = rms_amplitude(theta_tail)
    max_idx = int(np.argmax(amp))
    second_amp = np.partition(amp, -2)[-2] if params.N > 1 else amp[0]
    selectivity = float(amp[max_idx] / (second_amp + 1e-12))

    # Selectivity excluding the driven node.
    non_drive_idx = [i for i in range(params.N) if i != params.drive_index]
    if non_drive_idx:
        amp_non = amp[non_drive_idx]
        nd_max_local = int(np.argmax(amp_non))
        nd_max_idx = int(non_drive_idx[nd_max_local])
        nd_second = np.partition(amp_non, -2)[-2] if len(amp_non) > 1 else amp_non[0]
        selectivity_nd = float(amp_non[nd_max_local] / (nd_second + 1e-12))
        amp_nd_max = float(amp_non[nd_max_local])
    else:
        nd_max_idx = -1
        nd_second = 0.0
        selectivity_nd = 0.0
        amp_nd_max = 0.0

    # FFT-like amplitude at driving frequency for steady-state tail.
    phase = np.exp(-1j * params.Omega * t_tail)
    amp_fft = np.abs(np.mean(theta_tail * phase[:, None], axis=0))
    fft_max_idx = int(np.argmax(amp_fft))
    fft_second = np.partition(amp_fft, -2)[-2] if params.N > 1 else amp_fft[0]
    selectivity_fft = float(amp_fft[fft_max_idx] / (fft_second + 1e-12))

    if non_drive_idx:
        amp_fft_nd = amp_fft[non_drive_idx]
        fft_nd_local = int(np.argmax(amp_fft_nd))
        fft_nd_idx = int(non_drive_idx[fft_nd_local])
        fft_nd_second = np.partition(amp_fft_nd, -2)[-2] if len(amp_fft_nd) > 1 else amp_fft_nd[0]
        selectivity_fft_nd = float(amp_fft_nd[fft_nd_local] / (fft_nd_second + 1e-12))
        amp_fft_nd_max = float(amp_fft_nd[fft_nd_local])
    else:
        fft_nd_idx = -1
        fft_nd_second = 0.0
        selectivity_fft_nd = 0.0
        amp_fft_nd_max = 0.0

    summary = {
        "N": params.N,
        "gamma": params.gamma,
        "K": params.K,
        "F": params.F,
        "Omega": params.Omega,
        "w0": params.w0,
        "detune_eps": params.detune_eps,
        "detune_target": params.detune_target,
        "discard_ratio": params.discard_ratio,
        "t_total": params.t_total,
        "dt": params.dt,
        "seed": params.seed,
        "drive_index": params.drive_index,
        "amp": amp.tolist(),
        "amp_max": float(amp[max_idx]),
        "amp_second": float(second_amp),
        "selectivity": selectivity,
        "target_idx": max_idx,
        "amp_nd_max": amp_nd_max,
        "amp_nd_second": float(nd_second),
        "selectivity_nd": selectivity_nd,
        "target_nd_idx": nd_max_idx,
        "amp_fft": amp_fft.tolist(),
        "amp_fft_max": float(amp_fft[fft_max_idx]),
        "amp_fft_second": float(fft_second),
        "selectivity_fft": selectivity_fft,
        "target_fft_idx": fft_max_idx,
        "amp_fft_nd_max": amp_fft_nd_max,
        "amp_fft_nd_second": float(fft_nd_second),
        "selectivity_fft_nd": selectivity_fft_nd,
        "target_fft_nd_idx": fft_nd_idx,
        "t_eval": [float(t_eval[0]), float(t_eval[-1]), float(params.dt)],
        "status": "ok" if sol.success else "fail",
        "message": sol.message,
    }

    if save_raw and raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_path,
            t=t_eval,
            theta=theta,
            params=json.dumps(summary, ensure_ascii=False),
        )

    if plot_path is not None:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(20, 4), dpi=400)
        for idx in range(params.N):
            ax.plot(t_eval, theta[:, idx], label=f"node {idx}", linewidth=0.6)
        ax.set_xlabel("time")
        ax.set_ylabel("theta")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"N={params.N}, K={params.K}, Omega={params.Omega}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        summary["plot_path"] = str(plot_path)

    return summary


def append_summary(summary_path: Path, record: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not summary_path.exists()
    serializable = {
        k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
        for k, v in record.items()
    }
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if header_needed:
            writer.writeheader()
        writer.writerow(serializable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single driven coupled pendulum simulation.")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.08)
    p.add_argument("--K", type=float, default=0.4)
    p.add_argument("--F", type=float, default=0.1)
    p.add_argument("--Omega", type=float, default=0.95)
    p.add_argument("--w0", type=float, default=1.0)
    p.add_argument("--detune_eps", type=float, default=0.0, help="Small frequency gradient to break degeneracy.")
    p.add_argument(
        "--detune_target",
        type=int,
        default=-1,
        help="-1 for gradient detune; >=0 to detune a single node.",
    )
    p.add_argument("--discard_ratio", type=float, default=0.5)
    p.add_argument("--t_total", type=float, default=2000.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drive_index", type=int, default=0)
    p.add_argument("--save_raw", action="store_true", help="Save raw time series to npz.")
    p.add_argument("--raw_path", type=Path, default=None, help="Path to raw npz.")
    p.add_argument("--plot_path", type=Path, default=None, help="Optional path to save theta(t) plot.")
    p.add_argument("--summary_path", type=Path, default=Path("data/summary/single.csv"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    params = Params(
        N=args.N,
        gamma=args.gamma,
        K=args.K,
        F=args.F,
        Omega=args.Omega,
        w0=args.w0,
        detune_eps=args.detune_eps,
        detune_target=args.detune_target,
        discard_ratio=args.discard_ratio,
        t_total=args.t_total,
        dt=args.dt,
        seed=args.seed,
        drive_index=args.drive_index,
    )

    if args.drive_index < 0 or args.drive_index >= params.N:
        raise ValueError(f"drive_index must be in [0, {params.N - 1}]")

    raw_path = args.raw_path
    if args.save_raw:
        if raw_path is None:
            raw_path = Path(f"data/raw/run_N{params.N}_K{params.K}_O{params.Omega}_seed{params.seed}.npz")
        else:
            raw_path = Path(raw_path)

    summary = run_simulation(params, save_raw=args.save_raw, raw_path=raw_path, plot_path=args.plot_path)
    append_summary(Path(args.summary_path), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.save_raw:
        print(f"raw saved to {raw_path}")
    print(f"summary appended to {args.summary_path}")


if __name__ == "__main__":
    main()
