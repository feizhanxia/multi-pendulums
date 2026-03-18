#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from src.model_kij import ParamsKij, initial_conditions, pendulum_ode_kij
from src.simulate_kij import run_simulation


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Plot verification figures for all stable final candidates.")
    p.add_argument("--json_path", type=Path, default=root / "results" / "data" / "n5_strict_results.json")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=root / "results" / "figures" / "stable_high_verifications",
        help="Directory used to save one verification figure per stable candidate.",
    )
    p.add_argument("--zoom_half_width", type=float, default=0.2, help="Half-width around best omega for zoom panel.")
    p.add_argument("--theta_seed", type=int, default=None, help="Seed for theta(t) panel; default uses first final seed.")
    p.add_argument("--workers", type=int, default=10, help="Number of worker processes for parallel plotting.")
    return p.parse_args()


def simulate_theta(params: ParamsKij) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, params.t_total, params.dt)
    if t_eval[-1] < params.t_total:
        t_eval = np.append(t_eval, params.t_total)
    y0 = initial_conditions(params)
    sol = solve_ivp(
        lambda t, y: pendulum_ode_kij(t, y, params),
        (0.0, params.t_total),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    theta = sol.y[: params.N].T
    return t_eval, theta


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    return f"{mins:02d}m{secs:02d}s"


def _print_progress(done: int, total: int, start_time: float, extra: str = "") -> None:
    if total <= 0:
        print("[plot] 0/0")
        return
    elapsed = time.perf_counter() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 1e-12 else 0.0
    message = (
        f"\r[plot] {done}/{total} "
        f"({100.0 * done / total:5.1f}%) "
        f"elapsed={_format_seconds(elapsed)} "
        f"eta={_format_seconds(eta)}"
    )
    if extra:
        message += f" | {extra}"
    print(message, end="" if done < total else "\n", flush=True)


def plot_candidate(
    candidate: dict,
    cfg: dict,
    out_path: Path,
    zoom_half_width: float,
    theta_seed_override: int | None,
) -> None:
    use_target_band = bool(cfg.get("use_target_band", False))
    K = np.array(candidate["K"], dtype=float)
    omega_selected = float(candidate["omega"])
    seed_results = {int(r["seed"]): r for r in candidate.get("seed_results", [])}

    n_nodes = int(cfg.get("N", K.shape[0]))
    gamma = float(cfg.get("gamma", 0.08))
    force = float(cfg.get("F", 0.1))
    w0 = float(cfg.get("w0", 1.0))
    drive_index = int(cfg.get("drive_index", 0))
    t_total = float(cfg.get("t_total", 800.0))
    dt = float(cfg.get("dt", 0.1))
    discard_ratio = float(cfg.get("discard_ratio", 0.5))
    omega_fine = cfg.get("omega_fine", [0.5, 1.5, 0.02])
    final_seeds = [int(s) for s in cfg.get("final_seeds", [0, 42, 123, 456])]

    omegas = np.arange(float(omega_fine[0]), float(omega_fine[1]) + 1e-12, float(omega_fine[2]))
    fixed_seed = final_seeds[0] if final_seeds else 0
    amp_by_omega = []
    for om in omegas:
        p = ParamsKij(
            N=n_nodes,
            gamma=gamma,
            F=force,
            Omega=float(om),
            w0=w0,
            K=K,
            discard_ratio=discard_ratio,
            t_total=t_total,
            dt=dt,
            seed=fixed_seed,
            drive_index=drive_index,
        )
        amp_by_omega.append(run_simulation(p)["amp_fft"])
    amp_by_omega = np.array(amp_by_omega, dtype=float)

    if seed_results and all(s in seed_results and "amp_fft" in seed_results[s] for s in final_seeds):
        amp_by_seed = np.array([seed_results[s]["amp_fft"] for s in final_seeds], dtype=float)
    else:
        amp_by_seed = []
        for s in final_seeds:
            p = ParamsKij(
                N=n_nodes,
                gamma=gamma,
                F=force,
                Omega=omega_selected,
                w0=w0,
                K=K,
                discard_ratio=discard_ratio,
                t_total=t_total,
                dt=dt,
                seed=s,
                drive_index=drive_index,
            )
            amp_by_seed.append(run_simulation(p)["amp_fft"])
        amp_by_seed = np.array(amp_by_seed, dtype=float)

    non_drive_idx = [i for i in range(n_nodes) if i != drive_index]
    sel = amp_by_omega[:, non_drive_idx]
    sel_ratio = sel.max(axis=1) / (np.partition(sel, -2, axis=1)[:, -2] + 1e-12)
    best_idx = int(np.argmax(sel_ratio))
    best_ratio = float(sel_ratio[best_idx])
    winner_idx = int(non_drive_idx[int(np.argmax(sel[best_idx]))])
    omega_best = float(omegas[best_idx])

    theta_seed = theta_seed_override if theta_seed_override is not None else (final_seeds[0] if final_seeds else 0)
    theta_params = ParamsKij(
        N=n_nodes,
        gamma=gamma,
        F=force,
        Omega=omega_best,
        w0=w0,
        K=K,
        discard_ratio=discard_ratio,
        t_total=t_total,
        dt=dt,
        seed=int(theta_seed),
        drive_index=drive_index,
    )
    t_eval, theta = simulate_theta(theta_params)

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(18, 14), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.85, 1.0])
    ax_full = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_sel = fig.add_subplot(gs[1, 0])
    ax_zoom = fig.add_subplot(gs[1, 1])
    ax_theta = fig.add_subplot(gs[2, :])

    def node_label(i: int) -> str:
        return f"Node {i} (drive)" if i == drive_index else f"Node {i}"

    for i in range(amp_by_omega.shape[1]):
        ax_full.plot(omegas, amp_by_omega[:, i], marker="o", markersize=2.5, linewidth=2, label=node_label(i))
    ax_full.axvline(omega_best, color="red", linestyle="--", linewidth=1.8, alpha=0.8, label=f"Best Omega={omega_best}")
    y_at_best = float(amp_by_omega[best_idx, winner_idx])
    ax_full.annotate(
        f"Best: {best_ratio:.2f}x",
        xy=(omega_best, y_at_best),
        xytext=(omega_best - 0.15, y_at_best * 0.82),
        color="red",
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="red"),
    )
    ax_full.set_title(
        f"N={n_nodes} K#{candidate['kidx']} Amplitude-Frequency (Peak selectivity: {best_ratio:.2f}x)",
        fontsize=14,
        fontweight="bold",
    )
    ax_full.set_xlabel("Driving Frequency (Omega)")
    ax_full.set_ylabel("FFT Amplitude")
    ax_full.set_xlim(float(omegas.min()), float(omegas.max()))
    ax_full.set_ylim(0.0, float(max(0.05, amp_by_omega.max() * 1.15)))
    ax_full.legend(loc="upper right", framealpha=0.9)

    zx0 = omega_best - max(float(zoom_half_width), 1e-9)
    zx1 = omega_best + max(float(zoom_half_width), 1e-9)
    zoom_mask = (omegas >= zx0 - 1e-12) & (omegas <= zx1 + 1e-12)
    omegas_zoom = omegas[zoom_mask]
    sel_ratio_zoom = sel_ratio[zoom_mask]
    ax_zoom.plot(omegas_zoom, sel_ratio_zoom, color="#b22222", marker="o", markersize=3.0, linewidth=2.0)
    ax_zoom.axvline(omega_best, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(0.0, max(2.0, float(np.max(sel_ratio_zoom)) * 1.10))
    ax_zoom.set_title("Zoom Near Best Omega", fontsize=14, fontweight="bold")
    ax_zoom.set_xlabel("Driving Frequency (Omega)")
    ax_zoom.set_ylabel("Selectivity")

    x = np.arange(amp_by_seed.shape[1])
    width = 0.8 / max(len(final_seeds), 1)
    for j, s in enumerate(final_seeds):
        offset = (j - (len(final_seeds) - 1) / 2.0) * width
        ax_bar.bar(x + offset, amp_by_seed[j], width=width, label=f"Seed {s}", edgecolor="black", alpha=0.88)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([node_label(int(i)) for i in x])
    ax_bar.set_xlabel("Node")
    ax_bar.set_ylabel("Amplitude")
    ax_bar.set_title(f"Seed Robustness at Omega={omega_best:.2f} ({len(final_seeds)} Seeds)", fontsize=14, fontweight="bold")
    ax_bar.set_ylim(0.0, float(max(0.05, amp_by_seed.max() * 1.18)))
    ax_bar.legend(loc="upper left", framealpha=0.9, ncol=min(4, len(final_seeds)))

    ax_sel.plot(omegas, sel_ratio, color="#b22222", linewidth=2.4, marker="o", markersize=3.2)
    ax_sel.axvline(omega_best, color="black", linestyle="--", linewidth=1.5, alpha=0.75)
    ax_sel.axvspan(zx0, zx1, color="#f6bd60", alpha=0.16)
    ax_sel.scatter([omega_best], [best_ratio], color="#b22222", s=45, zorder=4)
    ax_sel.set_title("Selectivity vs Omega", fontsize=14, fontweight="bold")
    ax_sel.set_xlabel("Driving Frequency (Omega)")
    ax_sel.set_ylabel("Selectivity")
    ax_sel.set_xlim(float(omegas.min()), float(omegas.max()))
    ax_sel.set_ylim(0.0, max(2.0, best_ratio * 1.18))
    ax_sel.grid(True, alpha=0.55)

    for i in range(theta.shape[1]):
        label = f"Node {i} (drive)" if i == drive_index else f"Node {i}"
        ax_theta.plot(t_eval, theta[:, i], linewidth=1.1, label=label)
    ax_theta.set_title(f"Theta Time Series at Best Omega={omega_best:.2f} (seed={theta_seed})", fontsize=14, fontweight="bold")
    ax_theta.set_xlabel("Time t")
    ax_theta.set_ylabel("Angle theta")
    ax_theta.legend(loc="upper right", ncol=min(3, n_nodes), framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_worker(payload: dict) -> int:
    plot_candidate(
        candidate=payload["candidate"],
        cfg=payload["cfg"],
        out_path=Path(payload["out_path"]),
        zoom_half_width=float(payload["zoom_half_width"]),
        theta_seed_override=payload["theta_seed_override"],
    )
    return int(payload["candidate"]["kidx"])


def main() -> None:
    args = parse_args()
    data = json.loads(args.json_path.read_text(encoding="utf-8"))["strict_search"]
    cfg = data.get("config", {})
    final = data.get("final_verification", [])
    stable = [c for c in final if bool(c.get("stable"))]
    if not stable:
        raise ValueError("No stable final candidates found in the JSON file.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stable = sorted(stable, key=lambda x: int(x["kidx"]))
    workers = max(1, min(int(args.workers), os.cpu_count() or 1))
    print(f"saving {len(stable)} stable verification figures to {args.out_dir} with {workers} workers")

    start_time = time.perf_counter()
    payloads = [
        {
            "candidate": candidate,
            "cfg": cfg,
            "out_path": str(args.out_dir / f"K{int(candidate['kidx'])}_verification_en.png"),
            "zoom_half_width": float(args.zoom_half_width),
            "theta_seed_override": args.theta_seed,
        }
        for candidate in stable
    ]
    if workers == 1:
        for idx, payload in enumerate(payloads, start=1):
            kidx = _plot_worker(payload)
            _print_progress(idx, len(payloads), start_time, extra=f"K{kidx}")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_plot_worker, payload) for payload in payloads]
            for fut in as_completed(futures):
                kidx = fut.result()
                done += 1
                _print_progress(done, len(payloads), start_time, extra=f"K{kidx}")

    print(f"done: {args.out_dir}")


if __name__ == "__main__":
    main()
