#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from src.model_kij import ParamsKij, initial_conditions, pendulum_ode_kij
from src.simulate_kij import run_simulation


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Plot strict-search summary with full/zoom/robustness/theta panels.")
    p.add_argument("--json_path", type=Path, default=root / "results" / "data" / "n5_strict_results.json")
    p.add_argument("--out_path", type=Path, default=root / "results" / "figures" / "n5_top1_verification_en.png")
    p.add_argument("--zoom_half_width", type=float, default=0.2, help="Half-width around best omega for zoom panel.")
    p.add_argument("--theta_seed", type=int, default=None, help="Seed for theta(t) panel; default uses first final seed.")
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


def main() -> None:
    args = parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.json_path.read_text(encoding="utf-8"))["strict_search"]
    cfg = data.get("config", {})
    use_target_band = bool(cfg.get("use_target_band", False))
    target_omega = float(cfg.get("target_omega", 1.5))
    final = data.get("final_verification", [])
    refined = data.get("refined_results", [])
    robust = data.get("robustness_test", [])
    coarse = data.get("coarse_results", [])

    # Fallback priority: final -> refined -> robustness -> coarse.
    if final:
        candidate_pool = [
            {
                "kidx": c["kidx"],
                "K": c["K"],
                "omega": c["omega"],
                "avg_selectivity": c["avg_selectivity"],
                "seed_results": c.get("seed_results", []),
                "source": "final_verification",
            }
            for c in final
        ]
    elif refined:
        candidate_pool = [
            {
                "kidx": c["kidx"],
                "K": c["K"],
                "omega": c["refined_omega"],
                "avg_selectivity": c["refined_selectivity"],
                "seed_results": [],
                "source": "refined_results",
            }
            for c in refined
        ]
    elif robust:
        candidate_pool = [
            {
                "kidx": c["kidx"],
                "K": c["K"],
                "omega": c["original_omega"],
                "avg_selectivity": c["min_selectivity"],
                "seed_results": [],
                "source": "robustness_test",
            }
            for c in robust
        ]
    elif coarse:
        candidate_pool = [
            {
                "kidx": c["kidx"],
                "K": c["K"],
                "omega": c["best_omega"],
                "avg_selectivity": c["best_selectivity"],
                "seed_results": [],
                "source": "coarse_results",
            }
            for c in coarse
        ]
    else:
        raise ValueError("No candidates found in strict_search JSON.")

    # Always visualize the strongest verified candidate. Using target-band
    # proximity here made the plot highlight a candidate that was not the
    # actual stable winner of the run.
    best = max(candidate_pool, key=lambda x: x["avg_selectivity"])
    K = np.array(best["K"], dtype=float)
    omega_selected = float(best["omega"])
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

    seeds = final_seeds
    seed_results = {int(r["seed"]): r for r in best.get("seed_results", [])}
    if seed_results and all(s in seed_results and "amp_fft" in seed_results[s] for s in seeds):
        amp_by_seed = np.array([seed_results[s]["amp_fft"] for s in seeds], dtype=float)
    else:
        amp_by_seed = []
        for s in seeds:
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

    theta_seed = args.theta_seed if args.theta_seed is not None else (seeds[0] if seeds else 0)
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
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_sel = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
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
    source_tag = best.get("source", "unknown")
    ax_full.set_title(
        f"N={n_nodes} K#{best['kidx']} Amplitude-Frequency (Peak selectivity: {best_ratio:.2f}x, source={source_tag})",
        fontsize=14,
        fontweight="bold",
    )
    ax_full.set_xlabel("Driving Frequency (Omega)")
    ax_full.set_ylabel("FFT Amplitude")
    ax_full.set_xlim(float(omegas.min()), float(omegas.max()))
    ax_full.set_ylim(0.0, float(max(0.05, amp_by_omega.max() * 1.15)))
    ax_full.legend(loc="upper right", framealpha=0.9)

    # Recompute zoom panel data independently, instead of slicing full-range points.
    zoom_half_width = max(float(args.zoom_half_width), 1e-9)
    omega_step = float(omega_fine[2]) if len(omega_fine) >= 3 else 0.02
    zx0 = omega_best - zoom_half_width
    zx1 = omega_best + zoom_half_width
    omegas_zoom = np.arange(zx0, zx1 + 1e-12, omega_step)
    amp_by_omega_zoom = []
    for om in omegas_zoom:
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
        amp_by_omega_zoom.append(run_simulation(p)["amp_fft"])
    amp_by_omega_zoom = np.array(amp_by_omega_zoom, dtype=float)
    sel_zoom = amp_by_omega_zoom[:, non_drive_idx]
    sel_ratio_zoom = sel_zoom.max(axis=1) / (np.partition(sel_zoom, -2, axis=1)[:, -2] + 1e-12)
    ax_zoom.plot(omegas_zoom, sel_ratio_zoom, color="#b22222", marker="o", markersize=3.0, linewidth=2.0)
    ax_zoom.axvline(omega_best, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_zoom.set_xlim(zx0, zx1)
    y_zoom_max = float(np.max(sel_ratio_zoom))
    ax_zoom.set_ylim(0.0, max(2.0, y_zoom_max * 1.10))
    ax_zoom.set_title("Zoom Near Best Omega", fontsize=14, fontweight="bold")
    ax_zoom.set_xlabel("Driving Frequency (Omega)")
    ax_zoom.set_ylabel("Selectivity")

    x = np.arange(amp_by_seed.shape[1])
    width = 0.8 / max(len(seeds), 1)
    for j, s in enumerate(seeds):
        offset = (j - (len(seeds) - 1) / 2.0) * width
        ax_bar.bar(x + offset, amp_by_seed[j], width=width, label=f"Seed {s}", edgecolor="black", alpha=0.88)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([node_label(int(i)) for i in x])
    ax_bar.set_xlabel("Node")
    ax_bar.set_ylabel("Amplitude")
    ax_bar.set_title(f"Seed Robustness at Omega={omega_best:.2f} ({len(seeds)} Seeds)", fontsize=14, fontweight="bold")
    ax_bar.set_ylim(0.0, float(max(0.05, amp_by_seed.max() * 1.18)))
    ax_bar.legend(loc="upper left", framealpha=0.9, ncol=min(4, len(seeds)))

    ax_sel.plot(omegas, sel_ratio, color="#b22222", linewidth=2.4, marker="o", markersize=3.2)
    ax_sel.axvline(omega_best, color="black", linestyle="--", linewidth=1.5, alpha=0.75)
    if use_target_band:
        ax_sel.axvspan(target_omega - float(cfg.get("target_half_width", 0.2)), target_omega + float(cfg.get("target_half_width", 0.2)), color="#f6bd60", alpha=0.16)
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

    fig.savefig(args.out_path, dpi=150)
    plt.close(fig)
    print(f"json: {args.json_path}")
    print(f"saved: {args.out_path}")


if __name__ == "__main__":
    main()
