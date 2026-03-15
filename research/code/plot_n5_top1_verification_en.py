#!/usr/bin/env python3
"""Recreate research/figures/n5_top1_verification_en.png from strict-search results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model_kij import ParamsKij
from simulate_kij import run_simulation


ROOT = Path(__file__).resolve().parents[2]
STRICT_JSON = ROOT / "research" / "data" / "n5_strict_results.json"
OUT_PATH = ROOT / "research" / "figures" / "n5_top1_verification_en.png"


def load_k28() -> tuple[np.ndarray, float]:
    strict = json.loads(STRICT_JSON.read_text(encoding="utf-8"))["strict_search"]
    final = strict["final_verification"][0]
    return np.array(final["K"], dtype=float), float(final["omega"])


def scan_amplitude_frequency(K: np.ndarray, omega_min: float = 1.3, omega_max: float = 1.7, step: float = 0.01):
    omegas = np.arange(omega_min, omega_max + 1e-12, step)
    amps = []
    for omega in omegas:
        params = ParamsKij(
            N=K.shape[0],
            gamma=0.08,
            F=0.1,
            Omega=float(omega),
            w0=1.0,
            K=K,
            discard_ratio=0.5,
            t_total=800.0,
            dt=0.1,
            seed=0,
            drive_index=0,
        )
        rec = run_simulation(params)
        amps.append(rec["amp_fft"])
    return omegas, np.array(amps, dtype=float)


def seed_robustness_amplitudes(K: np.ndarray, omega_best: float, seeds: list[int]):
    vals = []
    for s in seeds:
        params = ParamsKij(
            N=K.shape[0],
            gamma=0.08,
            F=0.1,
            Omega=omega_best,
            w0=1.0,
            K=K,
            discard_ratio=0.5,
            t_total=800.0,
            dt=0.1,
            seed=s,
            drive_index=0,
        )
        rec = run_simulation(params)
        vals.append(rec["amp_fft"])
    return np.array(vals, dtype=float)


def main() -> None:
    K, omega_best = load_k28()
    omegas, amp_by_omega = scan_amplitude_frequency(K)
    seed_ids = [0, 1, 2, 3]
    amp_by_seed = seed_robustness_amplitudes(K, omega_best, seed_ids)

    sel = amp_by_omega[:, 1:]
    sel_ratio = sel.max(axis=1) / (np.partition(sel, -2, axis=1)[:, -2] + 1e-12)
    best_idx = int(np.argmax(sel_ratio))
    best_ratio = float(sel_ratio[best_idx])

    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    for i in range(amp_by_omega.shape[1]):
        ax1.plot(omegas, amp_by_omega[:, i], marker="o", markersize=2.5, linewidth=2, label=f"Node {i}")
    ax1.axvline(omega_best, color="red", linestyle="--", linewidth=1.8, alpha=0.8, label=f"Best Omega={omega_best}")
    ax1.annotate(
        f"Best: {best_ratio:.2f}x",
        xy=(omega_best, amp_by_omega[best_idx, 4]),
        xytext=(omega_best - 0.12, amp_by_omega[best_idx, 4] - 0.03),
        color="red",
        fontsize=13,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="red"),
    )
    ax1.set_title(f"N=5 K#28 Amplitude-Frequency (Best: {best_ratio:.2f}x)", fontsize=15, fontweight="bold")
    ax1.set_xlabel("Driving Frequency (Omega)")
    ax1.set_ylabel("FFT Amplitude")
    ax1.set_xlim(1.3, 1.7)
    ax1.set_ylim(0.0, 0.2)
    ax1.legend(loc="upper right", framealpha=0.9)

    x = np.arange(amp_by_seed.shape[1])
    width = 0.18
    for j, s in enumerate(seed_ids):
        ax2.bar(x + (j - 1.5) * width, amp_by_seed[j], width=width, label=f"Seed {s}", edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Node {i}" for i in x])
    ax2.set_xlabel("Node")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Robustness Test at Omega={omega_best} ({len(seed_ids)} Seeds)", fontsize=15, fontweight="bold")
    ax2.set_ylim(0.0, 0.15)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
