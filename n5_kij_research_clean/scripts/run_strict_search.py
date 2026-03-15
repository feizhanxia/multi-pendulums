#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np

from src.model_kij import ParamsKij
from src.simulate_kij import run_simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict search for N=5 K_ij.")
    p.add_argument("--target_omega", type=float, default=1.5, help="Center omega for band-focused selection.")
    p.add_argument("--target_half_width", type=float, default=0.2, help="Half width of target omega band.")
    p.add_argument("--use_target_band", action="store_true", help="If set, select best only inside target band.")
    p.add_argument("--n_samples", type=int, default=120, help="Number of random K_ij candidates in coarse stage.")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process count. Default: min(16, cpu_count-2), with lower bound 1.",
    )
    return p.parse_args()


def frange(start: float, stop: float, step: float) -> list[float]:
    vals = []
    v = start
    while v <= stop + 1e-12:
        vals.append(round(v, 10))
        v += step
    return vals


def sample_kij(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    K = rng.uniform(low, high, size=(n, n))
    np.fill_diagonal(K, 0.0)
    return K


def evaluate_best_for_k(
    K: np.ndarray,
    omegas: list[float],
    base: ParamsKij,
    seed: int,
    use_target_band: bool,
    target_omega: float,
    target_half_width: float,
) -> dict:
    best_sel = -1.0
    best_omega = omegas[0]
    best_amps = None
    if use_target_band:
        band_omegas = [om for om in omegas if abs(om - target_omega) <= target_half_width + 1e-12]
        eval_omegas = band_omegas if band_omegas else omegas
    else:
        eval_omegas = omegas
    for i, om in enumerate(eval_omegas):
        params = ParamsKij(**{**base.__dict__, "Omega": om, "K": K, "seed": seed + i})
        rec = run_simulation(params)
        sel = float(rec["selectivity_fft_nd"])
        if sel > best_sel:
            best_sel = sel
            best_omega = om
            best_amps = rec["amp_fft"]
    return {
        "best_omega": float(best_omega),
        "best_selectivity": float(best_sel),
        "best_amps": best_amps,
    }


def seed_check(K: np.ndarray, omega: float, base: ParamsKij, seeds: list[int]) -> dict:
    out = {}
    sels = []
    for s in seeds:
        params = ParamsKij(**{**base.__dict__, "Omega": omega, "K": K, "seed": s})
        rec = run_simulation(params)
        sels.append(float(rec["selectivity_fft_nd"]))
        out[f"seed_{s}"] = {
            "selectivity": float(rec["selectivity_fft_nd"]),
            "amps": rec["amp_fft"],
        }
    out["min_selectivity"] = float(min(sels))
    out["avg_selectivity"] = float(np.mean(sels))
    return out


def _coarse_worker(payload: dict) -> dict:
    K = np.array(payload["K"], dtype=float)
    omegas = payload["omegas"]
    base = ParamsKij(**payload["base"])
    best = evaluate_best_for_k(
        K=K,
        omegas=omegas,
        base=base,
        seed=int(payload["seed"]),
        use_target_band=bool(payload["use_target_band"]),
        target_omega=float(payload["target_omega"]),
        target_half_width=float(payload["target_half_width"]),
    )
    return {
        "kidx": int(payload["kidx"]),
        "K": K.tolist(),
        "best_omega": best["best_omega"],
        "best_selectivity": best["best_selectivity"],
        "best_amps": best["best_amps"],
    }


def _auto_workers(user_workers: int | None) -> int:
    if user_workers is not None:
        return max(1, int(user_workers))
    cpu = os.cpu_count() or 1
    return max(1, min(16, cpu - 2))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_json = root / "data" / "n5_strict_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "N": 5,
        "gamma": 0.08,
        "F": 0.1,
        "w0": 1.0,
        "drive_index": 0,
        "t_total": 800.0,
        "dt": 0.1,
        "discard_ratio": 0.5,
        "k_range": [-1.0, 1.0],
        "n_samples": int(args.n_samples),
        "omega_coarse": [0.5, 1.5, 0.1],
        "omega_fine": [0.5, 1.5, 0.02],
        "top_k": 5,
        "robust_seeds": [0, 42],
        "final_seeds": [0, 42, 123, 456],
        "pass_threshold": 2.0,
        "rng_seed": 0,
        "use_target_band": bool(args.use_target_band),
        "target_omega": float(args.target_omega),
        "target_half_width": float(args.target_half_width),
        "workers": _auto_workers(args.workers),
    }

    base = ParamsKij(
        N=cfg["N"],
        gamma=cfg["gamma"],
        F=cfg["F"],
        Omega=1.0,
        w0=cfg["w0"],
        K=None,
        discard_ratio=cfg["discard_ratio"],
        t_total=cfg["t_total"],
        dt=cfg["dt"],
        seed=0,
        drive_index=cfg["drive_index"],
    )
    rng = np.random.default_rng(cfg["rng_seed"])

    omega_coarse = frange(*cfg["omega_coarse"])
    omega_fine = frange(*cfg["omega_fine"])

    # Pre-sample K candidates once for deterministic behavior.
    k_candidates = []
    for kidx in range(cfg["n_samples"]):
        K = sample_kij(cfg["N"], cfg["k_range"][0], cfg["k_range"][1], rng)
        k_candidates.append({"kidx": int(kidx), "K": K.tolist()})

    print(
        f"[coarse] evaluating {cfg['n_samples']} random K_ij with {cfg['workers']} workers "
        f"(target_band={'on' if cfg['use_target_band'] else 'off'})"
    )
    coarse_payloads = [
        {
            "kidx": c["kidx"],
            "K": c["K"],
            "omegas": omega_coarse,
            "base": base.__dict__,
            "seed": 10000 * (c["kidx"] + 1),
            "use_target_band": cfg["use_target_band"],
            "target_omega": cfg["target_omega"],
            "target_half_width": cfg["target_half_width"],
        }
        for c in k_candidates
    ]

    coarse_results = []
    if cfg["workers"] == 1:
        for i, payload in enumerate(coarse_payloads, start=1):
            coarse_results.append(_coarse_worker(payload))
            print(f"[coarse] progress {i}/{len(coarse_payloads)}")
    else:
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
            futures = [ex.submit(_coarse_worker, payload) for payload in coarse_payloads]
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                coarse_results.append(fut.result())
                done += 1
                print(f"[coarse] progress {done}/{total}")

    all_candidates = sorted(coarse_results, key=lambda x: x["kidx"])

    top = sorted(all_candidates, key=lambda x: x["best_selectivity"], reverse=True)[: cfg["top_k"]]
    print(f"[robustness] evaluating top-{len(top)} coarse candidates")

    robustness_test = []
    passed = []
    for rank, c in enumerate(top, start=1):
        K = np.array(c["K"], dtype=float)
        checks = seed_check(K, float(c["best_omega"]), base, cfg["robust_seeds"])
        item = {
            "rank": rank,
            "kidx": c["kidx"],
            "K": c["K"],
            "original_omega": c["best_omega"],
            "original_selectivity": c["best_selectivity"],
            **checks,
            "passed": bool(checks["min_selectivity"] > cfg["pass_threshold"]),
        }
        if item["passed"]:
            passed.append(item)
        robustness_test.append(item)
        print(f"[robustness] progress {rank}/{len(top)} (passed={len(passed)})")

    print(f"[refined] evaluating {len(passed)} passed candidates")
    refined_results = []
    refined_passed = []
    for i, c in enumerate(passed, start=1):
        K = np.array(c["K"], dtype=float)
        best = evaluate_best_for_k(
            K,
            omega_fine,
            base,
            seed=200000 + c["kidx"] * 100,
            use_target_band=cfg["use_target_band"],
            target_omega=cfg["target_omega"],
            target_half_width=cfg["target_half_width"],
        )
        fcheck = seed_check(K, best["best_omega"], base, cfg["robust_seeds"])
        item = {
            "kidx": c["kidx"],
            "K": c["K"],
            "refined_omega": best["best_omega"],
            "refined_selectivity": best["best_selectivity"],
            "final_seed_0": fcheck["seed_0"]["selectivity"],
            "final_seed_42": fcheck["seed_42"]["selectivity"],
            "final_passed": bool(fcheck["min_selectivity"] > cfg["pass_threshold"]),
            "best_amps": best["best_amps"],
        }
        refined_results.append(item)
        if item["final_passed"]:
            refined_passed.append(item)
        print(f"[refined] progress {i}/{len(passed)} (final_passed={len(refined_passed)})")

    print(f"[final] verifying {len(refined_passed)} final candidates across {len(cfg['final_seeds'])} seeds")
    final_verification = []
    for i, c in enumerate(refined_passed, start=1):
        K = np.array(c["K"], dtype=float)
        v = seed_check(K, float(c["refined_omega"]), base, cfg["final_seeds"])
        final_verification.append(
            {
                "kidx": c["kidx"],
                "K": c["K"],
                "omega": c["refined_omega"],
                "seed_results": [
                    {
                        "seed": s,
                        "selectivity": v[f"seed_{s}"]["selectivity"],
                        "amp_fft": v[f"seed_{s}"]["amps"],
                    }
                    for s in cfg["final_seeds"]
                ],
                "min_selectivity": v["min_selectivity"],
                "avg_selectivity": v["avg_selectivity"],
                "stable": bool(v["min_selectivity"] > cfg["pass_threshold"]),
            }
        )
        print(f"[final] progress {i}/{len(refined_passed)}")

    result = {
        "strict_search": {
            "config": cfg,
            "n_samples": cfg["n_samples"],
            "omega_coarse_step": cfg["omega_coarse"][2],
            "omega_fine_step": cfg["omega_fine"][2],
            "coarse_results": coarse_results,
            "robustness_test": robustness_test,
            "passed_candidates": len(passed),
            "failed_candidates": len(top) - len(passed),
            "refined_results": refined_results,
            "final_verification": final_verification,
        }
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_json}")
    if final_verification:
        best = max(final_verification, key=lambda x: x["avg_selectivity"])
        print(f"stable winner: K#{best['kidx']} @ omega={best['omega']:.2f}, avg={best['avg_selectivity']:.2f}x")
    else:
        print("no stable candidate found in this run.")


if __name__ == "__main__":
    main()
