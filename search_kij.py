from __future__ import annotations

import argparse
import json
from datetime import datetime
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


def vector_dim(n: int) -> int:
    return n * (n - 1)


def vector_to_kij(vec: np.ndarray, n: int) -> np.ndarray:
    if vec.shape[0] != vector_dim(n):
        raise ValueError("K vector length mismatch.")
    K = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            K[i, j] = vec[idx]
            idx += 1
    return K


def sample_kij_vector(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low, high, size=(vector_dim(n),))


def _window_scores(
    winners: np.ndarray,
    selectivity: np.ndarray,
    target_node: int,
    window_size: int,
    omegas: np.ndarray,
) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []
    for start in range(0, len(winners) - window_size + 1):
        end = start + window_size - 1
        win_ratio = float(np.mean(winners[start : end + 1] == target_node))
        avg_sel = float(np.mean(selectivity[start : end + 1]))
        score = win_ratio
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


def best_band_pair(
    omegas: np.ndarray,
    winners: np.ndarray,
    selectivity: np.ndarray,
    node_a: int,
    node_b: int,
    window_size: int,
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    scores_a = _window_scores(winners, selectivity, node_a, window_size, omegas)
    scores_b = _window_scores(winners, selectivity, node_b, window_size, omegas)

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

    top_a = max(scores_a, key=lambda x: x["score"])
    top_b = max(scores_b, key=lambda x: x["score"])
    return top_a, top_b, False


def evaluate_kij(
    K: np.ndarray,
    omega_vals: List[float],
    params_base: ParamsKij,
    node_a: int,
    node_b: int,
    window_size: int,
) -> Dict[str, object]:
    amps = []
    for idx, omega in enumerate(omega_vals):
        params = ParamsKij(
            N=params_base.N,
            gamma=params_base.gamma,
            F=params_base.F,
            Omega=omega,
            w0=params_base.w0,
            K=K,
            discard_ratio=params_base.discard_ratio,
            t_total=params_base.t_total,
            dt=params_base.dt,
            seed=params_base.seed + idx,
            drive_index=params_base.drive_index,
        )
        result = run_simulation(params)
        amps.append(result["amp_fft"])

    omegas = np.array(omega_vals, dtype=float)
    amps_arr = np.array(amps, dtype=float)
    amp_a = amps_arr[:, node_a]
    amp_b = amps_arr[:, node_b]
    winners = np.where(amp_a >= amp_b, node_a, node_b)
    selectivity = np.maximum(amp_a, amp_b) / (np.minimum(amp_a, amp_b) + 1e-12)

    band_a, band_b, non_overlap = best_band_pair(
        omegas, winners, selectivity, node_a, node_b, window_size
    )
    score_sum = float(
        np.sqrt(band_a["win_ratio"] * band_b["win_ratio"])
        * ((band_a["avg_selectivity"] + band_b["avg_selectivity"]) / 2.0)
    )

    return {
        "K": K.tolist(),
        "score": float(score_sum),
        "band_a": band_a,
        "band_b": band_b,
        "non_overlap": non_overlap,
    }


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float, variance: float) -> np.ndarray:
    dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return variance * np.exp(-0.5 * dists / (length_scale**2))


def gp_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    length_scale: float,
    variance: float,
    noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    K = rbf_kernel(X_train, X_train, length_scale, variance)
    K += (noise + 1e-9) * np.eye(len(X_train))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    K_s = rbf_kernel(X_train, X_test, length_scale, variance)
    mu = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    K_ss = rbf_kernel(X_test, X_test, length_scale, variance)
    var = np.clip(np.diag(K_ss) - np.sum(v * v, axis=0), 1e-12, None)
    return mu, var


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search K_ij to separate node 1/2 bands (RMS-based).")
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.08)
    p.add_argument("--F", type=float, default=0.1)
    p.add_argument("--w0", type=float, default=1.0)
    p.add_argument("--drive_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=["random", "bo"], default="bo")
    p.add_argument("--k_range", type=float, nargs=2, default=[-1.0, 1.0], metavar=("LOW", "HIGH"))
    p.add_argument("--n_samples", type=int, default=20, help="Used only for method=random.")
    p.add_argument("--n_init", type=int, default=8, help="Initial random points for BO.")
    p.add_argument("--n_iter", type=int, default=12, help="BO iterations after init.")
    p.add_argument("--n_candidates", type=int, default=200, help="Random candidates per BO step.")
    p.add_argument("--bo_length_scale", type=float, default=1.0)
    p.add_argument("--bo_variance", type=float, default=1.0)
    p.add_argument("--bo_noise", type=float, default=1e-6)
    p.add_argument("--bo_kappa", type=float, default=2.0)
    p.add_argument("--omega_range", type=float, nargs=3, default=[0.6, 1.4, 0.03], metavar=("START", "STOP", "STEP"))
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--node_a", type=int, default=1)
    p.add_argument("--node_b", type=int, default=2)
    p.add_argument("--t_total", type=float, default=800.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--discard_ratio", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--out_root", type=Path, default=Path("data/summary/search_runs"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    omega_vals = _frange(*args.omega_range)
    base_params = ParamsKij(
        N=args.N,
        gamma=args.gamma,
        F=args.F,
        Omega=omega_vals[0],
        w0=args.w0,
        K=None,
        discard_ratio=args.discard_ratio,
        t_total=args.t_total,
        dt=args.dt,
        seed=args.seed,
        drive_index=args.drive_index,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results: List[Dict[str, object]] = []
    if args.method == "random":
        for idx in range(args.n_samples):
            vec = sample_kij_vector(args.N, args.k_range[0], args.k_range[1], rng)
            K = vector_to_kij(vec, args.N)
            result = evaluate_kij(K, omega_vals, base_params, args.node_a, args.node_b, args.window_size)
            result["vector"] = vec.tolist()
            results.append(result)
            print(f"[{idx+1}/{args.n_samples}] score={result['score']:.4f}")
    else:
        X: List[np.ndarray] = []
        y: List[float] = []
        total = args.n_init + args.n_iter
        for idx in range(total):
            if idx < args.n_init:
                vec = sample_kij_vector(args.N, args.k_range[0], args.k_range[1], rng)
            else:
                X_arr = np.vstack(X)
                y_arr = np.array(y, dtype=float)
                y_mean = float(np.mean(y_arr))
                y_std = float(np.std(y_arr) + 1e-8)
                y_norm = (y_arr - y_mean) / y_std

                candidates = rng.uniform(
                    args.k_range[0],
                    args.k_range[1],
                    size=(args.n_candidates, vector_dim(args.N)),
                )
                mu, var = gp_predict(
                    X_arr,
                    y_norm,
                    candidates,
                    args.bo_length_scale,
                    args.bo_variance,
                    args.bo_noise,
                )
                ucb = mu + args.bo_kappa * np.sqrt(var)
                vec = candidates[int(np.argmax(ucb))]

            K = vector_to_kij(vec, args.N)
            result = evaluate_kij(K, omega_vals, base_params, args.node_a, args.node_b, args.window_size)
            result["vector"] = vec.tolist()
            results.append(result)
            X.append(vec)
            y.append(float(result["score"]))
            print(f"[{idx+1}/{total}] score={result['score']:.4f}")

    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    top_results = results_sorted[: args.top_k]

    all_path = run_dir / "search_kij_all.json"
    top_path = run_dir / "search_kij_top.json"
    all_path.write_text(json.dumps(results_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
    top_path.write_text(json.dumps(top_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run dir: {run_dir}")
    print(f"all results saved to {all_path}")
    print(f"top results saved to {top_path}")


if __name__ == "__main__":
    main()
