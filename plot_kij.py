from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_kij(values: list[float], n: int) -> np.ndarray:
    if len(values) != n * n:
        raise ValueError(f"--K requires {n*n} values (row-major).")
    K = np.array(values, dtype=float).reshape(n, n)
    np.fill_diagonal(K, 0.0)
    return K


def plot_kij(K: np.ndarray, out_path: Path, title: str | None = None) -> None:
    v = np.max(np.abs(K)) if K.size else 1.0
    if v == 0:
        v = 1.0
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=200)
    im = ax.imshow(K, cmap="coolwarm", vmin=-v, vmax=v)
    ax.set_xlabel("j (source)")
    ax.set_ylabel("i (target)")
    ax.set_xticks(range(K.shape[1]))
    ax.set_yticks(range(K.shape[0]))
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="K_ij")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot K_ij heatmap.")
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--K", type=float, nargs="+", required=True, help="K_ij row-major (N*N values).")
    p.add_argument("--out_path", type=Path, default=Path("figures/kij_heatmap.png"))
    p.add_argument("--title", type=str, default="K_ij heatmap")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    K = _parse_kij(args.K, args.N)
    plot_kij(K, Path(args.out_path), title=args.title)
    print(f"saved heatmap to {args.out_path}")


if __name__ == "__main__":
    main()
