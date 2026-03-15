from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParamsKij:
    N: int = 5
    gamma: float = 0.08
    F: float = 0.1
    Omega: float = 1.0
    w0: float = 1.0
    K: np.ndarray | None = None
    discard_ratio: float = 0.5
    t_total: float = 800.0
    dt: float = 0.1
    seed: int = 0
    drive_index: int = 0


def initial_conditions(params: ParamsKij, noise_scale: float = 1e-3) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    theta0 = noise_scale * rng.standard_normal(params.N)
    omega0 = noise_scale * rng.standard_normal(params.N)
    return np.concatenate([theta0, omega0])


def _k_matrix(params: ParamsKij) -> np.ndarray:
    if params.K is None:
        raise ValueError("K must be provided.")
    K = np.array(params.K, dtype=float, copy=True)
    if K.shape != (params.N, params.N):
        raise ValueError(f"K must have shape ({params.N}, {params.N}).")
    np.fill_diagonal(K, 0.0)
    return K


def pendulum_ode_kij(t: float, y: np.ndarray, params: ParamsKij) -> np.ndarray:
    N = params.N
    theta = y[:N]
    omega = y[N:]
    K = _k_matrix(params)

    coupling = (K * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)

    drive = np.zeros_like(theta)
    drive[params.drive_index] = params.F * np.cos(params.Omega * t)

    dtheta_dt = omega
    domega_dt = -params.gamma * omega - (params.w0**2) * np.sin(theta) + coupling + drive
    return np.concatenate([dtheta_dt, domega_dt])
