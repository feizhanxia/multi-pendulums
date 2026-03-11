from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParamsKij:
    """Parameters for N=3 pendulum network with general K_ij coupling."""

    N: int = 3
    gamma: float = 0.08
    F: float = 0.1
    Omega: float = 1.0
    w0: float = 1.0
    K: np.ndarray | None = None  # shape (N, N), diagonal should be 0
    discard_ratio: float = 0.5
    t_total: float = 2000.0
    dt: float = 0.5
    seed: int = 0
    drive_index: int = 0


def initial_conditions(params: ParamsKij, noise_scale: float = 1e-3) -> np.ndarray:
    """Small random angles/velocities to break symmetry."""
    rng = np.random.default_rng(params.seed)
    theta0 = noise_scale * rng.standard_normal(params.N)
    omega0 = noise_scale * rng.standard_normal(params.N)
    return np.concatenate([theta0, omega0])


def _k_matrix(params: ParamsKij) -> np.ndarray:
    if params.K is None:
        raise ValueError("K matrix must be provided for ParamsKij.")
    K = np.array(params.K, dtype=float, copy=True)
    if K.shape != (params.N, params.N):
        raise ValueError(f"K matrix must be shape ({params.N}, {params.N}).")
    np.fill_diagonal(K, 0.0)
    return K


def pendulum_ode_kij(t: float, y: np.ndarray, params: ParamsKij) -> np.ndarray:
    """ODE for N=3 pendulums with general (possibly asymmetric) coupling K_ij."""
    N = params.N
    theta = y[:N]
    omega = y[N:]

    K = _k_matrix(params)
    # Coupling term: sum_j K_ij * sin(theta_j - theta_i)
    coupling = (K * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)

    drive = np.zeros_like(theta)
    drive[params.drive_index] = params.F * np.cos(params.Omega * t)

    dtheta_dt = omega
    domega_dt = -params.gamma * omega - (params.w0**2) * np.sin(theta) + coupling + drive
    return np.concatenate([dtheta_dt, domega_dt])


def rms_amplitude(theta: np.ndarray) -> np.ndarray:
    """Root-mean-square amplitude per node over time axis 0."""
    return np.sqrt(np.mean(theta**2, axis=0))
