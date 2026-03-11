from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Params:
    """Container for pendulum network parameters."""

    N: int = 4
    gamma: float = 0.08
    K: float = 0.4
    F: float = 0.1
    Omega: float = 0.95
    w0: float = 1.0
    detune_eps: float = 0.0  # detune magnitude
    detune_target: int = -1  # -1: gradient; >=0: only that node detuned
    discard_ratio: float = 0.5
    t_total: float = 2000.0
    dt: float = 0.5
    seed: int = 0
    drive_index: int = 0


def initial_conditions(params: Params, noise_scale: float = 1e-3) -> np.ndarray:
    """Small random angles/velocities to break symmetry."""
    rng = np.random.default_rng(params.seed)
    theta0 = noise_scale * rng.standard_normal(params.N)
    omega0 = noise_scale * rng.standard_normal(params.N)
    return np.concatenate([theta0, omega0])


def pendulum_ode(t: float, y: np.ndarray, params: Params) -> np.ndarray:
    """ODE for all-to-all coupled damped driven pendulums."""
    N = params.N
    theta = y[:N]
    omega = y[N:]

    # Small detune to break degeneracy.
    detune = params.detune_eps
    if detune != 0.0:
        w0_vec = np.full(N, params.w0)
        if params.detune_target >= 0 and params.detune_target < N:
            w0_vec[params.detune_target] = params.w0 * (1.0 + detune)
        else:
            # Gradient spread around drive_index as center.
            scale = np.linspace(-0.5, 0.5, N)
            scale = np.roll(scale, -params.drive_index)
            w0_vec = params.w0 * (1.0 + detune * scale)
    else:
        w0_vec = np.full(N, params.w0)

    # Coupling term: average sine of phase differences
    coupling = np.sin(theta[:, None] - theta[None, :]).sum(axis=1) * (params.K / N)

    # Driving term only on drive_index
    drive = np.zeros_like(theta)
    drive[params.drive_index] = params.F * np.cos(params.Omega * t)

    dtheta_dt = omega
    domega_dt = -params.gamma * omega - (w0_vec ** 2) * np.sin(theta) + coupling + drive
    return np.concatenate([dtheta_dt, domega_dt])


def rms_amplitude(theta: np.ndarray) -> np.ndarray:
    """Root-mean-square amplitude per node over time axis 0."""
    return np.sqrt(np.mean(theta ** 2, axis=0))
