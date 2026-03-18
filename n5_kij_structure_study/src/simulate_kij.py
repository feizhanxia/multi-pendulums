from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .model_kij import ParamsKij, initial_conditions, pendulum_ode_kij


def run_simulation(params: ParamsKij) -> dict[str, Any]:
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
    discard_idx = int(len(t_eval) * params.discard_ratio)
    theta_tail = theta[discard_idx:]
    t_tail = t_eval[discard_idx:]

    phase = np.exp(-1j * params.Omega * t_tail)
    amp_fft = np.abs(np.mean(theta_tail * phase[:, None], axis=0))

    non_drive_idx = [i for i in range(params.N) if i != params.drive_index]
    amp_fft_nd = amp_fft[non_drive_idx]
    nd_order = np.argsort(amp_fft_nd)
    nd_max = float(amp_fft_nd[nd_order[-1]])
    nd_second = float(amp_fft_nd[nd_order[-2]])
    selectivity_nd = float(nd_max / (nd_second + 1e-12))

    return {
        "amp_fft": amp_fft.tolist(),
        "selectivity_fft_nd": selectivity_nd,
    }
