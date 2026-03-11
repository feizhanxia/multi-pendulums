# Driven All-to-All Pendulum Network 
(Minimal Neural-Wave Analog)

2025-1-4 Yuanlong Li

# Model
- Nodes: $i=1\dots N$ pendula with angles $\theta_i(t)$, $N=4$ or $5$.
- Dynamics (damped, driven, all-to-all coupling):
  $$
  \ddot{\theta}_i + \gamma \dot{\theta}_i + \omega_{0,i}^2 \sin \theta_i
  = \frac{K}{N}\sum_{j=1}^N \sin(\theta_j-\theta_i) + F\cos(\Omega t)\,\delta_{i,\mathrm{drive}}
  $$
- Drive: only one node (index `drive_index`, default 0) has the external force.
- Detune (symmetry breaking):
  - Base $\omega_0$ shared; optional detune factor `detune_eps`.
  - `detune_target` = 2: $\omega_{0,\mathrm{target}} = \omega_0(1+\mathrm{detune\_eps})$, others $\omega_0$.
- Damping: $\gamma$ identical across nodes.
- Coupling: uniform all-to-all with strength $K$.

# Numerics
- Integrator: `solve_ivp` (RK45), state $[\theta_1..\theta_N, \omega_1..\omega_N]$.
- Time grid: step `dt`, total `t_total`; discard initial fraction `discard_ratio` for steady state.
- Initial conditions: small random noise (seeded) on angles/velocities to avoid perfect symmetry.

## Observables
- RMS amplitude: $A_i = \sqrt{\langle \theta_i^2\rangle_t}$ over the steady tail.
- Main-frequency complex amplitude: $\tilde{A}_i = \langle \theta_i e^{-i\Omega t}\rangle_t$, report $|\tilde{A}_i|$.
- Selectivity (include drive): $\max A_i / \text{2nd max}$; same for $|\tilde{A}_i|$.
- Selectivity (exclude drive): computed over non-driven nodes only.

## Parameters (per run)
- $N,\ \gamma,\ K,\ F,\ \Omega,\ \omega_0,\ \text{detune\_eps},\ \text{detune\_target},\ \text{drive\_index},\ t_\text{total},\ dt,\ \text{discard\_ratio},\ \text{seed}$.

# Results
- **Heatmap — main-frequency selectivity (non-driven):**
  <img src="figures/heatmap_selectivity_fft_nd.png" style="justify-content:center;width:80%;display:flex;" />
  - Metric: `selectivity_fft_nd` = max/second main-frequency amplitude among non-driven nodes.
  - Scan: N=5, F=0.0025, detune_eps=0.12 on node 2, Ω∈[0.898,0.902], K∈[0.34,0.40].

# Results
- **Representative single-run θ(t):**
  ![](figures/single_case_1.png)
- ![](figures/single_case_2.png)
  - *Overlay of θ_i(t) for all nodes in one run .*
- **Top cases (bar plots from scan summary):**
  - RMS amplitudes A_i for the top three selectivity points in the current summary CSV. 
<div style="display:flex; justify-content:center; align-items:center; gap:12px; flex-wrap:nowrap;">
  <img src="figures/top1_Omega0.510_K1.000.png" style="width:25%;" />
  <img src="figures/top2_Omega0.520_K0.990.png" style="width:25%;" />
  <img src="figures/top3_Omega0.510_K0.990.png" style="width:25%;" />
</div>