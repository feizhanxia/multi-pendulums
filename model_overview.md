# Driven All-to-All Pendulum Network (Minimal Neural-Wave Analog)

## Model
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
  <!-- - If `detune_target` >= 0: $\omega_{0,\mathrm{target}} = \omega_0(1+\mathrm{detune\_eps})$, others $\omega_0$. -->
  <!-- - If `detune_target` = -1: linear gradient around the drive node. -->
- Damping: $\gamma$ identical across nodes.
- Coupling: uniform all-to-all with strength $K$.

## Numerics
- Integrator: `solve_ivp` (RK45), state $[\theta_1..\theta_N, \omega_1..\omega_N]$.
- Time grid: step `dt`, total `t_total`; discard initial fraction `discard_ratio` for steady state.
- Initial conditions: small random noise (seeded) on angles/velocities to avoid perfect symmetry.

## Observables
- RMS amplitude: $A_i = \sqrt{\langle \theta_i^2\rangle_t}$ over the steady tail.
- Main-frequency complex amplitude: $\tilde{A}_i = \langle \theta_i e^{-i\Omega t}\rangle_t$, report $|\tilde{A}_i|$.
- Selectivity (include drive): $\max A_i / \text{2nd max}$; same for $|\tilde{A}_i|$.
- Selectivity (exclude drive): computed over non-driven nodes only.
- Targets recorded: argmax indices for RMS and main-frequency metrics.

## Parameters (per run)
- $N,\ \gamma,\ K,\ F,\ \Omega,\ \omega_0,\ \text{detune\_eps},\ \text{detune\_target},\ \text{drive\_index},\ t_\text{total},\ dt,\ \text{discard\_ratio},\ \text{seed}$.

## Workflow (files)
- `simulate.py`: single run; optional raw `.npz` save.
- `scan.py`: grid over $\Omega, K$ (and optionally $N$); parallel; writes CSV summary.
- `analyze.py`: plots heatmaps (default `selectivity_fft_nd`) and bar plots for top cases.
- `main.py`: toggles single run / scan / analyze with centralized arguments.
- Data: `data/summary/*.csv`; Figures: `figures/`.

## Expected Phenomenon
- Goal: for some $(\Omega, K)$, a non-driven node shows large steady amplitude (or main-frequency amplitude), while other non-driven nodes are suppressed via mode selection + phase interference.
- Symmetry breaking (detune or tiny biases) is required to pick a specific non-driven node in an otherwise symmetric network.

## Results
- Heatmap — main-frequency selectivity (non-driven):
  ![](figures/heatmap_selectivity_fft_nd.png)
  - Metric: `selectivity_fft_nd` = max/second main-frequency amplitude among non-driven nodes.
  - Scan: N=5, F=0.0025, detune_eps=0.12 on node 2, Ω∈[0.898,0.902], K∈[0.34,0.40].
- Representative single-run θ(t):
  ![](figures/single_case.png)
  - Overlay of θ_i(t) for all nodes in one run (from `simulate.py --plot_path`).
<!-- - Top cases (bar plots from scan summary):
  ![](figures/top1_Omega0.510_K1.000.png)
  ![](figures/top2_Omega0.520_K0.990.png)
  ![](figures/top3_Omega0.510_K0.990.png)
  - RMS amplitudes A_i for the top three selectivity points in the current summary CSV. -->
