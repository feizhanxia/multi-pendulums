# K_ij Search Summary (N=3)

## Goal
Design an asymmetric coupling matrix K_ij (allowing positive/negative values) so that different drive-frequency bands selectively excite node 1 vs node 2, while the drive remains on node 0. The selection metric is based on the main-frequency response (amp_fft), not RMS.

## Dynamics
$$
\ddot{\theta}_i + \gamma \dot{\theta}_i + \omega_0^2 \sin \theta_i
= \sum_{j=1}^N K_{ij}\sin(\theta_j-\theta_i) + F\cos(\Omega t)\,\delta_{i,\mathrm{drive}}
$$

## Search Method
- **Optimization**: Bayesian optimization over the 6 off-diagonal entries of K_ij.
- **Objective**: maximize
$$
\mathrm{score}=\sqrt{\mathrm{win\_ratio}_a\cdot \mathrm{win\_ratio}_b}\times
\frac{\mathrm{avg\_selectivity}_a+\mathrm{avg\_selectivity}_b}{2}
$$
  where win_ratio is the fraction of Omegas in a band where the target node wins, and avg_selectivity is the mean top-1/top-2 main-frequency ratio.
- **Omega scan**: coarse grid for speed; the best bands are discovered automatically with a sliding window.

## Current Best K_ij
The top-scoring K_ij (from the latest search run) has been copied into `main_kij.py` for reproducible scans and plots.

## Figures in `figures/kij_run`
- `kij_heatmap.png`  
  Heatmap of the selected K_ij matrix. Colors show the sign and magnitude of each directed coupling.

- `single_kij.png`  
  Time-series overlay of θ_i(t) for a single Omega. Useful for sanity checking phase relationships and steady-state behavior.

- `winner_selectivity_kij.png`  
  Scatter plot of Omega vs selectivity (main-frequency top-1/top-2), colored by winner (node 1 vs node 2). This visualizes the frequency regions where each node dominates.

- `band_report_kij.json`  
  Machine-readable summary of the two best frequency bands found automatically, including their Omega ranges, win ratios, and average selectivity.
