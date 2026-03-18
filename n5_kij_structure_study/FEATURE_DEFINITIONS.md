# N=5 Kij Structure Study: Feature Definitions

This note collects the mathematical definitions of the structural and
response-level features used in the current `N=5` `K_{ij}` structure study.

Throughout this document:

- `K \in \mathbb{R}^{N \times N}` is the coupling matrix
- `N = 5`
- `K_{ii} = 0`
- `K_{ij}` means the influence from node `j` to node `i`
- `d` is the driven node
- `\mathcal{N}_d = \{0,\dots,N-1\}\setminus\{d\}`
- `m` is the coarse-stage dominant non-driven node

That is,

$$
m
=
\arg\max_{i \in \mathcal{N}_d} A_i(K,\Omega_{\mathrm{best}}).
$$

## Response Quantities

For a fixed pair `(K,\Omega)`, define the steady-state Fourier response
amplitude at the driving frequency by

$$
A_i(K,\Omega)
=
\left|
\frac{1}{T_{\mathrm{tail}}}
\int_{t \in \mathrm{tail}}
\theta_i(t)e^{-i\Omega t}\,dt
\right|.
$$

The selectivity metric is

$$
S(K,\Omega)
=
\frac{
\max_{i \in \mathcal{N}_d} A_i(K,\Omega)
}{
\max_{j \in \mathcal{N}_d \setminus \{m^*(K,\Omega)\}} A_j(K,\Omega) + \varepsilon
},
$$

where

$$
m^*(K,\Omega)=\arg\max_{i \in \mathcal{N}_d} A_i(K,\Omega).
$$

The coarse best selectivity is

$$
S_{\mathrm{coarse}}(K)
=
\max_{\Omega \in \Omega_{\mathrm{coarse}}} S(K,\Omega).
$$

## Matrix Norms and Nonreciprocity

The Frobenius norm of the coupling matrix is

$$
\|K\|_F
=
\sqrt{\sum_{i=1}^{N}\sum_{j=1}^{N} K_{ij}^2}.
$$

The antisymmetric mismatch is

$$
\Delta_{\mathrm{asym}}(K)
=
K-K^{\mathsf T}.
$$

Its Frobenius norm is

$$
\|K-K^{\mathsf T}\|_F
=
\sqrt{\sum_{i=1}^{N}\sum_{j=1}^{N}(K_{ij}-K_{ji})^2}.
$$

This quantity is the absolute nonreciprocity strength.

The normalized nonreciprocity used in the analysis figures is

$$
\eta_{\mathrm{nr}}(K)
=
\frac{\|K-K^{\mathsf T}\|_F}{\|K\|_F + \varepsilon}.
$$

In the code, this is the feature named `asymmetry_ratio`.

## Metadata and Response-Level Features

- `kidx`  
  Integer index of the sampled matrix.

- `coarse_rank`  
  Rank after sorting all coarse candidates by `best_selectivity` in descending
  order.

- `best_omega`  
  Frequency that maximizes coarse selectivity for the candidate.

- `best_selectivity`  

$$
\max_{\Omega \in \Omega_{\mathrm{coarse}}} S(K,\Omega).
$$

- `target_idx`  
  Index of the dominant non-driven node at `best_omega`.

- `target_amp`  

$$
A_m(K,\Omega_{\mathrm{best}}).
$$

- `second_amp`  
  The second-largest non-driven amplitude at `best_omega`.

- `target_dominance_coarse`  

$$
\frac{\mathrm{target\_amp}}{\mathrm{second\_amp} + \varepsilon}.
$$

- `coarse_target_share`  

$$
\frac{A_m(K,\Omega_{\mathrm{best}})}{\sum_{i \in \mathcal{N}_d} A_i(K,\Omega_{\mathrm{best}}) + \varepsilon}.
$$

- `final_avg_selectivity`  
  Mean selectivity across final verification seeds.

- `final_min_selectivity`  
  Minimum selectivity across final verification seeds.

## Sign and Magnitude Features

Let

$$
M = \#\{(i,j): i \neq j,\; K_{ij}\neq 0\}.
$$

- `fro_norm`  

$$
\|K\|_F.
$$

- `asymmetry_norm`  

$$
\|K-K^{\mathsf T}\|_F.
$$

- `asymmetry_ratio`  

$$
\frac{\|K-K^{\mathsf T}\|_F}{\|K\|_F + \varepsilon}.
$$

- `positive_ratio`  

$$
\frac{\#\{K_{ij}>0,\; i\neq j\}}{\#\{K_{ij}\neq 0,\; i\neq j\}}.
$$

- `negative_ratio`  

$$
\frac{\#\{K_{ij}<0,\; i\neq j\}}{\#\{K_{ij}\neq 0,\; i\neq j\}}.
$$

- `mean_abs_weight`  

$$
\frac{1}{M}\sum_{i\neq j}|K_{ij}|.
$$

- `std_abs_weight`  
  Standard deviation of the off-diagonal absolute values `|K_{ij}|`.

- `max_abs_weight`  

$$
\max_{i\neq j}|K_{ij}|.
$$

## Node Strength Features

Because `K_{ij}` denotes coupling from `j` to `i`, row sums are incoming
strengths and column sums are outgoing strengths.

Define

$$
I_i(K)=\sum_{j=1}^{N}|K_{ij}|,
\qquad
O_i(K)=\sum_{j=1}^{N}|K_{ji}|.
$$

Then:

- `drive_in_strength`  

$$
I_d(K).
$$

- `drive_out_strength`  

$$
O_d(K).
$$

- `target_in_strength`  

$$
I_m(K).
$$

- `target_out_strength`  

$$
O_m(K).
$$

- `target_sink_bias`  

$$
I_m(K)-O_m(K).
$$

Positive `target_sink_bias` means the selected target receives more total
coupling than it sends out, in absolute-value sense.

## Directed Path Features

The direct gain proxy from the driven node to node `i` is

$$
P^{(1)}_{d\to i}(K)=|K_{id}|.
$$

The two-hop gain proxy is

$$
P^{(2)}_{d\to i}(K)
=
\sum_{r \neq d,i}|K_{rd}K_{ir}|.
$$

The three-hop gain proxy is

$$
P^{(3)}_{d\to i}(K)
=
\sum_{r_1 \neq d,i}\sum_{r_2 \neq d,i,r_1}
|K_{r_1 d}K_{r_2 r_1}K_{i r_2}|.
$$

Using these proxies, the code defines:

- `target_direct_from_drive`  

$$
P^{(1)}_{d\to m}(K).
$$

- `target_path_2hop`  

$$
P^{(2)}_{d\to m}(K).
$$

- `target_path_3hop`  

$$
P^{(3)}_{d\to m}(K).
$$

Let the best competing non-driven node under the same path metric be

$$
P^{(\ell)}_{\mathrm{comp}}(K)
=
\max_{r \in \mathcal{N}_d \setminus \{m\}} P^{(\ell)}_{d\to r}(K).
$$

Then:

- `target_path_advantage_2hop`  

$$
\frac{P^{(2)}_{d\to m}(K)}{P^{(2)}_{\mathrm{comp}}(K)+\varepsilon}.
$$

- `target_path_advantage_3hop`  

$$
\frac{P^{(3)}_{d\to m}(K)}{P^{(3)}_{\mathrm{comp}}(K)+\varepsilon}.
$$

These features measure whether the selected target has a stronger short
directed path gain than competing non-driven nodes.

## Feature Score Used in `analysis_overview`

To rank structural features by class-separation ability, the code uses the
normalized mean-gap score

$$
\mathrm{score}(f)
=
\frac{
\max_{\ell}\mu_{\ell}(f)-\min_{\ell}\mu_{\ell}(f)
}{
\sigma_{\mathrm{all}}(f)+\varepsilon
},
$$

where:

- `f` is a feature
- `\ell` runs over the sample labels
- `\mu_{\ell}(f)` is the mean of feature `f` inside label `\ell`
- `\sigma_{\mathrm{all}}(f)` is the global standard deviation of feature `f`

Larger values mean that the feature separates the labeled sample classes more
strongly relative to its overall spread.
