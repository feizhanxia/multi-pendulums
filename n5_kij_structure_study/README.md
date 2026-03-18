# N=5 Nonreciprocal Kij Structure Study

## Purpose

This directory is used for the next-stage study built on top of
`n5_kij_research_clean/`.

The objective is no longer only to find a high-scoring coupling matrix
`K = (K_{ij})`, but to characterize which structural properties of `K`
produce stable directional energy transfer and single-node selective
excitation in the current forced nonlinear pendulum network.

Suggested directory usage:

- `code/`: analysis scripts, feature extraction, ablation studies
- `results/`: CSV/JSON summaries, figures, notes, intermediate outputs

For the precise definitions of the structural features used in the analysis,
see `FEATURE_DEFINITIONS.md`.

## Current Model

We study a network of `N = 5` damped, driven, nonlinearly coupled pendula.
The state variables are

\[
\theta_i(t), \qquad i \in \{1,\dots,N\}.
\]

The dynamics are

\[
\ddot{\theta}_i + \gamma \dot{\theta}_i + \omega_0^2 \sin \theta_i
=
\sum_{j=1}^{N} K_{ij}\sin(\theta_j-\theta_i)
+ \delta_{i,d} F \cos(\Omega t),
\]

where:

- `d` is the driven node
- `F` is the forcing amplitude
- `\Omega` is the driving frequency
- `\gamma` is the damping coefficient
- `\omega_0` is the natural frequency
- `K = (K_{ij}) \in \mathbb{R}^{N \times N}` is the coupling matrix
- `K_{ii} = 0`
- in general `K_{ij} \neq K_{ji}`, so the coupling is nonreciprocal

For the current project baseline, we mainly consider fixed

\[
N=5, \quad d=1 \text{ (or index 0 in code)}, \quad \gamma=0.08, \quad
F=0.1, \quad \omega_0=1.
\]

## Steady-State Response and Selectivity

For a given pair `(K, \Omega)`, let the steady-state response amplitude at the
driving frequency be

\[
A_i(K,\Omega)
=
\left|
\frac{1}{T_{\mathrm{tail}}}
\int_{t \in \mathrm{tail}}
\theta_i(t)e^{-i\Omega t}\,dt
\right|,
\]

which in code is approximated by the Fourier component of the tail segment.

Let the set of non-driven nodes be

\[
\mathcal{N}_d = \{1,\dots,N\}\setminus\{d\}.
\]

Define the dominant non-driven node

\[
m^*(K,\Omega)
=
\arg\max_{i \in \mathcal{N}_d} A_i(K,\Omega).
\]

Define the current selectivity metric

\[
S(K,\Omega)
=
\frac{
\max_{i \in \mathcal{N}_d} A_i(K,\Omega)
}{
\max_{i \in \mathcal{N}_d \setminus \{m^*\}} A_i(K,\Omega)
}.
\]

If initial conditions or random seeds are represented by `\xi \in \Xi`, then
the robust selectivity is

\[
S_{\min}(K,\Omega)
=
\min_{\xi \in \Xi} S(K,\Omega;\xi).
\]

## Research Objective

The main research problem is to characterize the set

\[
\mathcal{K}_{\mathrm{sel}}
=
\left\{
K:
\exists \Omega \text{ such that }
S_{\min}(K,\Omega) > S_{\mathrm{th}}
\right\},
\]

where `S_th` is a chosen robustness threshold, for example `S_th = 2`.

Equivalently, we seek to answer:

1. Which structural properties of `K` make `K \in \mathcal{K}_{\mathrm{sel}}`?
2. Why do such `K` produce directional energy transfer to one non-driven node?
3. Which observed high-scoring `K` are genuinely robust, and which are only
   fragile peak cases?

## Mathematical Reformulation of the Mechanism Question

The mechanism question can be stated as:

\[
\text{Given } K,\text{ why does there exist a node } m^*
\text{ such that }
A_{m^*}(K,\Omega)
\gg
A_j(K,\Omega), \quad \forall j \in \mathcal{N}_d,\; j \neq m^* \, ?
\]

This can be decomposed into three candidate mechanisms:

### 1. Nonreciprocal transport bias

\[
K_{ij} \neq K_{ji}
\quad \Longrightarrow \quad
\text{direction-dependent transfer}
\]

and thus a preferred propagation direction from the driven node toward a target
node.

### 2. Path-dominance mechanism

There may exist one or several dominant directed paths

\[
d \to i_1 \to \cdots \to m
\]

whose effective gain is larger than competing paths toward other nodes.

### 3. Localization / interference mechanism

At the working frequency `\Omega`, the network may support a response pattern
such that

\[
A_m(K,\Omega) \text{ is amplified}, \qquad
A_j(K,\Omega) \text{ is suppressed for } j \neq m,
\]

because of mode localization, phase alignment, or destructive interference on
competing nodes.

## Literature-Guided Principles

The following principles are not exact solutions for the present model. They
are working hypotheses translated from related literature.

### Principle A: Nonreciprocity is likely necessary

Related work on mechanical nonreciprocity indicates that directional transport
requires symmetry breaking. For the present model, this suggests that matrices
with

\[
\|K-K^{\mathsf T}\| \gg 0
\]

should be more promising than nearly symmetric matrices.

Working implication:

\[
K \approx K^{\mathsf T}
\quad \Rightarrow \quad
\text{less likely to produce strong single-node selectivity.}
\]

### Principle B: The target node should behave like a sink

If node `m` is the selected target, then successful matrices may satisfy an
effective inflow bias such as

\[
I_m(K) > O_m(K),
\]

where a simple proxy is

\[
I_m(K) = \sum_{j \neq m} |K_{mj}|, \qquad
O_m(K) = \sum_{j \neq m} |K_{jm}|.
\]

This is not a theorem; it is a testable design hypothesis.

### Principle C: Directed-path bias should favor one target

For a candidate target node `m`, define a path score proxy

\[
P_{d \to m}^{(2)}(K)
=
\sum_{j \neq d,m} |K_{jd}K_{mj}|,
\]

or more generally a weighted sum over short directed paths from `d` to `m`.

The design expectation is

\[
P_{d \to m}^{(\ell)}(K)
\gg
P_{d \to r}^{(\ell)}(K)
\quad \text{for competing nodes } r \neq m.
\]

### Principle D: High selectivity may require suppression of competitors

The desired condition is not only

\[
A_m(K,\Omega) \text{ large},
\]

but also

\[
A_r(K,\Omega) \text{ small for } r \neq m.
\]

Thus a good `K` may encode both:

- amplification toward the target node
- destructive competition among non-target nodes

### Principle E: The phenomenon is frequency-conditional

The object of interest is not `K` alone, but the pair `(K,\Omega)`.
Formally, one should study

\[
(K,\Omega)^*
=
\arg\max_{K,\Omega} S_{\min}(K,\Omega)
\]

or, more realistically, characterize the subset

\[
\mathcal{R}_{\mathrm{sel}}
=
\{(K,\Omega): S_{\min}(K,\Omega) > S_{\mathrm{th}}\}.
\]

This is consistent with the literature on targeted energy transfer,
localization, and nonlinear resonance locking.

## Proposed Feature Language for This Study

To study which `K` are effective, the matrix should be mapped to a feature
vector `\Phi(K)`. A first useful choice is

\[
\Phi(K)
=
\Big(
\|K\|_F,\;
\|K-K^{\mathsf T}\|_F,\;
\rho(K),\;
\text{positive-edge ratio},\;
\text{negative-edge ratio},\;
I_i,\;
O_i,\;
P_{d \to i}^{(2)},\;
P_{d \to i}^{(3)}
\Big)_{i=1}^{N}.
\]

Then the study becomes:

\[
\Phi(K)
\mapsto
\big(
S_{\max}(K),\;
S_{\min}(K),\;
m^*(K),\;
\Omega^*(K)
\big),
\]

where

\[
S_{\max}(K) = \max_{\Omega} S(K,\Omega), \qquad
\Omega^*(K) = \arg\max_{\Omega} S(K,\Omega).
\]

## What This README Commits Us To

This study is centered on the following statement:

\[
\text{Identify structural signatures of } K
\text{ that robustly induce single-node selective excitation.}
\]

Operationally, this means:

- compare stable high-selectivity matrices against unstable peak cases
- extract structural features from `K`
- test whether nonreciprocity, directed-path bias, and localization proxies
  predict robust selectivity
- verify candidate mechanisms by ablation:

\[
K
\;\to\;
\frac{K+K^{\mathsf T}}{2},
\qquad
K
\;\to\;
\frac{K-K^{\mathsf T}}{2},
\qquad
K
\;\to\;
\text{edge-pruned or sign-shuffled variants}
\]

and observe the resulting changes in `S(K,\Omega)`.

## References and Guidance Sources

These works do not solve the present model directly, but they motivate the
guiding principles above:

1. Kopidakis, Aubry, Tsironis, "Targeted Energy Transfer through Discrete
   Breathers in Nonlinear Systems", Phys. Rev. Lett. 87, 165501 (2001).
   https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.87.165501
2. Sato et al., "Observation of Locked Intrinsic Localized Vibrational Modes in
   a Micromechanical Oscillator Array", Phys. Rev. Lett. 90, 044102 (2003).
   https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.90.044102
3. Shaat, "Nonreciprocal elasticity and the realization of static and dynamic
   nonreciprocity", Scientific Reports 10, 2020.
   https://www.nature.com/articles/s41598-020-77949-4
4. Floquet-network and nonreciprocal transport literature, especially
   Phys. Rev. Applied 9, 044031 (2018).
   https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.9.044031
5. Nonlinearity-enhanced mode localization in coupled resonators, Int. J. Mech.
   Sci. (2024).
   https://www.sciencedirect.com/science/article/pii/S0020740324001760
6. "Coherent energy transfer in coupled nonlinear microelectromechanical
   resonators", Nature Communications (2025).
   https://www.nature.com/articles/s41467-025-59292-2
7. Asymmetry-enhanced targeted energy transfer / nonlinear energy sink
   literature, e.g. Commun. Nonlinear Sci. Numer. Simul. (2023).
   https://www.sciencedirect.com/science/article/abs/pii/S1007570422003859
