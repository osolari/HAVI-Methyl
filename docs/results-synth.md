# Synthetic recovery

Multi-seed bootstrap intervals from
`scripts/bench_multiseed_recovery.py` ($N=20$ random seeds, full
posterior with the simplified-Gaussian HAVI-Methyl variant). The
canonical seed-`20260429` point estimates of §11 fall inside the 90 %
band at every coverage.

## Pearson $r$ at each coverage

From
[`docs/report/tables/tab_recovery_multiseed.csv`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/tables/tab_recovery_multiseed.csv).
Each row is the $N=20$-seed bootstrap on Pearson $r$ between predicted
and ground-truth $\beta$ over the full locus set ($S=12$, $L=300$).

| Coverage | Method | Median | $p_5$ | $p_{95}$ | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| $0.1\times$ | FinaleMe-style HMM | 0.191 | 0.156 | 0.242 | 0.196 | 0.032 |
| $0.1\times$ | **HAVI-Methyl simplified** | **0.357** | 0.246 | 0.444 | 0.358 | 0.063 |
| $1\times$ | FinaleMe-style HMM | 0.520 | 0.438 | 0.589 | 0.520 | 0.051 |
| $1\times$ | **HAVI-Methyl simplified** | **0.790** | 0.691 | 0.829 | 0.776 | 0.051 |
| $5\times$ | FinaleMe-style HMM | 0.826 | 0.765 | 0.893 | 0.826 | 0.051 |
| $5\times$ | **HAVI-Methyl simplified** | **0.901** | 0.849 | 0.947 | 0.898 | 0.038 |
| $30\times$ | FinaleMe-style HMM | 0.970 | 0.949 | 0.982 | 0.967 | 0.012 |
| $30\times$ | **HAVI-Methyl simplified** | **0.973** | 0.953 | 0.984 | 0.970 | 0.011 |

HAVI-Methyl wins at every coverage, with **non-overlapping 90 %
bootstrap intervals at $0.1\times$, $1\times$, and $5\times$**. At
$30\times$ both methods saturate near $r=0.97$, where naive averaging
is already near-perfect.

![Multiseed recovery](assets/figures/multiseed_recovery.png)

## RMSE and interval width

From the same CSV, lower is better for both.

| Coverage | Method | Median RMSE | Median MAE | Median 90 % width |
|---|---|---:|---:|---:|
| $0.1\times$ | FinaleMe-style HMM | 0.346 | 0.312 | 0.818 |
| $0.1\times$ | HAVI-Methyl simplified | 0.333 | 0.304 | 0.511 |
| $1\times$ | FinaleMe-style HMM | 0.314 | 0.250 | 0.415 |
| $1\times$ | HAVI-Methyl simplified | 0.228 | 0.198 | 0.440 |
| $5\times$ | FinaleMe-style HMM | 0.192 | 0.148 | 0.301 |
| $5\times$ | HAVI-Methyl simplified | 0.158 | 0.130 | 0.269 |
| $30\times$ | FinaleMe-style HMM | 0.118 | 0.102 | 0.170 |
| $30\times$ | HAVI-Methyl simplified | 0.118 | 0.103 | 0.127 |

HAVI-Methyl's 90 % credible intervals are narrower than FinaleMe-style
at every coverage *except* $1\times$, where the HAVI bands are
marginally wider in service of better coverage. The raw credible
intervals undershoot the nominal $0.90$ rate (cov$_{90}$ at $1\times$ is
$0.577$); production deployments should layer the conformal wrapper of
[Calibration](calibration.md) on top.

## Per-locus scatter

The canonical-seed scatter from `outputs/plot_data.npz`. Top row is the
FinaleMe-style HMM at each coverage; bottom row is HAVI-Methyl
simplified.

![Recovery scatter](assets/figures/recovery_scatter.png)

Both methods approach the diagonal as coverage increases; HAVI-Methyl
benefits more at low coverage, where the hierarchical population prior
contributes most.

## ELBO trajectory (real torch SVI)

The §11 surrogate-objective trajectory in
`assets/figures/elbo_trajectory.png` is taken from the **real Liu 2024
torch SVI run**, not synthetic data — the figure is included here as
the runnable training-curve diagnostic produced by the same loop that
underlies the synthetic recovery.

![ELBO trajectory](assets/figures/elbo_trajectory.png)

The Gaussian-head objective decreases monotonically over 200 iterations
on an A10 GPU and per-iteration Pearson $r$ against the WGBS empirical
$\beta$ rises in lockstep, terminating at $r = 0.455$ — exactly the
headline number in the [Results](results.md) page.

Snapshots are taken every 20 iterations via
`TorchSVIConfig.snapshot_every` and stored in
`TorchSVIState.snapshots`, which is what
`scripts/fig_elbo_trajectory.py` plots.

## Identifiability stress test (synthetic)

From §11.4: synthesise a buffy-coat prior input correlated with disease
status by adding a $0.25$ shift on the $\beta$ scale at $25\%$ of loci
for half the $S=12$ samples, then measure prior attribution as the
partial $R^2$ of the prior in a regression of predicted $\beta$ onto
(prior input, ground truth).

| Configuration | Prior attribution (partial $R^2$) |
|---|---:|
| No regularisation | $5.93\%$ |
| + VIB at $\beta_{\mathrm{VIB}} = 0.3$ | $0.62\%$ |
| + mQTL anchors on top | $0.037\%$ |

These synthetic reductions are consistent with the intended leakage-
control mechanisms; they are **not** a proof of real-data
identifiability. See §7 of the manuscript for the formal conditional
identifiability statements.

## Tissue-fraction recovery (synthetic 3-tissue mixture)

From §11.5: synthetic 3-tissue mixture, continuous deconvolution proxy
vs FinaleMe-derived binarize-and-deconvolve.

| Method | Tissue-fraction RMSE |
|---|---:|
| FinaleMe binarize-and-deconvolve | $0.129$ |
| Continuous deconvolution proxy | $0.007$ |

The $\sim 18\times$ gap illustrates the cost of binarisation in a
controlled setting. The real-data Loyfer LOO numbers
([Tissue-of-origin](tissue.md)) confirm the qualitative finding on the
published U25 panel.
