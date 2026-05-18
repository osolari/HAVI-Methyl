# Tissue-of-origin

HAVI-Methyl ships a **variance-weighted Dirichlet head** that consumes
the posterior `(mean, var)` of $\beta_{s,\ell}$ — *not* binarised point
estimates — and outputs a Dirichlet posterior on tissue fractions for
each sample. The head can run **post-hoc** on a fitted variational
state, or **jointly** inside `fit_svi_torch` as an extra loss term.

## Head specification (§9)

For a reference panel $R\in[0,1]^{L\times T}$ ($R_{\ell t}$ is the
reference methylation level of tissue $t$ at locus $\ell$), the head
outputs

$$
\bm\pi_s\sim\mathrm{Dir}(\bm\alpha_s),\qquad
\bm\alpha_s = \mathrm{softplus}\!\big(W_R\,\E_q[\bm\beta_s] + b_R\big) + \epsilon_\alpha,
$$

with a small $\epsilon_\alpha > 0$ for numerical stability. The
expected Gaussian reference log-likelihood under
$q(\beta_{s,\ell})\approx\mathcal{N}(\hat\mu_{s,\ell},\hat\sigma^2_{s,\ell})$
is

$$
\mathcal{L}_{\mathrm{ToO}}(s) = -\frac{1}{2}\sum_\ell\!\left[\log(2\pi\sigma_R^2)+\frac{(\hat\mu_{s,\ell}-R_{\ell\cdot}\bar\pi_s)^2+\hat\sigma^2_{s,\ell}}{\sigma_R^2}\right],
$$

i.e. an *expected* log-likelihood — not a marginal Gaussian likelihood
with inflated variance. Per-CpG uncertainty is propagated into the
tissue-fraction objective instead of being thrown away by binarisation.

## Post-hoc API

```python
from havi_methyl import dirichlet_head_predict

# pred_mean, pred_var: (S, L) posterior summaries from predict_with_torch_state.
# R: (T, L) reference panel from load_loyfer_atlas_matrix.
pi_hat = dirichlet_head_predict(pred_mean, pred_var, R)
# pi_hat: (S, T) deconvolved tissue fractions.
```

## Joint training inside `fit_svi_torch`

The 2026-05-17 Phase 5 push wired the head as an optional joint loss.
Pass all three of `tissue_reference`, `tissue_target`, and
`tissue_weight > 0` to `fit_svi_torch`:

```python
state = fit_svi_torch(
    bags=ds.bags,
    n_frag=ds.n,
    n_meth=ds.n_meth,
    n_obs=ds.n_total,
    n_iter=200,
    config=cfg,
    tissue_reference=R,          # (T, L) reference panel
    tissue_target=pi_true,       # (S, T) ground-truth mixture
    tissue_weight=1.0,           # lambda_ToO; 0 recovers post-hoc head
)
```

On each mini-batch the head solves a differentiable
`torch.linalg.lstsq` on the locus subset to recover $\hat\pi$, then adds
$\lambda_{\mathrm{ToO}}\|\hat\pi - \pi_{\mathrm{true}}\|^2$ to the total
loss. Leaving `tissue_weight=0` (the default) recovers the post-hoc
head of §9.

## HDP nonparametric extension

For novel-tissue discovery, §9.2 replaces the finite Dirichlet by a
hierarchical Dirichlet process \citep{teh2006hdp}: the tissue mixture is
$\sum_k \pi_{s,k}\delta_{\theta_k}$ with $\bm\pi_s\sim\mathrm{GEM}(\alpha)$.
`hdp_truncated_deconvolve` implements a variational truncation at
$T_{\max} = 64$. On the Loyfer U25 panel HDP lands between
FinaleMe-binarised and continuous lstsq — see the LOO numbers below.

## Loyfer LOO measured result

From `outputs/tables/bench_tissue_loo.csv` (loaded from
`data/loyfer_panel/Atlas.U25.l4.hg38.tsv`, $T = 36$ cell types, $L = 900$
markers):

| Method | In-panel RMSE | Worst-tissue RMSE | LOO mean RMSE | LOO worst RMSE |
|---|---:|---:|---:|---:|
| FinaleMe-binarized + QP | 0.0367 | 0.0450 | 0.0377 | 0.0381 |
| Continuous lstsq | 0.0273 | 0.0364 | 0.0280 | 0.0284 |
| **HAVI-Methyl Dirichlet head** | **0.0169** | **0.0265** | **0.0174** | **0.0178** |
| HDP-truncated ($T_{\max}=64$) | 0.0329 | 0.0549 | 0.0347 | 0.0363 |

The variance-weighted Dirichlet head wins every column.

![Loyfer LOO RMSE](assets/figures/loyfer_loo_rmse.png)

## Per-tissue breakdown — 36/36

The per-tissue LOO RMSE is in
`docs/report/tables/bench_loyfer_loo_per_tissue.csv`. **HAVI-Methyl
Dirichlet head wins LOO RMSE at every one of the 36 cell types.** The
median advantage over continuous least squares is $+0.011$, and the
worst case (Eryth-prog) trails the leader by only $0.011$.

![Loyfer LOO per tissue](assets/figures/loyfer_loo_per_tissue.png)

The bars are sorted by HAVI's advantage over continuous lstsq; orange
is shortest at every tissue, including the rightmost (most-adverse-for-HAVI)
column.

## How to read this

The cost of binarisation in the FinaleMe-binarised pipeline is the
$0.037$ vs $0.017$ gap (a $2.2\times$ RMSE difference). The cost of
**not propagating posterior uncertainty** in the continuous lstsq
baseline is the $0.028$ vs $0.017$ gap (a $1.6\times$ RMSE
difference). The variance-weighted head closes both — that is the §9
empirical claim.
