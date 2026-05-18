# Model architecture

The full HAVI-Methyl torch loop composes four pieces: a
permutation-invariant **Set Transformer fragment-bag encoder**, a
**Gaussian or Conditional NSF posterior head** on the logit scale, a
**Beta-Binomial reconstruction** (or Bernoulli / Categorical / NB
depending on the observation regime), and **Robbins-Monro
recentering** on the population and sample shifts. This page maps each
piece to the manuscript and to the `TorchSVIConfig` field that controls
it.

## Variational family

Under the structured family of §4, the variational posterior factorises
as

$$
q(\bm{\mu}^{\mathrm{pop}},\bm{\delta},\bm{\eta}) =
\prod_\ell q_{\lambda_\ell}(\mu^{\mathrm{pop}}_\ell)\;
\prod_s q_{\nu_s}(\delta_s)\;
\prod_{s,\ell} q_\phi(\eta_{s,\ell}\mid F_{s,\ell};\,\bar m_\ell,\bar m_s^\delta).
$$

The population and sample-shift factors are mean-field Gaussian
($q_{\lambda_\ell}=\mathcal{N}(m_\ell,v_\ell)$ and likewise for
$\nu_s=(m_s^\delta,v_s^\delta)$); their stochastic natural-gradient
updates are approximate because the local layer is non-conjugate. The
per-(sample, locus) layer is a one-dimensional flow on the logit scale,
sampled by

$$
\eta_{s,\ell} = T_\phi(\epsilon;\,c_{s,\ell}),\qquad
\epsilon\sim\mathcal{N}(0,1),
$$

with $T_\phi$ either a Gaussian re-parameterisation
(`posterior="gaussian"`) or a stack of conditional neural-spline-flow
(NSF) blocks (`posterior="flow"`).

## Set Transformer encoder

The fragment bag $F_{s,\ell}=\{f_{s,\ell,j}\}$ is permutation-invariant.
The encoder is an *induced-set-attention block* (ISAB) stack followed
by *pooling-by-multi-head-attention* (PMA):

$$
c^{\mathrm{frag}}_{s,\ell} = \mathrm{PMA}_1\!\big(
\mathrm{ISAB}^{(L_e)}\circ\cdots\circ\mathrm{ISAB}^{(1)}(\{f_{s,\ell,j}\})\big).
$$

With $m=$ `num_inducing` inducing points, ISAB is $O(nm)$ instead of
$O(n^2)$ in fragments. The full encoder context concatenates the
fragment summary with sequence context, current variational means, and
log-coverage:

$$
c_{s,\ell} = [c^{\mathrm{frag}}_{s,\ell}\,\|\,c^{\mathrm{seq}}_\ell\,\|\,
\bar m_\ell\,\|\,\bar m_s^\delta\,\|\,\log(1+n^{\mathrm{frag}}_{s,\ell})].
$$

## Posterior head

The `_GaussianPosteriorHead` is a 2-layer MLP from the encoder context
to $(\mu_\phi,\log\sigma_\phi)$ with `log_sigma` clamped to $[-3, 3]$
for numerical stability. The Conditional NSF flow head (selected by
`posterior="flow"`) uses `num_bins+1` knots with conservative zero-init
so that the block starts as the identity map and never NaN-s out
during the early iterations.

## Reconstruction

For the paired-validation regime (the published Liu 2024 benchmark
target), the reconstruction term in the ELBO is the Beta-Binomial
log-pmf

$$
\log\BB(n^m;\,n^{\mathrm{cpg}},\,\kappa\beta,\,\kappa(1-\beta)) =
\log\!\binom{n^{\mathrm{cpg}}}{n^m} + \log B(n^m+\kappa\beta,\,n^{\mathrm{cpg}}-n^m+\kappa(1-\beta))
- \log B(\kappa\beta,\,\kappa(1-\beta)),
$$

with reparameterisation gradient through $\beta = \sigma(\eta)$ given by
the digamma identity in §5.2. Direct calls fall back to a Bernoulli
likelihood; sequence-conditioned 4-mer end motifs use a Categorical
head; locus coverage uses a Negative-Binomial with mean-dispersion
parameterisation.

## Robbins-Monro recentering

The additive form $\eta = \mu^{\mathrm{pop}} + \delta + \text{noise}$
is invariant under $(\mu^{\mathrm{pop}},\delta) \mapsto
(\mu^{\mathrm{pop}}+c,\delta-c)$. The loop enforces
$\sum_s \delta_s = 0$ at every iteration by computing
$\bar\delta = S^{-1}\sum_s m_s^\delta$ and then re-centring
$m_s^\delta \gets m_s^\delta - \bar\delta$,
$m_\ell \gets m_\ell + \bar\delta$. The correction is **global or
running**, not mini-batch-only. The `recentering_history` field of
`TorchSVIState` records the per-iteration shift.

Stochastic natural-gradient updates on the population and sample
factors use Robbins-Monro step sizes $\rho_t = (t+1)^{-r}$ with
$r = $ `rho_exponent` (default `0.6`):

$$
\lambda_\ell \gets (1-\rho_t)\lambda_\ell + \rho_t\,\hat\lambda_\ell^{\mathrm{nat}}.
$$

## `TorchSVIConfig` reference

The dataclass in `src/havi_methyl/torch_svi.py` exposes every
hyperparameter that affects the production-stack training loop. Every
field has a default that reproduces the headline real-data row when
paired with the standard Liu 2024 invocation.

| Field | Default | Meaning |
|---|---|---|
| `in_dim` | `5` | Fragment-feature dimension fed into the Set Transformer. |
| `hidden` | `32` | Hidden width of the ISAB / PMA blocks and the posterior head. |
| `num_inducing` | `16` | ISAB inducing-point count; trades attention memory for accuracy. |
| `num_layers` | `2` | Depth of the ISAB stack. |
| `kappa` | `20.0` | Beta-Binomial concentration $\kappa$, Eq. (4) of §3. |
| `sigma_eta` | `0.6` | Per-(sample,locus) prior std $\sigma_\eta$, Eq. (2) of §3. |
| `sigma_pop` | `2.0` | Population-prior std $\tau_0$, Eq. (1) of §3. |
| `sigma_delta` | `0.5` | Sample-shift prior std $\sigma_\delta$. |
| `lr` | `5e-3` | AdamW learning rate for the encoder + head. |
| `rho_exponent` | `0.6` | Robbins-Monro decay exponent $r$ in $\rho_t = (t+1)^{-r}$. |
| `batch_samples` | `4` | Mini-batch sample count $|B_s|$. |
| `batch_loci` | `32` | Mini-batch loci count $|B_\ell|$. |
| `posterior` | `"gaussian"` | `"gaussian"` for the headline run, `"flow"` for the Conditional NSF head. |
| `vib_weight` | `0.0` | VIB penalty weight $\beta_{\mathrm{VIB}}$ on prior-input leakage (§7). |
| `counterfactual_weight` | `0.0` | Counterfactual augmentation penalty weight $\lambda_{\mathrm{cf}}$. |
| `adversarial_weight` | `0.0` | **True gradient-reversal weight** $\lambda_{\mathrm{dom}}$ for the sample-id discriminator (see below). |
| `mqtl_anchors` | `None` | Optional tuple `(genotype, intercept, effect, anchor_idx)` for the mQTL IV loss. |
| `mqtl_weight` | `0.0` | mQTL anchor loss weight $\lambda_{\mathrm{mQTL}}$. |
| `k_iwae` | `1` | IWAE sample count $K$. `1` reproduces the standard reparam ELBO. |
| `iwae_dreg` | `False` | When `True` and $K>1$, applies the **Tucker-2019 doubly-reparameterised gradient estimator** (see below). |
| `device` | `"auto"` | `"auto"` picks `cuda > mps > cpu`; any explicit `torch.device` string is also accepted. |

## IWAE-DReG (`iwae_dreg=True`)

When `k_iwae=K>1`, the loop optimises the importance-weighted bound
\citep{burda2016iwae}

$$
\elbo_K = \E\!\left[\log\frac{1}{K}\sum_{k=1}^K\frac{p(\mathcal{O},\eta^{(k)})}{q_\phi(\eta^{(k)}\mid c)}\right]\ge \elbo_1.
$$

\citet{rainforth2018iwaeSNR} showed that increasing $K$ can degrade the
encoder gradient signal-to-noise ratio even as the bound tightens.
With `iwae_dreg=True`, the implementation evaluates
$\log q_\phi(\eta_k\mid x)$ with `(mu_q, log_sigma)` *detached*, so that
the encoder gradient through $\log q$ flows only through the
reparameterised path $\eta_k$ — this is the proper Tucker-2019
doubly-reparameterised gradient. Squared-importance-weight mixing
completes the estimator. Smoke test at $K=8$ shows ~10% ELBO-trajectory
variance reduction (Phase 1.4, recorded in
`outputs/tables/bench_torch_svi.csv`).

!!! note "K=1"
    DReG is undefined for $K=1$, so the toggle is no-op there. The
    headline real-data run uses $K=4$ standard IWAE without DReG; both
    paths are exposed for experimentation.

## Gradient-reversal adversarial head (`adversarial_weight > 0`)

The 2026-05-17 Phase 5 push replaced the previous context-variance proxy
with a true gradient-reversal head implemented as a
`torch.autograd.Function`: identity forward, sign-flipped scaled
gradient backward. A 2-layer MLP `_DomainDiscriminator` classifies the
encoder context into sample id; the discriminator trains normally via
AdamW, but the encoder receives the negated gradient and is pushed
toward a sample-invariant context. The loss multiplier $\lambda_{\mathrm{dom}}$
is `adversarial_weight`.

## Joint tissue-of-origin training

If the caller passes all three of `tissue_reference` (a $T \times L$
reference panel), `tissue_target` (the ground-truth $S \times T$
mixture), and `tissue_weight > 0`, the loop adds the variance-weighted
Dirichlet head as a joint training term. Each mini-batch solves a
differentiable `torch.linalg.lstsq` on the locus subset to recover
$\hat\pi$, then adds $\lambda_{\mathrm{ToO}}\|\hat\pi - \pi_{\mathrm{true}}\|^2$
to the total loss. Leaving `tissue_weight=0` recovers the post-hoc head
of §9. See [Tissue-of-origin](tissue.md).

## End-to-end loss

The total objective maximised by `fit_svi_torch` is

$$
\mathcal{J}_{\mathrm{total}} = \elbo
+ \lambda_{\mathrm{ToO}}\mathcal{L}_{\mathrm{ToO}}
- \beta_{\mathrm{VIB}}\mathcal{B}_{\mathrm{VIB}}
- \lambda_{\mathrm{cf}}\mathcal{L}_{\mathrm{cf}}
- \lambda_{\mathrm{mQTL}}\mathcal{L}_{\mathrm{mQTL}}
- \lambda_{\mathrm{dom}}\mathcal{L}_{\mathrm{dom}}.
$$

The PyTorch loop minimises $-\mathcal{J}_{\mathrm{total}}$ with the
corresponding sign changes; AdamW handles the neural parameters,
gradient clipping is at $g_{\mathrm{clip}}=5.0$, and KL annealing on
the local-layer KL is applied for the first $T_{\mathrm{anneal}}$
iterations.
