# Concepts

This page introduces the mental model that makes the HAVI-Methyl API
natural. The reference treatment is the manuscript at
[`docs/report/main.pdf`](report.md); here we summarise the three ideas
that recur throughout.

## 1. Three observation regimes — keep them distinct

The manuscript and the codebase are scrupulous about not conflating
three observation regimes; ignoring the distinction is what produces
spurious negative correlations on real data.

| Regime | Observation | Implementation hook |
|---|---|---|
| **Direct methylation** | Per-CpG bisulfite or nanopore call $y_{s,\ell,j,k}\in\{0,1\}$ | Bernoulli likelihood, Eq. (3) of §3 |
| **Paired-validation / pseudo-count** | Locus-level methylated count $n^m_{s,\ell}$ and total CpG trials $n^{\mathrm{cpg}}_{s,\ell}$ from a separate caller or assay | Beta-Binomial pseudo-likelihood $\BB(n^m;\,n^{\mathrm{cpg}},\,\kappa\beta,\,\kappa(1-\beta))$, Eq. (4) of §3, with concentration $\kappa$ absorbing classifier uncertainty |
| **WGS-only deployment** | Fragment bag $F_{s,\ell}=\{f_{s,\ell,j}\}$ only; methylation is inferred | Encoder context only; reconstruction omitted, validation against held-out truth when available |

The Liu 2024 evaluation in §12.4 is the **paired-validation** regime:
$n^m_{s,\ell}$ is the number of methylated reads in the WGBS BAM and
$n^{\mathrm{cpg}}_{s,\ell}$ is the WGBS *read coverage* at the CpG —
**not** the WGS fragment count, which is a separate stream feeding only
the encoder context. The loader exposes this explicitly:
`load_finaleme_dataset` returns both `ds.n` (WGS fragment counts,
$S\times L$) and `ds.n_total` (WGBS coverage, $S\times L$), and only the
latter belongs in the Beta-Binomial trials slot. Conflating the two
collapses the full-torch posterior to $r \approx 0$; the fix is what
lifts HAVI-Methyl from $r = -0.07$ to $r = 0.467$ on Liu 2024. See the
[Architecture](architecture.md) page for the exact `n_obs=ds.n_total`
contract.

## 2. cfDNA fragmentomics carries methylation signal

cfDNA fragments are not random shears — they are produced by a
programmed combination of apoptotic caspase-activated DNase, DNase1L3,
and other endonucleases acting on chromatinised DNA. The resulting
fragment-length distribution carries a ~10.4 bp periodicity reflecting
nucleosome rotational positioning. HAVI-Methyl uses the same feature
set as the FinaleMe baseline:

- Fragment length, with the canonical 167 bp mononucleosomal mode and
  secondary peaks near di- and tri-nucleosomal multiples.
- 4-mer end motifs at the 5' and 3' ends.
- GC content.
- Orientation (relative to TSS).
- Mappability.
- Inferred distance to the nearest nucleosome centre.

These features enter the **encoder** through a permutation-invariant
Set Transformer (ISAB + PMA) that produces a per-locus context vector
$c^{\mathrm{frag}}_{s,\ell}$. The generative observation model is
*separate* from the recognition model; the encoder operates on the
fragment bag $F_{s,\ell}$, but the likelihood is paid only on observed
methylation modalities $\mathcal{O}_{s,\ell}$. Conflating the two is
exactly the bug that produces spurious correlations.

## 3. A three-level hierarchy on logit-$\beta$

HAVI-Methyl places a three-level Gaussian hierarchy directly on the
logit-methylation latent $\eta_{s,\ell} = \logit(\beta_{s,\ell})$:

$$
\begin{aligned}
\mu^{\mathrm{pop}}_\ell &\sim \mathcal{N}(\mu_0,\,\tau_0^2),
 \qquad \ell = 1,\dots,L \quad \text{(population locus prior)} \\
\delta_s &\sim \mathcal{N}(0,\,\sigma_\delta^2),
 \qquad s = 1,\dots,S \quad \text{(per-sample shift)} \\
\eta_{s,\ell}\mid \mu^{\mathrm{pop}}_\ell,\delta_s
 &\sim \mathcal{N}\!\big(\mu^{\mathrm{pop}}_\ell + \delta_s,\,\sigma_\eta^2\big),
 \quad \beta_{s,\ell} = \sigma(\eta_{s,\ell}).
\end{aligned}
$$

The Gaussian prior on the logit scale gives heavier tails near
$\{0,1\}$ than a Beta prior would, which is useful for hyper- and
hypo-methylated loci. The additive form is invariant under the shift
$\mu^{\mathrm{pop}}_\ell \mapsto \mu^{\mathrm{pop}}_\ell + c$,
$\delta_s \mapsto \delta_s - c$, so the implementation enforces
$\sum_s \delta_s = 0$ by a **global or running** Robbins-Monro
recentering correction at every iteration (not a mini-batch-only
correction — that lesson is in the manuscript).

### Why hierarchical pooling helps

The population prior pools statistical strength across samples at the
same locus. The information-theoretic statement (Sec. 13 of the
manuscript) is that the posterior variance of $\beta_{s,\ell}$ under
the hierarchical model is upper-bounded by

$$
\mathrm{Var}(\beta_{s,\ell}\mid \text{data}) \le
\frac{\sigma_\eta^2}{S} + \tau_0^2\Big(1 - \frac{1}{1 + \sigma_\eta^2/(S\,\tau_0^2)}\Big),
$$

which is strictly smaller than the per-sample posterior variance for
any $S > 1$. `havi_methyl.bounds.hierarchical_pooling_variance` and
`hierarchical_pooling_shrinkage` ship the closed-form numbers.

## 4. The architecture mirrors the math

The implementation is intentionally structured so that each section of
the manuscript corresponds to a single module:

| Manuscript section | Module | What it computes |
|---|---|---|
| §3 model | `model.py`, `likelihoods.py` | Three-level prior, Bernoulli / Beta-Binomial / Categorical / NB reconstruction terms |
| §4 variational family | `varfamily.py`, `encoders.py`, `flow.py` | Population/sample mean-field Gaussians + Set Transformer + Conditional NSF flow |
| §5 ELBO | `elbo.py`, `distributions.py` | Reconstruction + KL decomposition, IWAE, DReG |
| §6 algorithm | `svi.py`, `torch_svi.py` | The full SVI loop, Robbins-Monro updates, global recentering |
| §7 identifiability | `identifiability.py` | VIB, mQTL anchors, counterfactual, gradient-reversal domain head |
| §8 calibration | `calibration.py` | Split conformal, CQR, Mondrian, conformal risk control |
| §9 tissue-of-origin | `tissue.py` | Variance-weighted Dirichlet head, HDP truncation |
| §10 simulator | `simulator.py` | Chromatin-aware cfDNA simulator |
| §11 synthetic | `pipeline.py` | End-to-end synthetic experiment harness |
| §13 bounds | `bounds.py` | Hierarchical pooling variance, Fano lower bound, VIB upper bound |

If you only remember one thing from this page: HAVI-Methyl is a
hierarchical Bayesian generalisation of FinaleMe with a continuous
logit-$\beta$ latent, an amortised flow posterior, a Beta-Binomial
reconstruction tied to the *correct* trials parameter, and a conformal
calibration wrapper on top.
