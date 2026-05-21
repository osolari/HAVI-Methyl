---
hide:
  - navigation
  - toc
---

<div class="saim-hero" markdown>
  <img src="assets/saim_logo.png" alt="sAIm Labs" class="saim-hero-logo">
  <h1>HAVI-Methyl</h1>
  <p><strong>Hierarchical Amortized Variational Inference for cell-free DNA
  methylation prediction.</strong> Set Transformer encoder + Gaussian /
  Conditional NSF posterior head + Beta-Binomial reconstruction +
  Robbins-Monro recentering, trained end-to-end on real Liu 2024 paired
  cfDNA WGS/WGBS, with calibrated per-CpG posteriors and a
  variance-weighted Dirichlet tissue-of-origin head.</p>
  <p class="saim-hero-badges">
    <a href="https://github.com/osolari/HAVI-Methyl">GitHub</a>
    <a href="https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/main.pdf">Manuscript PDF</a>
    <a href="quickstart.md">Quickstart</a>
    <a href="results.md">Real-data results</a>
  </p>
</div>

<div class="saim-cite" markdown>
**Citation.** If HAVI-Methyl is useful to your research, please cite the
companion paper:

> Omid Shams Solari (2026). *HAVI-Methyl: Hierarchical Amortized
> Variational Inference for Methylation Prediction from cfDNA
> Fragmentomics.* sAIm Labs.
> [PDF](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/main.pdf){target=_blank} ·
> [GitHub](https://github.com/osolari/HAVI-Methyl){target=_blank}

```bibtex
@article{solari2026havimethyl,
  title   = {HAVI-Methyl: Hierarchical Amortized Variational Inference
             for Methylation Prediction from cfDNA Fragmentomics},
  author  = {Solari, Omid Shams},
  year    = {2026},
  note    = {sAIm Labs technical report},
  url     = {https://github.com/osolari/HAVI-Methyl},
}
```
</div>

## What HAVI-Methyl does

cfDNA whole-genome sequencing has become the dominant non-invasive
substrate for non-invasive prenatal testing, minimal-residual-disease
monitoring, and early cancer detection, but methylation information
remains expensive to access because bisulfite conversion is destructive
and biased. The current state of the art for *fragmentomic* methylation
recovery is FinaleMe, a per-sample two-state hidden Markov model.
HAVI-Methyl is its hierarchical Bayesian successor: a continuous logit-$\beta$
latent with direct Bernoulli, Beta-Binomial, Categorical, or Negative-Binomial
reconstruction terms; a Set Transformer fragment-bag encoder with optional
long-range DNA context; and a one-dimensional normalising-flow variational
posterior on the logit scale. The training objective is the structured ELBO
plus optional VIB, mQTL-anchor, counterfactual, and gradient-reversal
domain-adversarial penalties, with conformal wrapping at deployment.

Three observation regimes are kept distinct throughout: direct
bisulfite/nanopore calls, paired-validation pseudo-counts (Beta-Binomial
with a separate over-dispersion $\kappa$), and WGS-only deployment where
fragment features are the only observation. The real-data Liu 2024 panel
is the paired-validation regime — successes come from WGBS *reads* and
trials come from WGBS *coverage*, not from the WGS fragment count, which
is a separate stream feeding only the encoder context.

## Headline result

On the published Liu 2024 paired cfDNA WGS/WGBS panel ($S=77$ patients,
$L=782$ high-variance CpGs, manifest-paired via Supplementary Table 1),
the full HAVI-Methyl torch loop (Set Transformer + Gaussian posterior +
Beta-Binomial reconstruction + Robbins-Monro recentering) trained
$500$ iterations on a single A10G GPU achieves a $\sim 6.0\times$ lift in
Pearson $r$ over the FinaleMe-style HMM baseline ($0.467$ vs.\ $0.078$),
AUC $0.750$ vs.\ $0.564$, credible-interval ECE $0.311$ vs.\ $0.474$.

| Method | Pearson $r$ | AUC at $\beta=0.5$ | ECE (credible) | ICC(2,1) |
|---|---:|---:|---:|---:|
| FinaleMe-style HMM | 0.078 | 0.564 | 0.474 | 0.052 |
| HAVI-Methyl simplified (full) | 0.081 | 0.564 | 0.416 | 0.053 |
| HAVI-Methyl simplified (no flow) | 0.082 | 0.565 | 0.416 | 0.054 |
| HAVI-Methyl simplified (no hierarchy) | 0.078 | 0.564 | 0.318 | 0.052 |
| **HAVI-Methyl (full torch)** | **0.467** | **0.750** | **0.311** | **0.436** |

All five rows come from
[`docs/report/tables/bench_finaleme_realdata.csv`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/tables/bench_finaleme_realdata.csv).

![Liu 2024 paired metrics](assets/figures/finaleme_paired_metrics.png)

The simplified-numpy rows are deliberately initialised from the FinaleMe
baseline and therefore reproduce its per-locus mean almost exactly; they
bound how much of the gap can be attributed to hierarchical shrinkage
alone. The architectural lift comes from the Set Transformer +
Beta-Binomial head, run with the *correct* trials parameter
(`ds.n_total` = WGBS read coverage, not `ds.n` = WGS fragment count).

Five further results round out the picture: (i) **per-stratum recovery** —
HAVI is the only method with positive Pearson $r$ in every WGBS-depth
stratum (interior $\approx +0.28$ vs. FinaleMe $\approx -0.05$; extreme
$\approx +0.64$ vs. $\approx +0.15$); (ii) **Loyfer LOO tissue-of-origin**
— the variance-weighted Dirichlet head wins LOO RMSE at every one of the
36 cell types in the U25 panel ($0.017$ vs. lstsq $0.028$);
(iii) **multi-seed synthetic recovery** — $N=20$ seeds, non-overlapping
$90\%$ bootstrap intervals at $0.1\times$, $1\times$, $5\times$;
(iv) **App. H simulator validation** — all five axes verified after a
3-mode length-mixture re-fit to real Liu 2024 fragments;
(v) **prior-leakage stress** — prior attribution drops from $5.93\%$ to
$0.62\%$ via VIB and to $0.037\%$ with mQTL anchors. See
[Real-data results](results.md) for the full breakdown.

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg } **[Quickstart](quickstart.md)**

    Install, simulate, fit, predict — 30 seconds to a working
    posterior.

-   :material-school:{ .lg } **[Concepts](concepts.md)**

    Three observation regimes, the three-level hierarchy on
    logit-$\beta$, and why fragmentomics is informative at all.

-   :material-chip:{ .lg } **[Architecture](architecture.md)**

    Set Transformer encoder, Gaussian / Conditional NSF posterior head,
    Beta-Binomial reconstruction, IWAE-DReG, gradient-reversal head.

-   :material-database:{ .lg } **[Datasets](datasets.md)**

    The Liu 2024 paired panel loader (`load_finaleme_dataset`), the
    Loyfer U25 atlas loader, and the high-variance CpG panel.

-   :material-chart-bell-curve:{ .lg } **[Calibration](calibration.md)**

    Split-conformal wrapper, the A5 ablation (0.879 empirical at 0.90
    nominal), worst-stratum diagnostics.

-   :material-microscope:{ .lg } **[Tissue-of-origin](tissue.md)**

    Variance-weighted Dirichlet head, joint training inside
    `fit_svi_torch`, the 36-of-36 Loyfer LOO win.

-   :material-chart-line:{ .lg } **[Real-data results](results.md)**

    Liu 2024 head-to-head, per-stratum recovery, Loyfer LOO, the
    A0..A5 ablation matrix.

-   :material-flask:{ .lg } **[Synthetic recovery](results-synth.md)**

    $N=20$-seed bootstrap intervals at $0.1\times$, $1\times$,
    $5\times$, $30\times$ coverage.

-   :material-book:{ .lg } **[API reference](api.md)**

    The public symbols of `havi_methyl` with manuscript
    cross-references.

-   :material-file-pdf-box:{ .lg } **[Manuscript](report.md)**

    The technical report this implementation matches one-to-one.

</div>
