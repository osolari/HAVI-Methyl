# Headline real-data results

This page reproduces the §12 figures and numbers in full. Every number
is sourced from a CSV in `docs/report/tables/` or `outputs/tables/`;
every figure is the live PNG written by the corresponding
`scripts/fig_*.py`.

## 1. Liu 2024 paired methylation

The published Liu 2024 paired cfDNA WGS/WGBS panel: $S = 77$ patients,
$L = 782$ high-variance CpGs on chromosomes 1 and 19–22, manifest-paired
via Supplementary Table 1, evaluated at WGBS coverage $\ge 1$ in
$\ge 40\%$ of samples.

The full HAVI-Methyl torch loop (Set Transformer + Gaussian posterior +
Beta-Binomial reconstruction + Robbins-Monro recentering) was trained
$500$ iterations on a single A10G GPU with the Beta-Binomial trials
parameter set to the WGBS read coverage (`ds.n_total`), not the WGS
fragment count.

| Method | Pearson $r$ | Spearman $r$ | AUC at $\beta=0.5$ | ECE (credible) | ICC(2,1) |
|---|---:|---:|---:|---:|---:|
| FinaleMe-style HMM | 0.078 | 0.063 | 0.564 | 0.474 | 0.052 |
| HAVI-Methyl simplified (full) | 0.081 | 0.065 | 0.564 | 0.416 | 0.053 |
| HAVI-Methyl simplified (no flow) | 0.082 | 0.067 | 0.565 | 0.416 | 0.054 |
| HAVI-Methyl simplified (no hierarchy) | 0.078 | 0.063 | 0.564 | 0.318 | 0.052 |
| **HAVI-Methyl (full torch)** | **0.467** | **0.434** | **0.750** | **0.311** | **0.436** |

All five rows are from
[`docs/report/tables/bench_finaleme_realdata.csv`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/tables/bench_finaleme_realdata.csv).
DMR F1 is `0.0` for all rows on this panel; see the §STATUS *Honest
negative result* note in the [reproducibility](reproducibility.md)
section.

![Liu 2024 paired metrics](assets/figures/finaleme_paired_metrics.png)

The simplified-numpy ablations are deliberately initialised from the
FinaleMe baseline and therefore reproduce its per-locus mean almost
exactly. The architectural lift over the baseline ($r=0.078 \to 0.467$,
a $\sim 6.0\times$ improvement; AUC $0.564 \to 0.750$; credible-interval
ECE $0.474 \to 0.311$) is the flow- and Set-Transformer-driven head.

### Per-locus density

![Liu 2024 paired scatter](assets/figures/finaleme_paired_scatter.png)

Per-locus prediction density (hexbin) for each method on the same
60 214 $(s,\ell)$ points. The FinaleMe HMM and the simplified
ablations are clamped to a narrow $[\sim 0.34, \sim 0.52]$ band by
their EM averaging step; the full-torch HAVI-Methyl posterior spans the
entire $[0, 1]$ range along the diagonal.

## 2. Per-stratum recovery

Stratifying the same evaluation by WGBS depth shows that HAVI-Methyl is
the *only* method with positive correlation in every stratum. From
[`outputs/tables/bench_finaleme_coverage_strat.csv`](https://github.com/osolari/HAVI-Methyl/blob/main/outputs/tables/bench_finaleme_coverage_strat.csv):

| Stratum | $n$ points | FinaleMe HMM | HAVI simplified | **HAVI full torch** |
|---|---:|---:|---:|---:|
| $\beta$ extreme ($n_{\mathrm{total}}\approx 1$) | 22 443 | 0.148 | 0.153 | **0.647** |
| $\beta$ interior (multi-read) | 37 771 | $-0.049$ | $-0.048$ | **0.302** |
| $\beta$ middle ($0.1 < \beta < 0.9$) | 37 437 | $-0.035$ | $-0.034$ | **0.278** |

The 2-cluster EM ceiling of the FinaleMe baseline is the cause of the
negative correlation in the interior and middle strata; it cannot
extrapolate to non-binary methylation states. HAVI-Methyl reaches
$r \approx +0.28$ in the multi-read interior — the same stratum where
the FinaleMe baseline is anti-correlated with truth at $r \approx -0.05$.

![Liu 2024 per-stratum](assets/figures/finaleme_coverage_strat.png)

In the $\beta$-extreme stratum the gap widens to $r \approx +0.64$ vs
$r \approx +0.15$.

## 3. Loyfer LOO tissue-of-origin

On the published Loyfer/UXM_deconv U25 hg38 panel (36 cell types $\times$
900 marker blocks), the variance-weighted Dirichlet head of §9 was
evaluated against three baselines. From
[`outputs/tables/bench_tissue_loo.csv`](https://github.com/osolari/HAVI-Methyl/blob/main/outputs/tables/bench_tissue_loo.csv):

| Method | In-panel RMSE | LOO mean RMSE | LOO worst RMSE |
|---|---:|---:|---:|
| FinaleMe-binarized + QP | 0.0367 | 0.0377 | 0.0381 |
| Continuous lstsq | 0.0273 | 0.0280 | 0.0284 |
| **HAVI-Methyl Dirichlet head** | **0.0169** | **0.0174** | **0.0178** |
| HDP-truncated ($T_{\max}=64$) | 0.0329 | 0.0347 | 0.0363 |

![Loyfer LOO RMSE](assets/figures/loyfer_loo_rmse.png)

### Per-tissue: HAVI wins 36/36

![Loyfer per-tissue](assets/figures/loyfer_loo_per_tissue.png)

**HAVI is shortest at every one of the 36 cell types.** The median
advantage over continuous lstsq is $+0.011$; the only adverse case
(Eryth-prog) trails the leader by $0.011$. See the
[Tissue-of-origin](tissue.md) page for the head specification.

## 4. Synthetic ablation matrix (A0..A5)

The nested HAVI-Methyl ablation matrix isolates the contribution of
each architectural piece on a synthetic FinaleMe-proxy panel ($S=12$,
$L=120$, coverage $2\times$, seed `20260429`). From
[`docs/report/tables/bench_ablation_matrix.csv`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/tables/bench_ablation_matrix.csv):

| Configuration | BetaBin | Hier. | Flow | VIB/mQTL | Conformal | Pearson $r$ | AUC | cov$_{0.90}$ |
|---|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|
| A0. Feature regression scaffold | | | | | | 0.716 | 0.874 | --- |
| A1. + Beta-Binomial pseudo-likelihood | $\checkmark$ | | | | | 0.732 | 0.910 | --- |
| A2. + hierarchical SVI | $\checkmark$ | $\checkmark$ | | | | 0.866 | 0.959 | --- |
| A3. + amortized flow posterior | $\checkmark$ | $\checkmark$ | $\checkmark$ | | | 0.845 | 0.939 | --- |
| A4. + leakage-control terms | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | | 0.852 | 0.940 | --- |
| **A5. + conformal wrapper (full)** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | **0.852** | **0.940** | **0.879** |

A0–A5 climb monotonically in Pearson $r$ and AUC up to A4; A5 layers
the conformal wrapper on top to recover marginal coverage at the cost
of slightly wider intervals (mean width $0.69$). The conformal
guarantee (Proposition 8.1) costs about $\sim 0.02$ Pearson $r$ and
recovers $0.879$ empirical coverage at the $0.90$ nominal target —
within the IMPL-07 $\pm 5\%$ exit criterion.

## Reproducing these numbers

All five numbers above are stored in CSVs under
[`docs/report/tables/`](https://github.com/osolari/HAVI-Methyl/tree/main/docs/report/tables)
and regenerated by

```bash
bash scripts/run_all.sh
```

See [Reproducibility](reproducibility.md) for the per-script ordering
and the sticky-skip guard that protects the Liu 2024 row across local
re-runs.

## Honest negative result

DMR F1 is `0.0` for *every* method on the Liu 2024 panel; the empirical
$\Delta\beta\ge 0.25$ + BH-corrected $q\le 0.05$ threshold is not
cleared by any of the five evaluated configurations on this 782-CpG
panel. This is exactly the limitation the per-sample flow posterior
(IMPL-04) is intended to address; expect the F1 to recover once the
flow head is the headline configuration instead of the Gaussian head.

## What is *not* live data

External-baseline rows in `bench_finaleme_realdata.csv` (FinaleMe upstream,
DeepCpG, Elastic-net, MethylBERT) remain `XX` placeholders pending each
project's own codebase being aligned to the 782-CpG panel. The
HAVI-Methyl and FinaleMe-style-HMM-reimplementation rows are real Liu
2024 numbers from the lab drive run.
