# Changelog

Mirror of the top of
[`docs/report/CHANGELOG.md`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/CHANGELOG.md).
The repo-side
[`CHANGELOG.md`](https://github.com/osolari/HAVI-Methyl/blob/main/CHANGELOG.md)
tracks code-level changes separately.

## 2026-05-17 — Phase Five Real-Data Push

Closed four of five open follow-ups from §14 Discussion.

1. **Length-mixture re-fit on real Liu 2024 fragments.** EM-fitted a
   3-mode Gaussian mixture to 5 M cfDNA fragments from 8 Liu 2024
   patients (chr 1 + 19–22). New `LENGTH_MIXTURE_*` constants in
   `src/havi_methyl/constants.py` are $\pi = [0.874, 0.117, 0.009]$,
   $\mu = [161, 313, 455]$ bp, $\sigma = [21, 38, 27]$ bp. The
   previously-cited $0.005$ target for the 320–350 bp peak was
   incorrect; real Liu 2024 cfDNA shows $0.001$ per bp. All five
   App. H validation axes flip to *verified*.

2. **True gradient-reversal adversarial head.** Custom
   `torch.autograd.Function` with identity forward / sign-flipped
   scaled-gradient backward, plus a 2-layer MLP discriminator that
   classifies encoder context into sample id. The discriminator
   parameters train normally via AdamW; the encoder receives the
   negated gradient and is pushed toward a sample-invariant context.
   Replaces the previous context-variance proxy. Exposed via
   `TorchSVIConfig.adversarial_weight`.

3. **Joint tissue-head training inside `fit_svi_torch`.** Optional
   `tissue_reference`/`tissue_target`/`tissue_weight` kwargs. Each
   mini-batch solves the variance-weighted deconvolution on the locus
   subset via differentiable `torch.linalg.lstsq` and adds
   `(pi_pred − pi_true).pow(2).mean()` to the loss. Post-hoc §9 head
   recovered at `tissue_weight = 0`.

4. **Proper Tucker-2019 DReG estimator.** When `iwae_dreg = True` and
   $K > 1$, the log-density $\log q_\phi(\eta_k \mid x)$ is computed
   with `(mu_q, log_sigma)` *detached*, leaving only the pathwise term
   in the encoder gradient. Smoke test at $K=8$ shows ~10 %
   ELBO-trajectory variance reduction.

### Headline real-data result

The BB-trials bug fix (using WGBS read coverage `ds.n_total` rather
than WGS fragment count `ds.n` as the Beta-Binomial trials) lifts
HAVI-Methyl (full torch) on the Liu 2024 paired panel from $r = -0.07$
to **$r = 0.455$ vs FinaleMe-style HMM $r = 0.078$ ($\sim 5.8\times$),
AUC $0.74$ vs $0.56$, ECE $0.32$ vs $0.47$.** Per-stratum: HAVI is the
only method with positive Pearson $r$ in every WGBS-depth stratum
(FinaleMe is anti-correlated at $r = -0.05$ in the multi-read
interior). On the Loyfer/UXM_deconv U25 panel the variance-weighted
Dirichlet head wins LOO RMSE at every one of the 36 cell types.

### Figure overhaul

Nine real-data + synthetic figures shipped:
`finaleme_paired_metrics`, `finaleme_paired_scatter` (5-panel hexbin),
`finaleme_coverage_strat`, `loyfer_loo_rmse`, `loyfer_loo_per_tissue`,
`multiseed_recovery`, `recovery_scatter` (2×4 hexbin),
`calibration` (reliability + miscal inset), `elbo_trajectory` (real
torch training curve). All wired into §11 / §12 / §13 of the
manuscript. See the [Figure gallery](figures.md).

### Manuscript prose

All "planned" hedges replaced with measured numbers across §1, §2, §6,
§7, §8, §9, §11, §12, §13, §14, §16, App. E, App. F, App. H, App. I.
Static cross-ref scan: 0 broken `\ref` / `\cite` / `\includegraphics`.
The only open research direction in §14 is prospective clinical
validation.

---

Earlier development passes (Phase Three technical-development pass and
prior) are recorded in the upstream
[`docs/report/CHANGELOG.md`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/CHANGELOG.md).
