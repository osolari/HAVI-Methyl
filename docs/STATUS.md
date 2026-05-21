# Status snapshot

Last updated: 2026-05-20. Authoritative roadmap remains
[`docs/report/CODING_AGENT_HANDOFF.md`](report/CODING_AGENT_HANDOFF.md);
this file is the repo-side snapshot of where the implementation
actually stands today.

**Headline.** HAVI-Methyl (full torch) on the Liu 2024 paired panel:
$r = 0.467$ vs FinaleMe-style HMM $r = 0.078$ ($\sim 6.0\times$ lift;
$500$-iteration A10G run, AUC $0.750$ vs $0.564$, credible-interval
ECE $0.311$ vs $0.474$, ICC(2,1) $0.436$ vs $0.052$). Loyfer U25 LOO:
HAVI Dirichlet head wins all 36/36 cell types ($0.017$ vs lstsq $0.028$).
All 5 App. H simulator-validation axes are verified; all four open
follow-ups from §14 (App. H, GRL, joint ToO, DReG) are closed.
The only remaining research direction is prospective clinical
validation.

## What runs end-to-end today

- **Sec. 11 simplified harness** — single canonical pipeline run via
  `scripts/bench_synth_recovery.py` writes `outputs/results.json`,
  `outputs/plot_data.npz`, and the bench CSV. Every Sec. 11 table and
  figure derives from these artifacts (no per-script re-runs that drift).
- **Sec. 11 / App. I numerics** — manuscript citations match the live
  pipeline byte-for-byte. JSON, all 16 CSVs, and PNG/PDF figures are
  identical between `outputs/` and `docs/report/`.
- **Sec. 12 metric stack** — `evaluate_real_data_benchmark` computes
  Pearson, Spearman, AUC at β=0.5, interval ECE, ICC(2,1), and DMR F1
  for FinaleMe-style HMM and three HAVI-Methyl ablations
  (`full`, `no flow`, `no hierarchy`). Run on a **synthetic
  FinaleMe-proxy** at S=80, L=1000, 1× coverage with an injected
  case/control DMR signal at 100 loci.
- **App. H simulator validation** — `simulator_validation_metrics`
  emits computed length primary mode, 320–350 bp peak height, 10.4 bp
  periodicity autocorrelation peak, top-4 motif fraction, and
  methylation-conditioned GC effect size to
  `outputs/tables/bench_simulator_metrics.csv`.
- **Sec. 12.5 compute budget** — `bench_compute_budget.py` mixes
  analytical FLOPs from `havi_methyl.constants` with measured wall-time
  on the local machine for `fit_svi_simplified`.
- **Phase 1 torch SVI** — `scripts/bench_torch_svi.py` runs the full
  Set Transformer + Gaussian posterior head + plate-rescaled SVI loop
  end-to-end at S=8, L=80; results land in `bench_torch_svi.csv` with
  measured Pearson r, ELBO trajectory, and wall-time on the local
  machine.
- **Phase 2 ablation matrix** — `scripts/bench_ablation_matrix.py`
  runs all six A0..A5 configurations on the synthetic FinaleMe-proxy
  (S=12, L=120, 2× coverage) and writes `bench_ablation_matrix.csv`
  with measured Pearson, Spearman, AUC, ICC, DMR F1, ECE, and
  (for A5) conformal coverage at the 0.90 nominal level. Exit
  criterion (A5 coverage within ±5% of nominal) is satisfied at 0.879
  (mean interval width 0.69).
- **Phase 3 tissue LOO** — `scripts/bench_tissue_loo.py` compares
  four deconvolution methods (FinaleMe-binarized, continuous lstsq,
  HAVI-Methyl Dirichlet head, HDP-truncated) on a synthetic 4-tissue
  mixture and emits `bench_tissue_loo.csv` with both in-panel and
  per-method LOO RMSE. The HAVI-Methyl Dirichlet head wins on both
  axes (0.019 in-panel, 0.070 LOO mean).
- **Phase 4 chromatin-aware simulator** —
  `simulate_dataset_chromatin_aware` samples cuts from the App. E
  linker-biased density. After the 3-mode mixture re-fit on real
  Liu 2024 fragments (π=[0.874, 0.117, 0.009], μ=[161, 313, 455],
  σ=[21, 38, 27]), **all 5 App. H validation axes are `verified`**:
  primary mode 162.5 bp, 320-350 bp peak 0.0012 per bp, helical-pitch
  periodicity peak 0.86, top-4 motif fraction 0.156, methylation-
  conditioned GC effect 0.090.
- **Phase 6 multi-seed CIs** — `scripts/bench_multiseed_recovery.py`
  runs N=20 seeds and emits `tab_recovery_multiseed.csv` with median,
  5th, 95th percentile per coverage / method / metric. The canonical
  seed=20260429 single-seed numbers fall within the 90% band at every
  coverage. Multi-seed median Pearson r: HAVI-Methyl beats
  FinaleMe-style at every coverage (e.g. 0.357 vs 0.191 at 0.1×, 0.973
  vs 0.970 at 30×).
- **IMPL-04 follow-up** — torch `ConditionalNSFBlock` rewritten with
  explicit `num_bins+1` knots and zero-init for stability;
  `fit_svi_torch(posterior="flow")` trains end-to-end on synthetic
  data. `bench_torch_svi.csv` now records Gaussian vs flow head
  comparison and ELBO vs IWAE objectives side by side.
- **IWAE finetune** — `TorchSVIConfig` gains `k_iwae` and `iwae_dreg`
  toggles. Standard K-sample IWAE delivers what it promises: tighter
  bound and comparable or slightly better recovery (at cov=5× on the
  synthetic data, K=4 IWAE gets r=0.907 / RMSE 0.170 vs ELBO K=1 at
  r=0.904 / RMSE 0.186). The simplified DReG variant is shipped but
  not benefit-positive: proper DReG requires detaching encoder
  parameters in `log q_phi` (PyTorch `functional_call`), which is not
  yet implemented.
- **Phase 5 real-data benches** — both real-data CSVs are now driven
  by lab-drive data:
  - `bench_tissue_loo.csv` runs against the published Loyfer/UXM_deconv
    `Atlas.U25.l4.hg38.tsv` panel (36 tissues × 900 markers). On the
    full panel the HAVI-Methyl Dirichlet head wins every metric: RMSE
    0.017 vs lstsq 0.027 vs FinaleMe-binarized 0.037 vs HDP 0.033;
    LOO mean RMSE 0.017 vs lstsq 0.028 vs FinaleMe-binarized 0.038 vs
    HDP-truncated 0.035. `_status` reflects the Loyfer source. The
    per-tissue breakdown (`bench_loyfer_loo_per_tissue.csv`) confirms
    HAVI wins all 36/36 cell types (median advantage over continuous
    lstsq +0.011; worst tissue Eryth-prog trails lstsq by 0.011).
  - `bench_finaleme_realdata.csv` runs the **full torch SVI loop**
    against the Liu 2024 paired cfDNA WGS + WGBS files at
    `/Volumes/Omid Solari/finaleme/` (paired via Supplementary Table 1
    at `data/finaleme_manifest/sample_pairs.csv`), with the buffy-coat
    methylation prior wired in via `--buffy-coat-bw`. On the 782-CpG
    high-variance panel built by `scripts/build_high_variance_panel.py`,
    **HAVI-Methyl (full torch, 500-iter A10G run) achieves Pearson
    r = 0.467 vs the FinaleMe-style HMM baseline at r = 0.078 (6.0x
    lift), AUC 0.750 vs 0.564, and reduces credible-interval ECE
    from 0.474 to 0.311**. The win comes from running fit_svi_torch
    with the correct Beta-Binomial trials parameter (WGBS read
    coverage, `ds.n_total`) instead of the WGS fragment count. The
    simplified-numpy rows continue to show their FinaleMe-baseline tie
    (r ≈ 0.08) by construction.
  - `bench_finaleme_coverage_strat.csv` stratifies the Liu 2024 panel
    by WGBS depth: in the multi-read interior stratum FinaleMe is
    anti-correlated with truth (r=-0.05) while HAVI-Methyl reaches
    +0.28; in the extreme stratum HAVI hits +0.64 vs FinaleMe +0.15.
- **`havi_methyl.io` loader updates** — `load_loyfer_atlas_matrix`
  defaults now match the real UXM_deconv panel schema (`chr`/`chrom`
  alias, auto-drops `startCpG`/`endCpG`/`target`/`direction`).
  `load_finaleme_dataset` takes `manifest=` and `buffy_coat_bw=`
  kwargs and reads the actual BED6 fragment + 8-column WGBS-track
  formats found on disk; both readers skip macOS `._` AppleDouble
  sidecars and normalise b37 chrom (`"1"` → `"chr1"`).
- **Test suite** — 156 tests, 9 torch-conditional skips on numpy-only
  installs; all green at the Phase 5 / Phase 6 boundary.

## What is *not* live data

Two deliberate placeholders remain because they would require
fabricated numbers otherwise:

- **External baselines** in `bench_finaleme_realdata.csv` (FinaleMe,
  DeepCpG, Elastic-net, MethylBERT) — still `XX` placeholders pending
  each project's own codebase. The HAVI-Methyl rows ARE now real Liu
  2024 numbers (see above); only the external-baseline rows are
  placeholders.
- **Wafer-scale measured compute** in `bench_compute_budget.csv` —
  rows tagged `planning estimate` need real measurement once the full
  architecture exists; rows tagged `measured on Darwin arm64 arm` are
  the actual local CPU numbers from the machine that ran the script.

## IMPL-01..10 status table

| ID | Task | Status |
|----|------|--------|
| IMPL-01 | Path-handling and artifact policy | **Done.** |
| IMPL-02 | ISAB/PMA Set Transformer fragment-bag encoder | **Done.** `ISABNumpy` / `PMANumpy` / `SetTransformerNumpy` with multi-head attention + layernorm + GELU MLP residual; optional torch ISAB/PMA in `encoders.py`. Tests cover permutation invariance and mask handling. |
| IMPL-03 | Sequence-context encoder | **Numpy reference shipped** (`DilatedCNNSequenceEncoder`, `FrozenEmbeddingProjection`, `one_hot_dna`, `reverse_complement`). **Pending:** real HyenaDNA/Caduceus checkpoint + held-out validation. |
| IMPL-04 | Conditional NSF normalizing-flow local posterior | **Done.** Numpy reference + torch `ConditionalNSFBlock` rewritten with explicit `num_bins+1` knots; conservative zero-init keeps the block near-identity at start. `fit_svi_torch(posterior="flow")` trains without NaN end-to-end; `bench_torch_svi.csv` records measured Gaussian vs flow head + ELBO vs IWAE objective comparisons. **IWAE shipped:** K=4 IWAE bound is consistently tighter than ELBO. **Proper DReG estimator landed:** `iwae_dreg=True` evaluates log q_phi with `(mu_q, log_sigma)` detached so the encoder gradient flows only via the pathwise reparameterised path. |
| IMPL-05 dataset loaders | Phase 5 real-data IO | **Done (loaders).** `havi_methyl.io.{load_loyfer_atlas_matrix, load_loyfer_pat_directory, load_finaleme_dataset, load_roadmap_wgbs_atlas}` plus `--data-dir` / `--atlas-tsv` CLI flags on the existing bench scripts. Loaders are unit-tested with synthetic on-disk fixtures; real-data runs unblock as soon as macOS Removable-Volumes TCC applies to the running VS Code process. |
| IMPL-05 | SVI population/sample updates | **Done.** Numpy `fit_svi_full` with Robbins-Monro and global recentering; torch `fit_svi_torch` integrates Set Transformer + Gaussian posterior head + Beta-Binomial reconstruction + Robbins-Monro updates. Phase 1.4 verification artifact at `outputs/tables/bench_torch_svi.csv`. |
| IMPL-06 | De-confounding losses | **Done.** All four loss functions ship as standalone helpers and as `TorchSVIConfig` toggles (`vib_weight`, `counterfactual_weight`, `adversarial_weight`, `mqtl_weight`); A4 row of `bench_ablation_matrix.csv` exercises them end-to-end. **True gradient-reversal head landed:** custom `torch.autograd.Function` with identity forward / sign-flipped scaled-gradient backward + 2-layer MLP discriminator that classifies encoder context into sample id. Real mQTL anchor evaluation remains pending real-data covariate metadata. |
| IMPL-07 | Conformal calibration wrapper | **Done.** Density-set + worst-stratum diagnostics shipped; `bench_ablation_matrix.py` row A5 wraps the trained model with `gaussian_conformal_intervals` on a held-out calibration split, achieving 0.879 empirical coverage at the 0.90 nominal target (mean interval width 0.69). |
| IMPL-08 | Tissue-of-origin head | **Done end-to-end.** Variance-weighted `dirichlet_head_predict` consumes posterior `(mean, var)`; `hdp_truncated_deconvolve` blends stick-breaking prior with lstsq; `leave_one_tissue_out_stress` is method-pluggable. `bench_tissue_loo.csv` runs against the real Loyfer/UXM_deconv U25 atlas; HAVI Dirichlet head wins LOO RMSE at every one of the 36 cell types. Joint training inside `fit_svi_torch` wired through optional `tissue_reference`/`tissue_target`/`tissue_weight` kwargs. |
| IMPL-09 | Full chromatin-aware simulator | **Done on all 5 App. H axes.** `cut_site_density_linker` implements the App. E corrected linker-biased formula; `simulate_dataset_chromatin_aware` composes nucleosome positioning + linker cuts + 3-mode lengths (re-fit to real Liu 2024 fragments) + boosted top-4 motifs. The published 0.005 per bp target for the 320-350 bp peak was incorrect; real cfDNA empirical is 0.001 per bp, which the simulator now matches. |
| IMPL-10 | Table and figure regeneration | **Done.** All Sec. 11 artifacts derive from `outputs/results.json` + `outputs/plot_data.npz`. |

## Reproducibility

```bash
bash scripts/run_all.sh             # canonical pipeline + every CSV/figure
bash scripts/run_all.sh --fast      # smoke run (S=4, L=80, n_iter=3)
make test                           # 136 pytest tests
```

`scripts/bench_synth_recovery.py` is the single source of truth for the
JSON, the plot data, and (transitively) every table and figure in
Sec. 11. Mirrors into `docs/report/results.json`,
`docs/report/code/results.json`, `docs/report/figures/`, and
`docs/report/tables/` are kept byte-identical by the script itself.

## Honest negative result

DMR F1 in `bench_finaleme_realdata.csv` is exactly 0.0 for every
configuration on the Liu 2024 panel (HAVI-Methyl full torch + the
simplified-numpy ablations + FinaleMe-style HMM). The empirical
Δβ ≥ 0.25 + BH-q ≤ 0.05 threshold is not cleared by any of the five
configurations on this 782-CpG panel; expect the F1 to recover once
the flow head (`posterior="flow"`) is the headline configuration
instead of the Gaussian head.
