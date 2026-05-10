# Status snapshot

Last updated: 2026-05-09. Authoritative roadmap remains
[`docs/report/CODING_AGENT_HANDOFF.md`](report/CODING_AGENT_HANDOFF.md);
this file is the repo-side snapshot of where the implementation
actually stands today.

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
  criterion (A5 coverage within ±5% of nominal) is satisfied at 0.867.
- **Phase 3 tissue LOO** — `scripts/bench_tissue_loo.py` compares
  four deconvolution methods (FinaleMe-binarized, continuous lstsq,
  HAVI-Methyl Dirichlet head, HDP-truncated) on a synthetic 4-tissue
  mixture and emits `bench_tissue_loo.csv` with both in-panel and
  per-method LOO RMSE. The HAVI-Methyl Dirichlet head wins on both
  axes (0.019 in-panel, 0.070 LOO mean).
- **Phase 4 chromatin-aware simulator** —
  `simulate_dataset_chromatin_aware` samples cuts from the App. E
  linker-biased density; `simulator_validation_metrics(chromatin_aware
  =True)` flips 4 of 5 App. H axes to `verified` (primary mode 167.5
  bp, helical-pitch periodicity peak 0.86, top-4 motif fraction
  0.156, methylation-conditioned GC effect 0.090).
- **Phase 6 multi-seed CIs** — `scripts/bench_multiseed_recovery.py`
  runs N=20 seeds and emits `tab_recovery_multiseed.csv` with median,
  5th, 95th percentile per coverage / method / metric. The canonical
  seed=20260429 single-seed numbers fall within the 90% band at every
  coverage. Multi-seed median Pearson r: HAVI-Methyl beats
  FinaleMe-style at every coverage (e.g. 0.357 vs 0.191 at 0.1×, 0.973
  vs 0.970 at 30×).
- **Test suite** — 146 tests, all passing.

## What is *not* live data

Three deliberate placeholders remain because they would require
fabricated numbers otherwise:

- **EGA Liu 2024 numbers** in `bench_finaleme_realdata.csv` — every
  HAVI-Methyl row is the synthetic FinaleMe-proxy output, never the
  real EGA dataset. The `_status` column on every row makes this
  explicit. External baselines (FinaleMe, DeepCpG, Elastic-net,
  MethylBERT) are `XX` placeholders pending their own codebases.
- **Wafer-scale measured compute** in `bench_compute_budget.csv` —
  rows tagged `planning estimate` need real measurement once the full
  architecture exists; rows tagged `measured on Darwin arm64 arm` are
  the actual local CPU numbers from the machine that ran the script.
- **Multi-seed sensitivity sweeps** — Sec. 11 reports a single seed
  (`20260429`); multi-seed bootstrap intervals are scheduled for
  IMPL-05 follow-up.

## IMPL-01..10 status table

| ID | Task | Status |
|----|------|--------|
| IMPL-01 | Path-handling and artifact policy | **Done.** |
| IMPL-02 | ISAB/PMA Set Transformer fragment-bag encoder | **Done.** `ISABNumpy` / `PMANumpy` / `SetTransformerNumpy` with multi-head attention + layernorm + GELU MLP residual; optional torch ISAB/PMA in `encoders.py`. Tests cover permutation invariance and mask handling. |
| IMPL-03 | Sequence-context encoder | **Numpy reference shipped** (`DilatedCNNSequenceEncoder`, `FrozenEmbeddingProjection`, `one_hot_dna`, `reverse_complement`). **Pending:** real HyenaDNA/Caduceus checkpoint + held-out validation. |
| IMPL-04 | Conditional NSF normalizing-flow local posterior | **Numpy reference shipped** (`ConditionalRationalQuadraticSpline` with bisection inverse). Torch `ConditionalNSFStack` shipped with `forward`/`inverse`/`log_density`. **Pending:** stable rational-quadratic spline parameterisation in torch (current block produces NaN at random init due to indexing edge cases). DReG-IWAE objective. |
| IMPL-05 | SVI population/sample updates | **Done.** Numpy `fit_svi_full` with Robbins-Monro and global recentering; torch `fit_svi_torch` integrates Set Transformer + Gaussian posterior head + Beta-Binomial reconstruction + Robbins-Monro updates. Phase 1.4 verification artifact at `outputs/tables/bench_torch_svi.csv`. |
| IMPL-06 | De-confounding losses | **Done.** All four loss functions ship as standalone helpers and as `TorchSVIConfig` toggles (`vib_weight`, `counterfactual_weight`, `adversarial_weight`, `mqtl_weight`); A4 row of `bench_ablation_matrix.csv` exercises them end-to-end. **Pending:** true gradient-reversal head (current adversarial proxy is a context-variance penalty), real mQTL anchors. |
| IMPL-07 | Conformal calibration wrapper | **Done.** Density-set + worst-stratum diagnostics shipped; `bench_ablation_matrix.py` row A5 wraps the trained model with `gaussian_conformal_intervals` on a held-out calibration split, achieving 0.867 empirical coverage at the 0.90 nominal target. |
| IMPL-08 | Tissue-of-origin head | **Done on the synthetic proxy.** Variance-weighted `dirichlet_head_predict` consumes posterior `(mean, var)`; `hdp_truncated_deconvolve` blends stick-breaking prior with lstsq; `leave_one_tissue_out_stress` is now method-pluggable. `bench_tissue_loo.csv` records per-method in-panel + LOO RMSE on a synthetic 4-tissue mixture. **Pending:** atlas swap (Loyfer 2023) when accession is verified; joining the head's training loss with `fit_svi_torch`. |
| IMPL-09 | Full chromatin-aware simulator | **Done on 4 of 5 App. H axes.** `cut_site_density_linker` implements the App. E corrected linker-biased formula; `simulate_dataset_chromatin_aware` composes nucleosome positioning + linker cuts + 3-mode lengths + boosted top-4 motifs. Strauss repulsion already in `sample_nucleosomes`. Validation table flips axes to `verified` when targets are hit. **Pending:** length-mixture re-fitting on a real dataset to match the 320-350 bp secondary peak height target ~0.005. |
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

DMR F1 in `bench_finaleme_realdata.csv` is exactly 0.0 for the
HAVI-Methyl-simplified hierarchy variants on the synthetic proxy. The
empirical-Bayes shrinkage averages too much across loci to clear the
(Δβ ≥ 0.25, BH-q ≤ 0.05) DMR threshold even with a +0.4 case/control
shift. This is exactly the limitation the full per-sample flow
posterior (IMPL-04) is intended to address; expect the F1 to recover
once IMPL-02..05 are wired end-to-end.
