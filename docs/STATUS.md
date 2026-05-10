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
- **Test suite** — 137 tests, all passing.

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
| IMPL-06 | De-confounding losses | **All four loss functions shipped** (VIB, counterfactual, mQTL, domain-adversarial) plus cohort-balance diagnostic. **Pending:** integration into a joint training loop with gradient reversal. |
| IMPL-07 | Conformal calibration wrapper | **Density-set + worst-stratum diagnostics shipped** on top of split-conformal / CQR / Mondrian / risk-control. **Pending:** wiring through the full flow posterior. |
| IMPL-08 | Tissue-of-origin head | **LOO stress test shipped** on top of the existing Dirichlet head + HDP truncation. **Pending:** full Dirichlet-head training joined with the SVI loop. |
| IMPL-09 | Full chromatin-aware simulator | **Validation runner shipped**; compact simulator emits the App. H axes. **Pending:** chromatin tracks, methylation-conditioned cut-bias model, full Strauss repulsion. |
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
