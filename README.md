# HAVI-Methyl

**Hierarchical Amortized Variational Inference for Methylation Prediction from cfDNA Fragmentomics.**

HAVI-Methyl proposes a hierarchical Bayesian successor to FinaleMe's
per-sample binary HMM that quantifies uncertainty, pools statistical
strength across samples, de-confounds the prior, and trains end-to-end with
the tissue-of-origin head — see `docs/report/main.pdf` for the full
manuscript and `docs/report/CODING_AGENT_HANDOFF.md` for the implementation
roadmap.

This repository ships **both**: the released simplified-numpy harness
used for the synthetic recovery benchmark (Sec. 11), and the **full
torch SVI training stack** (Set Transformer encoder + Gaussian or
Conditional NSF flow posterior head + Beta-Binomial reconstruction +
Robbins-Monro recentering) that lands the headline real-data result.
On the published Liu 2024 paired cfDNA WGS/WGBS panel ($S=77$ patients,
$L=782$ high-variance CpGs), HAVI-Methyl (full torch, $500$ iterations
on a single A10G GPU) achieves Pearson $r = 0.467$ versus a
FinaleMe-style HMM baseline at $r = 0.078$ (a $\sim 6.0\times$ lift;
AUC $0.750$ vs $0.564$; credible-interval ECE $0.311$ vs $0.474$; cf.
`docs/report/figures/finaleme_paired_metrics.png`). On the
Loyfer/UXM_deconv U25 panel the variance-weighted Dirichlet head wins
LOO RMSE at every one of the 36 cell types (overall $0.017$ vs lstsq
$0.028$). All six implementation phases of
[`IMPLEMENTATION_ROADMAP.md`](docs/IMPLEMENTATION_ROADMAP.md) are
complete; the only remaining research direction is prospective
clinical validation.

Repository layout:

- `src/havi_methyl/` — Python package: closed-form distributions, ELBO,
  simplified SVI, baseline classifiers, calibration / identifiability /
  tissue helpers, and theoretical-bound utilities.
- `tests/` — pytest suite verifying the theoretical claims empirically.
- `scripts/` — figure / table / benchmark scripts. `bench_synth_recovery.py`
  is the canonical pipeline run; every figure and table downstream reads
  from `outputs/results.json` and `outputs/plot_data.npz` (IMPL-10 in
  `docs/report/CODING_AGENT_HANDOFF.md`).
- `notebooks/` — tutorial notebooks for simulator / inference / calibration.
- `docs/report/` — the manuscript, its TeX sources, and `code/run_experiments.py`,
  the standalone deliverable harness.

## Installation

```bash
make install-dev          # numpy/scipy/matplotlib/pandas + pytest/ruff/mypy
make install-torch        # adds torch for the production Set-Transformer + flow
```

The core library only requires `numpy` and `scipy`. The full HAVI-Methyl
encoder (Set Transformer + neural-spline flow) requires `torch`; without
it, the same math is exercised through the simplified-Gaussian variant of
Sec. 11.

## Quickstart

### Simplified-numpy path (CPU, runs in seconds)

```python
import numpy as np
import havi_methyl as hm

# 1. Simulate a small dataset (Sec. 11)
sim = hm.simulate_dataset(S=12, L=300, coverage=5.0, rng=20260429)

# 2. FinaleMe-style baseline
pred_baseline, fit = hm.finaleme_baseline_predict(sim.bags, sim.n)

# 3. Simplified HAVI-Methyl SVI (Sec. 6, Algorithm 1)
state = hm.fit_svi_simplified(pred_baseline, sim.n, n_iter=10)
pred_havi, std_havi = hm.predict_with_state(state, pred_baseline, sim.n)

# 4. Compare
print("FinaleMe Pearson r:", round(hm.pearson_r(sim.beta_sample, pred_baseline), 3))
print("HAVI-Methyl Pearson r:", round(hm.pearson_r(sim.beta_sample, pred_havi), 3))
```

### Full torch SVI path (GPU; lands the headline real-data row)

```python
import havi_methyl as hm
from havi_methyl import TorchSVIConfig, fit_svi_torch, predict_with_torch_state

sim = hm.simulate_dataset_chromatin_aware(S=16, L=400, coverage=2.0, rng=20260429)

cfg = TorchSVIConfig(
    in_dim=5, hidden=32, num_inducing=16, num_layers=2,
    kappa=20.0, posterior="gaussian",
    k_iwae=4, iwae_dreg=True,           # Tucker-2019 DReG-IWAE
)

state = fit_svi_torch(
    bags=sim.bags,
    n_frag=sim.n,                       # WGS fragment counts (encoder feature)
    n_meth=sim.n_meth,                  # Beta-Binomial successes
    n_obs=sim.n_total,                  # Beta-Binomial trials (= WGBS coverage on real data)
    n_iter=500, config=cfg, seed=20260429,
)
pred, var = predict_with_torch_state(state, sim.bags, sim.n)
```

For *real* Liu 2024 paired data the same code path consumes
`load_finaleme_dataset(...)` directly and the BB-trials slot must be
`ds.n_total` (WGBS coverage), not `ds.n` (WGS fragment count) — that
is the bug fix that lifts the full torch loop from $r = -0.07$ to
$r = 0.467$ on the panel.

## Reproducing the paper

```bash
bash scripts/run_all.sh                 # full S=12, L=300, n_iter=10 (Sec. 11)
bash scripts/run_all.sh --fast          # quick smoke run (S=4, L=80, n_iter=3)
bash scripts/run_all.sh --figures       # only figures
bash scripts/run_all.sh --tables        # only tables
bash scripts/run_all.sh --benchmarks    # only benchmarks
make report                             # regenerate figures + recompile docs/report/main.pdf
```

## Running benchmarks against real data (Phase 5)

The Phase 5 benches consume real datasets via the loaders in
`havi_methyl.io`: `load_finaleme_dataset`, `load_loyfer_atlas_matrix`,
`load_loyfer_pat_directory`, `load_roadmap_wgbs_atlas`.

```bash
# Loyfer 2023 tissue-of-origin LOO on the published U25 panel
# (36 cell types x 900 markers; HAVI Dirichlet head wins every metric).
python3 scripts/bench_tissue_loo.py \
    --atlas-tsv data/loyfer_panel/Atlas.U25.l4.hg38.tsv \
    --n-tissues 36 --loci 900 --samples 50
python3 scripts/fig_loyfer_loo_per_tissue.py     # per-tissue breakdown

# Liu 2024 paired cfDNA -> WGBS benchmark on a GPU.
# Requires the lab drive mounted (cf. data/finaleme_manifest/sample_pairs.csv
# for the donor-pairing manifest from Nature Comm Supplementary Table 1).
python3 scripts/bench_finaleme_realdata.py \
    --data-dir /path/to/finaleme \
    --manifest data/finaleme_manifest/sample_pairs.csv \
    --locus-panel data/finaleme_manifest/high_variance_cpgs.hg19.bed \
    --buffy-coat-bw /path/to/wgbs_buffyCoat_jensen2015GB.methy.hg19.bw \
    --torch-svi --torch-iter 500 --torch-device cuda --torch-iwae-k 4 \
    --torch-snapshot-every 20
python3 scripts/fig_finaleme_coverage_strat.py    # per-stratum breakdown
python3 scripts/fig_finaleme_paired_scatter.py    # hexbin density
python3 scripts/fig_finaleme_paired_metrics.py    # bar chart
python3 scripts/fig_elbo_trajectory.py            # training curve
```

If the lab drive is not mounted, `bench_finaleme_realdata.py` skips
the synthetic-proxy overwrite when a real `_status="Liu 2024..."`
CSV already exists at `outputs/tables/bench_finaleme_realdata.csv`
(documented in [scripts/run_all.sh](scripts/run_all.sh)).

The high-variance CpG panel is built by scanning the WGBS BEDs;
re-running is rarely needed since the panel is committed:

```bash
python3 scripts/build_high_variance_panel.py \
    --meth-dir /path/to/finaleme/meth_wgbs \
    --out data/finaleme_manifest/high_variance_cpgs.hg19.bed \
    --top-n 1000 --min-cov 1 --min-presence-frac 0.40 \
    --chroms chr1,chr19,chr20,chr21,chr22
```

`make report` requires a pdfLaTeX installation; on macOS, `brew install --cask
basictex` (≈100 MB, requires sudo) is the smallest path. The shipped
`docs/report/main.pdf` was built with the original deliverable's TeX
environment.

Every script accepts `--seed` (default `20260429`, the seed used for the
numbers reported in the manuscript) and `--fast`. Output:

- `outputs/figures/` — `recovery_scatter.{png,pdf}`, `calibration.{png,pdf}`,
  `elbo_trajectory.{png,pdf}`, `multiseed_recovery.{png,pdf}` (Sec. 11);
  `finaleme_paired_metrics.{png,pdf}`, `finaleme_paired_scatter.{png,pdf}`,
  `finaleme_coverage_strat.{png,pdf}`, `loyfer_loo_rmse.{png,pdf}`,
  `loyfer_loo_per_tissue.{png,pdf}` (Sec. 12 real-data). PDFs and PNGs
  are mirrored into `docs/report/figures/` so the LaTeX build can find them.
- `outputs/tables/` — every table in the manuscript as CSV (mirrored to
  `docs/report/tables/`).
- `outputs/results.json` — the JSON file that backs the numbers in Sec. 11
  (mirrored to `docs/report/results.json` and
  `docs/report/code/results.json`).
- `outputs/plot_data.npz` — per-coverage true / pred / interval / surrogate
  arrays consumed by the figure scripts. Regenerated only by
  `scripts/bench_synth_recovery.py`.

## Layout

```
src/havi_methyl/
  constants.py        Hyperparameters from Appendix F
  distributions.py    Closed-form densities, log-pmf, KL divergences
  likelihoods.py      Beta-Binomial / Categorical / NB reconstruction (Sec. 5.2)
  model.py            Hierarchical generative model (Sec. 3)
  varfamily.py        Variational factors (Sec. 4) + simplified posterior
  encoders.py         Set-Transformer interface (numpy + optional torch)
  flow.py             Rational-quadratic spline (App. C; numpy + optional torch)
  elbo.py             ELBO decomposition + IWAE + ESS (Sec. 5)
  svi.py              SVI loop (Sec. 6)
  identifiability.py  VIB / counterfactual / mQTL losses (Sec. 7)
  calibration.py      Conformal wrappers + Mondrian + CRC (Sec. 8)
  tissue.py           Dirichlet head + HDP truncation (Sec. 9)
  simulator.py        cfDNA simulator (Sec. 10, App. E)
  baseline.py         FinaleMe-style classifier + VB-HMM updates (App. B)
  bounds.py           Hierarchical pooling + Fano lower bound (Sec. 13)
  pipeline.py         End-to-end synthetic experiment harness (Sec. 11)
  utils.py            Numerics, RNG, metrics, validation
tests/                Pytest suite (156 tests; 9 torch-conditional skips on numpy-only install)
scripts/              Figure / table / benchmark scripts + run_all.sh
notebooks/            Tutorials: simulator, inference, calibration
docs/report/          The 52-page manuscript + supporting code
```

## API overview

### Simulator (Sec. 10, App. E)
`simulate_dataset`, `sample_methylation_track`, `sample_fragment_bag`,
`sample_fragment_lengths`, `sample_end_motifs`, `sample_coverage_nb`,
`SimulatorParams`, `cut_site_density`, `fragment_length_pdf`.

### Model (Sec. 3)
`HierarchicalModel`, `enforce_sum_zero_constraint`.

### Variational family (Sec. 4)
`PopulationLayer`, `SampleShiftLayer`, `GaussianFactor`,
`GaussianLocalPosterior`, `gaussian_observation_posterior`.

### Inference (Sec. 5–6)
`compute_elbo_gaussian`, `iwae_log_mean_exp`, `effective_sample_size`,
`fit_svi_simplified`, `predict_with_state`, `robbins_monro_step`.

### De-confounding & calibration (Sec. 7–8)
`vib_loss`, `vib_kl_to_unit_gaussian`, `mqtl_anchor_loss`,
`counterfactual_invariance_loss`, `prior_attribution_partial_r2`,
`vib_finite_leakage_bound`, `split_conformal_threshold`,
`gaussian_conformal_intervals`, `cqr_intervals`,
`mondrian_conformal_intervals`, `ConformalRiskController`.

### Tissue of origin (Sec. 9)
`dirichlet_alpha_from_logits`, `dirichlet_mean`, `too_loss_integrated`,
`stick_breaking`, `hdp_truncated_pi`,
`deconvolve_least_squares`, `binarize_and_deconvolve`.

### Theoretical bounds (Sec. 13)
`hierarchical_pooling_variance`, `hierarchical_pooling_shrinkage`,
`fano_error_lower_bound`, `fano_mse_lower_bound`,
`vib_information_upper_bound`.

### Pipeline (Sec. 11)
`run_synthetic_experiment`, `run_one_coverage`,
`run_identifiability_stress_test`, `run_tissue_recovery`, `cpg_poor_mask`.

### Metrics
`pearson_r`, `spearman_r`, `rmse`, `mae`, `empirical_coverage`,
`mean_interval_width`, `expected_calibration_error`, `icc_2_1`.

## Tests

```bash
make test                         # full pytest run
PYTHONPATH=src python -m pytest tests -v
```

Each test verifies a specific theoretical claim from the manuscript
(closed-form KL formulas vs. Monte Carlo, Fano bound monotonicity in MI,
hierarchical pooling shrinkage, conformal coverage at the nominal level,
simulator validation against published cfDNA distributions, etc.).

## License

MIT.

## Citation

If you use HAVI-Methyl, please cite the manuscript in `docs/report/main.pdf`.
