# API reference

Generated with [mkdocstrings](https://mkdocstrings.github.io/). Every
symbol below is exported from the top-level `havi_methyl` package and
cites the manuscript section it implements. See
[`src/havi_methyl/__init__.py`](https://github.com/osolari/HAVI-Methyl/blob/main/src/havi_methyl/__init__.py)
for the full public surface.

## Inference (full torch stack)

::: havi_methyl.fit_svi_torch
    options:
      show_source: false

::: havi_methyl.TorchSVIConfig

::: havi_methyl.predict_with_torch_state
    options:
      show_source: false

## Inference (simplified-numpy)

::: havi_methyl.fit_svi_simplified
    options:
      show_source: false

## Data loaders

::: havi_methyl.load_finaleme_dataset
    options:
      show_source: false

::: havi_methyl.load_loyfer_atlas_matrix
    options:
      show_source: false

## Real-data evaluation

::: havi_methyl.evaluate_real_data_benchmark
    options:
      show_source: false

## Tissue-of-origin

::: havi_methyl.dirichlet_head_predict
    options:
      show_source: false

## Simulator

::: havi_methyl.simulate_dataset_chromatin_aware
    options:
      show_source: false

::: havi_methyl.simulator_validation_metrics
    options:
      show_source: false

## Module overview

For symbols not rendered above, the module-level layout follows the
manuscript section ordering:

| Module | Section | Public surface |
|---|---|---|
| `havi_methyl.distributions` | §3, App. A | `bernoulli_log_pmf`, `beta_binomial_log_pmf`, `dirichlet_kl`, `gaussian_kl`, ... |
| `havi_methyl.likelihoods` | §3.2, §5.2 | Beta-Binomial / Categorical / Negative-Binomial reconstruction terms |
| `havi_methyl.model` | §3 | `HierarchicalModel`, `enforce_sum_zero_constraint` |
| `havi_methyl.varfamily` | §4 | `PopulationLayer`, `SampleShiftLayer`, `GaussianFactor`, `GaussianLocalPosterior` |
| `havi_methyl.encoders` | §4.4 | `ISABNumpy`, `PMANumpy`, `SetTransformerNumpy`, torch `SetTransformerEncoder` |
| `havi_methyl.flow` | §4.3, App. C | `RationalQuadraticSpline`, torch `ConditionalNSFBlock` |
| `havi_methyl.elbo` | §5 | `compute_elbo_gaussian`, `iwae_log_mean_exp`, `effective_sample_size` |
| `havi_methyl.svi` | §6 | `fit_svi_simplified`, `predict_with_state`, `robbins_monro_step` |
| `havi_methyl.torch_svi` | §6 | `TorchSVIConfig`, `fit_svi_torch`, `predict_with_torch_state` |
| `havi_methyl.identifiability` | §7 | `vib_loss`, `mqtl_anchor_loss`, `counterfactual_invariance_loss`, `prior_attribution_partial_r2` |
| `havi_methyl.calibration` | §8 | `split_conformal_threshold`, `gaussian_conformal_intervals`, `cqr_intervals`, `mondrian_conformal_intervals`, `ConformalRiskController` |
| `havi_methyl.tissue` | §9 | `dirichlet_head_predict`, `dirichlet_alpha_from_logits`, `hdp_truncated_pi`, `deconvolve_least_squares` |
| `havi_methyl.simulator` | §10, App. E | `simulate_dataset`, `simulate_dataset_chromatin_aware`, `sample_fragment_bag`, `cut_site_density`, `fragment_length_pdf`, `simulator_validation_metrics` |
| `havi_methyl.baseline` | App. B | `FinaleMeFit`, `finaleme_baseline_predict`, `finaleme_bootstrap_intervals`, `hmm_forward_backward`, `vbhmm_dirichlet_init_update`, `vbhmm_niw_update` |
| `havi_methyl.bounds` | §13 | `hierarchical_pooling_variance`, `hierarchical_pooling_shrinkage`, `fano_error_lower_bound`, `fano_mse_lower_bound`, `vib_information_upper_bound` |
| `havi_methyl.pipeline` | §11 | `run_synthetic_experiment`, `run_one_coverage`, `run_identifiability_stress_test`, `run_tissue_recovery`, `cpg_poor_mask` |
| `havi_methyl.io` | §12 Phase 5 | `load_finaleme_dataset`, `load_loyfer_atlas_matrix`, `load_loyfer_pat_directory`, `load_roadmap_wgbs_atlas` |
| `havi_methyl.utils` | App. F | `pearson_r`, `spearman_r`, `rmse`, `mae`, `empirical_coverage`, `mean_interval_width`, `expected_calibration_error`, `icc_2_1` |

## Manuscript cross-references

- Standing observation regimes: §2.1 (three-regime distinction
  enforced by `load_finaleme_dataset` returning `ds.n` and `ds.n_total`
  separately).
- Three-level hierarchy: §3.1, Eqs. (1)–(4) (`havi_methyl.model`).
- Variational family: §4 (`havi_methyl.varfamily`,
  `havi_methyl.encoders`, `havi_methyl.flow`).
- ELBO decomposition: §5.1, Eq. (5) (`havi_methyl.elbo`).
- IWAE-DReG: §5.3 + Phase 1 IMPL-04 finetune
  (`TorchSVIConfig.k_iwae`, `TorchSVIConfig.iwae_dreg`).
- Inference algorithm: §6, Algorithm 1
  (`havi_methyl.svi.fit_svi_simplified`,
  `havi_methyl.torch_svi.fit_svi_torch`).
- VIB / mQTL / counterfactual / domain-adversarial: §7
  (`havi_methyl.identifiability`,
  `TorchSVIConfig.{vib_weight, mqtl_weight, counterfactual_weight, adversarial_weight}`).
- Conformal wrapper (Proposition 8.1): §8.2
  (`havi_methyl.calibration.gaussian_conformal_intervals`).
- Variance-weighted Dirichlet head: §9
  (`havi_methyl.tissue.dirichlet_head_predict`).
- HDP truncation: §9.2 (`havi_methyl.tissue.hdp_truncated_pi`).
- Chromatin-aware simulator: §10, App. E
  (`havi_methyl.simulator.simulate_dataset_chromatin_aware`).
- Hierarchical pooling variance: §13.1
  (`havi_methyl.bounds.hierarchical_pooling_variance`).
- Fano lower bound: §13.3
  (`havi_methyl.bounds.fano_error_lower_bound`).
