"""HAVI-Methyl: Hierarchical Amortized Variational Inference for Methylation.

Public API surface. See ``docs/report/main.pdf`` for the full mathematical
specification; each module's docstring cites the relevant section.

Modules
-------
constants
    Default hyperparameters from Appendix F.
distributions
    Analytical density, log-pmf, log-pdf, and KL functions.
likelihoods
    Beta-Binomial / Categorical / Negative-Binomial reconstruction terms (Sec. 5.2).
model
    Hierarchical generative model on logit-beta (Sec. 3).
varfamily
    Variational family components (Sec. 4): Gaussian factors and conjugate
    posteriors used by the simplified Sec. 11 variant.
encoders, flow
    Set-Transformer and normalizing-flow implementations (Sec. 4.2-4.3,
    App. C, App. D). Pure-numpy fallbacks; optional torch counterparts.
elbo
    ELBO decomposition with mini-batch rescaling (Sec. 5).
svi
    Stochastic variational inference loop (Sec. 6, Algorithm 1).
identifiability
    VIB / counterfactual / mQTL anchor losses + leakage diagnostics (Sec. 7).
calibration
    Split-conformal, CQR, Mondrian, and conformal risk control (Sec. 8).
tissue
    Joint Dirichlet head with HDP truncation (Sec. 9).
simulator
    Chromatin-aware cfDNA fragmentation simulator (Sec. 10, App. E).
baseline
    FinaleMe-style classifier baseline + conjugate VB-HMM updates (App. B).
bounds
    Theoretical bounds: pooling variance reduction and Fano lower bound (Sec. 13).
pipeline
    End-to-end synthetic experiment harness (Sec. 11).
"""

from havi_methyl.baseline import (
    FinaleMeFit,
    NIWPosterior,
    finaleme_baseline_predict,
    finaleme_bootstrap_intervals,
    hmm_forward_backward,
    vbhmm_dirichlet_init_update,
    vbhmm_dirichlet_transition_update,
    vbhmm_niw_update,
)
from havi_methyl.bounds import (
    fano_error_lower_bound,
    fano_mse_lower_bound,
    hierarchical_pooling_shrinkage,
    hierarchical_pooling_variance,
    vib_information_upper_bound,
)
from havi_methyl.calibration import (
    ConformalRiskController,
    coverage_curve,
    cqr_intervals,
    gaussian_conformal_intervals,
    high_density_conformal_set,
    mondrian_conformal_intervals,
    split_conformal_threshold,
    worst_stratum_coverage,
)
from havi_methyl.constants import (
    DEFAULT_HPARAMS,
    Hyperparams,
)
from havi_methyl.distributions import (
    bernoulli_log_pmf,
    beta_binomial_grad_eta,
    beta_binomial_log_pmf,
    beta_binomial_log_pmf_from_beta,
    categorical_log_pmf,
    dirichlet_kl,
    dirichlet_log_pdf,
    gaussian_kl,
    gaussian_log_pdf,
    logit_normal_log_pdf,
    negative_binomial_log_pmf,
)
from havi_methyl.elbo import (
    ELBOTerms,
    compute_elbo_gaussian,
    effective_sample_size,
    iwae_log_mean_exp,
    kl_anneal_weight,
)
from havi_methyl.encoders import (
    DilatedCNNSequenceEncoder,
    FeedForwardNumpy,
    FrozenEmbeddingProjection,
    ISABNumpy,
    PMANumpy,
    SetMLPEncoder,
    SetTransformerNumpy,
    gelu,
    layer_norm,
    make_context_vector,
    masked_attention,
    masked_mean_pool,
    mean_pool_fragment_bag,
    multi_head_attention,
    one_hot_dna,
    reverse_complement,
    sum_pool_fragment_bag,
)
from havi_methyl.flow import (
    ConditionalRationalQuadraticSpline,
    RationalQuadraticSpline,
    StackedFlow,
    conditional_log_density,
)
from havi_methyl.identifiability import (
    IdentifiabilityResults,
    cohort_balance_diagnostic,
    counterfactual_invariance_loss,
    domain_adversarial_loss,
    mqtl_anchor_loss,
    prior_attribution_partial_r2,
    vib_finite_leakage_bound,
    vib_kl_to_unit_gaussian,
    vib_loss,
)
from havi_methyl.likelihoods import (
    end_motif_logits,
    joint_reconstruction_log_lik,
    reconstruction_log_lik_bb,
    reconstruction_log_lik_bern,
    reconstruction_log_lik_coverage,
    reconstruction_log_lik_motif,
)
from havi_methyl.model import HierarchicalModel, enforce_sum_zero_constraint
from havi_methyl.pipeline import (
    BenchmarkResult,
    CoverageRunResult,
    cpg_poor_mask,
    evaluate_real_data_benchmark,
    run_identifiability_stress_test,
    run_one_coverage,
    run_synthetic_experiment,
    run_tissue_recovery,
)
from havi_methyl.simulator import (
    SimulatedDataset,
    SimulatorParams,
    cut_site_density,
    cut_site_density_linker,
    fragment_length_pdf,
    sample_coverage_nb,
    sample_cut_positions,
    sample_end_motifs,
    sample_fragment_bag,
    sample_fragment_lengths,
    sample_methylation_track,
    sample_nucleosomes,
    simulate_dataset,
    simulate_dataset_chromatin_aware,
    simulator_validation_metrics,
)
from havi_methyl.svi import (
    SVIState,
    fit_svi_full,
    fit_svi_simplified,
    predict_with_state,
    recentering_residual,
    robbins_monro_step,
)

try:
    from havi_methyl.torch_svi import (
        TorchSVIConfig,
        TorchSVIState,
        fit_svi_torch,
        predict_with_torch_state,
    )
except ImportError:  # pragma: no cover
    TorchSVIConfig = None  # type: ignore[assignment,misc]
    TorchSVIState = None  # type: ignore[assignment,misc]
    fit_svi_torch = None  # type: ignore[assignment]
    predict_with_torch_state = None  # type: ignore[assignment]

from havi_methyl.io import (
    load_finaleme_dataset,
    load_loyfer_atlas_matrix,
    load_loyfer_pat_directory,
    load_roadmap_wgbs_atlas,
)
from havi_methyl.tissue import (
    TissueResults,
    binarize_and_deconvolve,
    deconvolve_least_squares,
    dirichlet_alpha_from_logits,
    dirichlet_head_predict,
    dirichlet_mean,
    hdp_truncated_deconvolve,
    hdp_truncated_pi,
    leave_one_tissue_out_stress,
    stick_breaking,
    too_loss_integrated,
)
from havi_methyl.utils import (
    auc_threshold,
    call_dmrs,
    dmr_f1,
    empirical_coverage,
    expected_calibration_error,
    icc_2_1,
    interval_ece,
    mae,
    mean_interval_width,
    pearson_r,
    rmse,
    safe_logit,
    sigmoid,
    spearman_r,
)
from havi_methyl.varfamily import (
    GaussianFactor,
    GaussianLocalPosterior,
    PopulationLayer,
    SampleShiftLayer,
    gaussian_observation_posterior,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_HPARAMS",
    "Hyperparams",
    "bernoulli_log_pmf",
    "beta_binomial_grad_eta",
    "beta_binomial_log_pmf",
    "beta_binomial_log_pmf_from_beta",
    "categorical_log_pmf",
    "dirichlet_kl",
    "dirichlet_log_pdf",
    "gaussian_kl",
    "gaussian_log_pdf",
    "logit_normal_log_pdf",
    "negative_binomial_log_pmf",
    "end_motif_logits",
    "joint_reconstruction_log_lik",
    "reconstruction_log_lik_bb",
    "reconstruction_log_lik_bern",
    "reconstruction_log_lik_coverage",
    "reconstruction_log_lik_motif",
    "HierarchicalModel",
    "enforce_sum_zero_constraint",
    "GaussianFactor",
    "GaussianLocalPosterior",
    "PopulationLayer",
    "SampleShiftLayer",
    "gaussian_observation_posterior",
    "ELBOTerms",
    "compute_elbo_gaussian",
    "effective_sample_size",
    "iwae_log_mean_exp",
    "kl_anneal_weight",
    "SVIState",
    "fit_svi_full",
    "fit_svi_simplified",
    "predict_with_state",
    "recentering_residual",
    "robbins_monro_step",
    "IdentifiabilityResults",
    "cohort_balance_diagnostic",
    "counterfactual_invariance_loss",
    "domain_adversarial_loss",
    "mqtl_anchor_loss",
    "prior_attribution_partial_r2",
    "vib_finite_leakage_bound",
    "vib_kl_to_unit_gaussian",
    "vib_loss",
    "ConformalRiskController",
    "cqr_intervals",
    "coverage_curve",
    "gaussian_conformal_intervals",
    "high_density_conformal_set",
    "mondrian_conformal_intervals",
    "split_conformal_threshold",
    "worst_stratum_coverage",
    "TissueResults",
    "binarize_and_deconvolve",
    "deconvolve_least_squares",
    "dirichlet_alpha_from_logits",
    "dirichlet_head_predict",
    "dirichlet_mean",
    "hdp_truncated_deconvolve",
    "hdp_truncated_pi",
    "leave_one_tissue_out_stress",
    "stick_breaking",
    "too_loss_integrated",
    "SimulatedDataset",
    "SimulatorParams",
    "cut_site_density",
    "cut_site_density_linker",
    "fragment_length_pdf",
    "sample_coverage_nb",
    "sample_cut_positions",
    "sample_end_motifs",
    "sample_fragment_bag",
    "sample_fragment_lengths",
    "sample_methylation_track",
    "sample_nucleosomes",
    "simulate_dataset",
    "simulate_dataset_chromatin_aware",
    "simulator_validation_metrics",
    "ConditionalRationalQuadraticSpline",
    "RationalQuadraticSpline",
    "StackedFlow",
    "conditional_log_density",
    "load_finaleme_dataset",
    "load_loyfer_atlas_matrix",
    "load_loyfer_pat_directory",
    "load_roadmap_wgbs_atlas",
    "DilatedCNNSequenceEncoder",
    "FeedForwardNumpy",
    "FrozenEmbeddingProjection",
    "ISABNumpy",
    "PMANumpy",
    "SetMLPEncoder",
    "SetTransformerNumpy",
    "gelu",
    "layer_norm",
    "make_context_vector",
    "masked_attention",
    "masked_mean_pool",
    "mean_pool_fragment_bag",
    "multi_head_attention",
    "one_hot_dna",
    "reverse_complement",
    "sum_pool_fragment_bag",
    "FinaleMeFit",
    "NIWPosterior",
    "finaleme_baseline_predict",
    "finaleme_bootstrap_intervals",
    "hmm_forward_backward",
    "vbhmm_dirichlet_init_update",
    "vbhmm_dirichlet_transition_update",
    "vbhmm_niw_update",
    "fano_error_lower_bound",
    "fano_mse_lower_bound",
    "hierarchical_pooling_shrinkage",
    "hierarchical_pooling_variance",
    "vib_information_upper_bound",
    "BenchmarkResult",
    "CoverageRunResult",
    "auc_threshold",
    "call_dmrs",
    "cpg_poor_mask",
    "dmr_f1",
    "evaluate_real_data_benchmark",
    "interval_ece",
    "run_identifiability_stress_test",
    "run_one_coverage",
    "run_synthetic_experiment",
    "run_tissue_recovery",
    "empirical_coverage",
    "expected_calibration_error",
    "icc_2_1",
    "mae",
    "mean_interval_width",
    "pearson_r",
    "rmse",
    "safe_logit",
    "sigmoid",
    "spearman_r",
    "__version__",
]
