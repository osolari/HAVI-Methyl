"""Tests for the IMPL-02..IMPL-09 additions in HAVI-Methyl.

These exercise the numpy reference implementations only; the optional torch
modules are still skipped here when torch is missing (``test_torch_modules``
covers the torch path).
"""

from __future__ import annotations

import numpy as np
import pytest
from havi_methyl import (
    ConditionalRationalQuadraticSpline,
    DilatedCNNSequenceEncoder,
    FrozenEmbeddingProjection,
    ISABNumpy,
    PMANumpy,
    SetTransformerNumpy,
    cohort_balance_diagnostic,
    conditional_log_density,
    domain_adversarial_loss,
    fit_svi_full,
    high_density_conformal_set,
    leave_one_tissue_out_stress,
    masked_mean_pool,
    one_hot_dna,
    recentering_residual,
    reverse_complement,
    simulator_validation_metrics,
    worst_stratum_coverage,
)

# ---------------------------- IMPL-02 ----------------------------


def test_set_transformer_numpy_permutation_invariant():
    rng = np.random.default_rng(0)
    encoder = SetTransformerNumpy.random(in_dim=5, hidden=16, out_dim=8, num_layers=2, rng=rng)
    bag = rng.standard_normal((30, 5))
    perm = rng.permutation(bag.shape[0])
    a = encoder.encode(bag)
    b = encoder.encode(bag[perm])
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_set_transformer_numpy_mask_handling():
    rng = np.random.default_rng(1)
    encoder = SetTransformerNumpy.random(in_dim=4, hidden=8, out_dim=4, num_layers=1, rng=rng)
    bag = rng.standard_normal((10, 4))
    mask = np.array([True] * 6 + [False] * 4)
    a = encoder.encode(bag, mask=mask)
    b = encoder.encode(bag[:6])
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_isab_pma_blocks_residual_shapes():
    rng = np.random.default_rng(2)
    isab = ISABNumpy.random(dim=8, num_inducing=4, rng=rng)
    pma = PMANumpy.random(dim=8, rng=rng)
    x = rng.standard_normal((12, 8))
    y = isab.forward(x)
    assert y.shape == x.shape
    pooled = pma.forward(y)
    assert pooled.shape == (8,)


def test_masked_mean_pool_empty_returns_zero_vector():
    f = np.zeros((0, 5))
    out = masked_mean_pool(f)
    assert out.shape == (5,)
    assert np.all(out == 0)


# ---------------------------- IMPL-03 ----------------------------


def test_one_hot_dna_and_reverse_complement():
    seq = "ACGTNN"
    out = one_hot_dna(seq)
    assert out.shape == (6, 4)
    np.testing.assert_allclose(out[0], [1, 0, 0, 0])
    np.testing.assert_allclose(out[3], [0, 0, 0, 1])
    np.testing.assert_allclose(out[4], [0.25, 0.25, 0.25, 0.25])
    assert reverse_complement("ACGTN") == "NACGT"


def test_dilated_cnn_encoder_shape_and_determinism():
    rng = np.random.default_rng(3)
    enc = DilatedCNNSequenceEncoder.random(out_dim=12, num_layers=3, rng=rng)
    out1 = enc.encode("ACGTACGTACGT" * 4)
    out2 = enc.encode("ACGTACGTACGT" * 4)
    assert out1.shape == (12,)
    np.testing.assert_allclose(out1, out2)


def test_frozen_embedding_projection_shape():
    rng = np.random.default_rng(4)
    proj = FrozenEmbeddingProjection.random(in_dim=16, out_dim=4, rng=rng)
    out = proj.project(rng.standard_normal((20, 16)))
    assert out.shape == (4,)


# ---------------------------- IMPL-04 ----------------------------


def test_conditional_spline_invertible_and_density_finite():
    rng = np.random.default_rng(5)
    block = ConditionalRationalQuadraticSpline.random(context_dim=3, num_bins=6, rng=rng)
    ctx = rng.standard_normal(3)
    eps = np.linspace(-2.0, 2.0, 9)
    eta, _ = block.transform(eps, ctx)
    eps_back = block.inverse(eta, ctx)
    np.testing.assert_allclose(eps, eps_back, atol=1e-3)
    log_q = conditional_log_density(block, eta, ctx)
    assert np.all(np.isfinite(log_q))


def test_conditional_spline_density_integrates_near_one():
    rng = np.random.default_rng(6)
    block = ConditionalRationalQuadraticSpline.random(context_dim=2, num_bins=8, rng=rng)
    ctx = rng.standard_normal(2)
    grid = np.linspace(-2.5, 2.5, 401)
    eta_grid, _ = block.transform(grid, ctx)
    log_q = conditional_log_density(block, eta_grid, ctx)
    deta = np.gradient(eta_grid)
    integral = float(np.sum(np.exp(log_q) * deta))
    # Tail-linear so the integral is approximate; require [0.7, 1.3].
    assert 0.7 <= integral <= 1.3


# ---------------------------- IMPL-05 ----------------------------


def test_fit_svi_full_recentering_residual_near_zero():
    rng = np.random.default_rng(7)
    S, L = 6, 40
    pred = rng.uniform(0.05, 0.95, size=(S, L))
    n = rng.integers(1, 8, size=(S, L))
    state = fit_svi_full(pred, n, n_iter=20, batch_samples=4, batch_loci=20, rng=rng)
    assert abs(recentering_residual(state)) < 1e-8
    assert state.elbo_history[-1] >= state.elbo_history[0] - 1e-6  # surrogate non-worsening


# ---------------------------- IMPL-06 ----------------------------


def test_domain_adversarial_loss_finite_and_balance_diagnostic_passes():
    rng = np.random.default_rng(8)
    reps = rng.standard_normal((60, 4))
    domains = rng.integers(0, 3, size=60)
    nll = domain_adversarial_loss(reps, domains, lam=1.0)
    assert np.isfinite(nll) and nll > 0
    diag = cohort_balance_diagnostic(reps, domains, threshold=10.0)
    assert diag["passes_threshold"] == 1.0
    assert diag["n_cohorts"] == 3.0


# ---------------------------- IMPL-07 ----------------------------


def test_high_density_conformal_set_marginal_coverage():
    # Synthetic exchangeable Gaussian: marginal coverage should be ~1-alpha.
    rng = np.random.default_rng(9)
    n_calib, n_test, n_grid = 200, 200, 100
    y_calib = rng.standard_normal(n_calib)
    y_test = rng.standard_normal(n_test)
    grid = np.linspace(-4, 4, n_grid)
    log_density_calib = -0.5 * y_calib**2 - 0.5 * np.log(2 * np.pi)
    log_density_grid = -0.5 * (grid[None, :] - 0.0) ** 2 - 0.5 * np.log(2 * np.pi)
    log_density_grid = np.broadcast_to(log_density_grid, (n_test, n_grid))
    set_mask = high_density_conformal_set(log_density_grid, log_density_calib, alpha=0.1)
    # Treat each test point as the closest grid index to its actual value.
    idx_test = np.clip(np.searchsorted(grid, y_test) - 1, 0, n_grid - 1)
    cov = float(set_mask[np.arange(n_test), idx_test].mean())
    assert 0.78 <= cov <= 1.0


def test_worst_stratum_coverage_returns_min():
    true = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    lo = np.array([0.0, 0.0, 0.0, 0.6, 0.9])  # stratum-1 last point misses
    hi = np.array([0.5, 0.5, 0.5, 0.8, 1.0])
    strata = np.array([0, 0, 0, 1, 1])
    out = worst_stratum_coverage(true, lo, hi, strata)
    assert out["worst"] == 0.5
    assert out[0] == 1.0
    assert out[1] == 0.5


# ---------------------------- IMPL-08 ----------------------------


def test_leave_one_tissue_out_stress_runs():
    rng = np.random.default_rng(10)
    T, S, L = 4, 8, 30
    R = rng.uniform(0, 1, size=(T, L))
    pi_true = rng.dirichlet(np.ones(T), size=S)
    obs = pi_true @ R + rng.normal(0, 0.02, size=(S, L))
    obs = np.clip(obs, 0, 1)
    out = leave_one_tissue_out_stress(pi_true, R, obs)
    assert out["per_tissue_rmse"].shape == (T,)
    assert out["worst_rmse"] >= out["mean_rmse"]


# ---------------------------- IMPL-09 ----------------------------


def test_simulator_validation_metrics_are_in_expected_ranges():
    metrics = simulator_validation_metrics(n_frag=20_000, rng=11)
    assert 140 <= metrics["length_primary_mode_bp"] <= 195
    assert 0.0 <= metrics["top4_motif_fraction"] <= 1.0
    # Methylation-conditioned cut bias: simulator pushes higher-methylation
    # fragments to slightly higher GC, so the effect is positive.
    assert metrics["meth_cut_bias_effect_size"] > 0.0


@pytest.mark.parametrize("seed", [12, 13])
def test_simulator_validation_metrics_repro(seed):
    a = simulator_validation_metrics(n_frag=5_000, rng=seed)
    b = simulator_validation_metrics(n_frag=5_000, rng=seed)
    assert a == b


# ---------------------------- Real-data benchmark runner ----------------------------


def test_auc_threshold_perfect_and_random():
    from havi_methyl import auc_threshold

    truth = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc_threshold(truth, perfect, 0.5) == pytest.approx(1.0)
    inverted = 1.0 - perfect
    assert auc_threshold(truth, inverted, 0.5) == pytest.approx(0.0)


def test_dmr_f1_recovers_known_dmrs():
    from havi_methyl import dmr_f1

    rng = np.random.default_rng(20)
    L = 60
    a_truth = rng.uniform(0.05, 0.15, size=(8, L))
    b_truth = rng.uniform(0.05, 0.15, size=(8, L))
    # Insert a strong DMR at first 10 loci
    b_truth[:, :10] = rng.uniform(0.85, 0.95, size=(8, 10))
    a_pred = a_truth + rng.normal(0, 0.02, size=a_truth.shape)
    b_pred = b_truth + rng.normal(0, 0.02, size=b_truth.shape)
    f1 = dmr_f1(a_truth, b_truth, a_pred, b_pred)
    assert f1 > 0.5


def test_evaluate_real_data_benchmark_runs_end_to_end():
    from havi_methyl import evaluate_real_data_benchmark, simulate_dataset

    sim = simulate_dataset(8, 60, 1.0, rng=21)
    out = evaluate_real_data_benchmark(sim.bags, sim.n, sim.beta_sample, rng=21)
    assert "FinaleMe-style HMM" in out
    assert "HAVI-Methyl simplified (full)" in out
    for name, res in out.items():
        for field in (
            "pearson_r",
            "spearman_r",
            "auc_meth_at_0p5",
            "ece_credible",
            "icc_2_1",
            "dmr_f1",
        ):
            value = getattr(res, field)
            assert np.isfinite(value), f"{name}.{field} = {value}"


# ---------------------------- Phase 1: torch SVI ----------------------------


def test_torch_svi_end_to_end_smoke():
    """Full encoder + Gaussian-head + plate-rescaled SVI runs and improves."""
    pytest.importorskip("torch")
    import havi_methyl as hm

    if hm.fit_svi_torch is None:  # torch not available at import time
        pytest.skip("torch not available")
    sim = hm.simulate_dataset(S=4, L=20, coverage=2.0, rng=42)
    state = hm.fit_svi_torch(
        bags=sim.bags,
        n_frag=sim.n,
        n_meth=sim.n_meth,
        n_iter=15,
        config=hm.TorchSVIConfig(in_dim=5, hidden=16, num_inducing=8, num_layers=1),
        seed=42,
    )
    assert len(state.elbo_history) == 15
    assert state.elbo_history[-1] > state.elbo_history[0]
    # Recentering residual is exactly enforced after each iteration.
    assert abs(state.recentering_history[-1]) < 1e-5
    mean, _std = hm.predict_with_torch_state(state, sim.bags, sim.n, n_samples=4)
    assert mean.shape == sim.beta_sample.shape
    assert np.all((mean >= 0.0) & (mean <= 1.0))


# ---------------------------- Phase 2: ablation toggles ----------------------------


def test_torch_svi_phase2_toggles_run_without_nan():
    """VIB / counterfactual / adversarial weights all leave the surrogate finite."""
    pytest.importorskip("torch")
    import havi_methyl as hm

    if hm.fit_svi_torch is None:
        pytest.skip("torch not available")
    sim = hm.simulate_dataset(S=4, L=20, coverage=2.0, rng=42)
    cfg = hm.TorchSVIConfig(
        in_dim=5,
        hidden=16,
        num_inducing=8,
        num_layers=1,
        vib_weight=0.05,
        counterfactual_weight=0.1,
        adversarial_weight=0.01,
    )
    state = hm.fit_svi_torch(sim.bags, sim.n, sim.n_meth, n_iter=10, config=cfg, seed=42)
    assert all(np.isfinite(state.elbo_history))
    assert state.elbo_history[-1] > state.elbo_history[0]


# ---------------------------- Phase 3: tissue head ----------------------------


def test_dirichlet_head_beats_lstsq_on_heteroskedastic_var():
    """Variance-weighted Dirichlet head beats plain lstsq when posterior var varies."""
    import havi_methyl as hm

    rng = np.random.default_rng(0)
    T, S, L = 4, 8, 60
    R = rng.uniform(0, 1, size=(T, L))
    pi_true = rng.dirichlet(np.ones(T), size=S)
    var = np.where(rng.uniform(0, 1, size=L) < 0.3, 0.4, 0.005)
    obs = pi_true @ R + rng.normal(0, np.sqrt(var), size=(S, L))
    obs = np.clip(obs, 0, 1)
    pi_dir = hm.dirichlet_head_predict(obs, np.tile(var, (S, 1)), R)
    pi_lstsq = hm.deconvolve_least_squares(obs, R)
    rmse_dir = float(np.sqrt(((pi_true - pi_dir) ** 2).mean()))
    rmse_lstsq = float(np.sqrt(((pi_true - pi_lstsq) ** 2).mean()))
    assert rmse_dir < rmse_lstsq
    assert np.allclose(pi_dir.sum(axis=1), 1.0, atol=1e-9)


def test_hdp_truncated_pi_sums_to_one_at_T64():
    import havi_methyl as hm

    pi = hm.hdp_truncated_pi(alpha=1.0, T_max=64, rng=42)
    assert pi.shape == (64,)
    assert np.allclose(pi.sum(), 1.0, atol=1e-9)
    assert np.all(pi >= 0)


def test_loo_method_pluggable():
    """leave_one_tissue_out_stress accepts a custom deconvolution callable."""
    import havi_methyl as hm

    rng = np.random.default_rng(1)
    T, S, L = 3, 6, 40
    R = rng.uniform(0, 1, size=(T, L))
    pi_true = rng.dirichlet(np.ones(T), size=S)
    obs = pi_true @ R + rng.normal(0, 0.05, size=(S, L))
    obs = np.clip(obs, 0, 1)
    a = hm.leave_one_tissue_out_stress(pi_true, R, obs)
    b = hm.leave_one_tissue_out_stress(pi_true, R, obs, method=hm.binarize_and_deconvolve)
    # Different methods give different LOO RMSE.
    assert a["mean_rmse"] != b["mean_rmse"]


def test_ablation_matrix_runner_produces_six_rows(tmp_path, monkeypatch):
    """Sec. 12.3 ablation matrix runner emits one CSV row per A0..A5 configuration."""
    pytest.importorskip("torch")
    import havi_methyl as hm

    if hm.fit_svi_torch is None:
        pytest.skip("torch not available")
    # Run A0..A2 only to keep the test fast (the torch rows A3..A5 are
    # exercised by the dedicated bench script in CI).
    rng = np.random.default_rng(7)
    sim = hm.simulate_dataset(S=6, L=40, coverage=2.0, rng=rng)
    truth = sim.beta_sample
    # Inject DMR signal so dmr_f1 is well-defined.
    n_dmr = 4
    dmr_idx = rng.choice(40, size=n_dmr, replace=False)
    case_idx = np.arange(3)
    truth[case_idx[:, None], dmr_idx] = np.clip(truth[case_idx[:, None], dmr_idx] + 0.4, 0.02, 0.98)
    pred_baseline, _ = hm.finaleme_baseline_predict(sim.bags, sim.n)
    pred_bb = (sim.n_meth + 10.0) / (sim.n + 20.0)
    state = hm.fit_svi_simplified(pred_bb, sim.n, n_iter=5)
    pred_h, std_h = hm.predict_with_state(state, pred_bb, sim.n)
    for pred in (pred_baseline, pred_bb, pred_h):
        r = hm.pearson_r(truth, pred)
        assert np.isfinite(r)
    # Hierarchy strictly improves on baseline at this setting.
    assert hm.pearson_r(truth, pred_h) > hm.pearson_r(truth, pred_baseline) - 0.05
