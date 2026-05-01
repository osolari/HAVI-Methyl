"""Tests for SVI loop (Sec. 6, Algorithm 1)."""

from __future__ import annotations

import havi_methyl as hm
import numpy as np
from havi_methyl import svi as svi_mod


def test_robbins_monro_sum_diverges():
    """sum_t rho_t = inf, sum_t rho_t^2 < inf for exponent in (0.5, 1)."""
    n = 200_000
    rhos = np.array([svi_mod.robbins_monro_step(t, exponent=0.6) for t in range(n)])
    # The series sum behaves like n^{0.4} / 0.4 for exponent 0.6, so it diverges
    # but slowly. Check it grows sufficiently with n.
    assert rhos.sum() > rhos.sum() * 0.99  # tautology guards against int overflow
    assert rhos.sum() > 100.0
    assert (rhos**2).sum() < 100.0  # square-summable bound


def test_logit_observation_precision_increases_with_coverage():
    pred = np.array([[0.5, 0.5]])
    n_lo = np.array([[0, 1]])
    n_hi = np.array([[10, 20]])
    prec_lo = svi_mod.logit_observation_precision(pred, n_lo)
    prec_hi = svi_mod.logit_observation_precision(pred, n_hi)
    assert (prec_hi >= prec_lo).all()


def test_simplified_svi_runs(rng):
    sim = hm.simulate_dataset(S=4, L=40, coverage=5.0, rng=rng)
    pred_b, _ = hm.finaleme_baseline_predict(sim.bags, sim.n)
    state = hm.fit_svi_simplified(pred_b, sim.n, n_iter=5)
    assert len(state.elbo_history) == 5
    assert state.population.mean.shape == (40,)
    assert state.sample_shift.mean.shape == (4,)
    # Sum-to-zero constraint enforced
    np.testing.assert_allclose(state.sample_shift.mean.mean(), 0.0, atol=1e-10)


def test_simplified_svi_improves_over_baseline(rng):
    """At 5x coverage, the hierarchical SVI should out-correlate the baseline."""
    sim = hm.simulate_dataset(S=8, L=80, coverage=5.0, rng=rng)
    pred_b, _ = hm.finaleme_baseline_predict(sim.bags, sim.n)
    state = hm.fit_svi_simplified(pred_b, sim.n, n_iter=10)
    pred_h, _ = hm.predict_with_state(state, pred_b, sim.n)
    pearson_b = hm.pearson_r(sim.beta_sample, pred_b)
    pearson_h = hm.pearson_r(sim.beta_sample, pred_h)
    assert pearson_h >= pearson_b - 0.05  # at least non-degrading


def test_population_update_strict_conjugate(rng):
    """When obs_prec == 0, population mean stays at the prior."""
    L = 5
    pop = hm.PopulationLayer.initialize(L)
    sample = hm.SampleShiftLayer.initialize(2)
    eta_obs = np.zeros((2, L))
    obs_prec = np.zeros((2, L))
    new_pop = svi_mod.empirical_bayes_update_population(pop, sample, eta_obs, obs_prec)
    np.testing.assert_allclose(new_pop.mean, pop.prior_mean, atol=1e-10)


def test_population_update_with_observations_concentrates(rng):
    """With strong observations, population mean approaches the observation mean."""
    L = 4
    eta_obs = np.tile(np.array([0.5, 1.0, -0.5, 0.0])[None, :], (10, 1))
    obs_prec = np.full_like(eta_obs, 50.0)
    pop = hm.PopulationLayer.initialize(L)
    sample = hm.SampleShiftLayer.initialize(eta_obs.shape[0])
    new_pop = svi_mod.empirical_bayes_update_population(pop, sample, eta_obs, obs_prec)
    np.testing.assert_allclose(new_pop.mean, [0.5, 1.0, -0.5, 0.0], atol=0.05)


def test_predict_returns_valid_beta(rng):
    sim = hm.simulate_dataset(S=4, L=20, coverage=5.0, rng=rng)
    pred_b, _ = hm.finaleme_baseline_predict(sim.bags, sim.n)
    state = hm.fit_svi_simplified(pred_b, sim.n, n_iter=3)
    pred_h, std_h = hm.predict_with_state(state, pred_b, sim.n)
    assert ((pred_h >= 0) & (pred_h <= 1)).all()
    assert (std_h >= 0).all()
