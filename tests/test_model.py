"""Tests for the hierarchical generative model (Sec. 3)."""

from __future__ import annotations

import numpy as np

from havi_methyl import enforce_sum_zero_constraint
from havi_methyl.model import HierarchicalModel as HM


def test_sample_prior_shapes(rng):
    model = HM()
    out = model.sample_prior(S=8, L=64, rng=rng)
    assert out["mu_pop"].shape == (64,)
    assert out["delta"].shape == (8,)
    assert out["eta"].shape == (8, 64)
    assert out["beta_pop"].shape == (64,)
    assert ((out["beta_pop"] > 0) & (out["beta_pop"] < 1)).all()
    assert ((out["beta_sample"] > 0) & (out["beta_sample"] < 1)).all()


def test_sum_to_zero_constraint(rng):
    """eq. \\ref{eq:sumzero}: sum_s delta_s = 0 after re-centering."""
    model = HM()
    out = model.sample_prior(S=20, L=10, rng=rng, enforce_sum_zero=True)
    np.testing.assert_allclose(out["delta"].mean(), 0.0, atol=1e-10)


def test_constraint_invariance(rng):
    """Re-centering leaves eta_{s,l} invariant since shift moves between layers."""
    mu_pop = rng.normal(0, 2, size=20)
    delta = rng.normal(0, 0.5, size=10)
    eta_before = mu_pop[None, :] + delta[:, None]
    new_mu, new_delta = enforce_sum_zero_constraint(mu_pop, delta)
    eta_after = new_mu[None, :] + new_delta[:, None]
    np.testing.assert_allclose(eta_before, eta_after, atol=1e-12)
    np.testing.assert_allclose(new_delta.mean(), 0.0, atol=1e-12)


def test_prior_log_density_finite(rng):
    """log p(mu^pop, delta, eta) finite for typical inputs."""
    model = HM()
    out = model.sample_prior(S=4, L=12, rng=rng)
    lp = model.log_prior(out["mu_pop"], out["delta"], out["eta"])
    assert np.isfinite(lp)


def test_conditional_eta_mean_broadcast():
    model = HM()
    mu_pop = np.array([0.0, 1.0, -1.0])
    delta = np.array([0.5, -0.5])
    out = model.conditional_eta_mean(mu_pop, delta)
    assert out.shape == (2, 3)
    expected = np.array([[0.5, 1.5, -0.5], [-0.5, 0.5, -1.5]])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_prior_consistency_montecarlo(rng):
    """Empirical mean of mu^pop draws matches mu_0 in the large-sample limit."""
    model = HM(mu_0=0.5, tau_0=0.1)
    means = []
    for _ in range(50):
        out = model.sample_prior(S=2, L=200, rng=rng, enforce_sum_zero=False)
        means.append(out["mu_pop"].mean())
    np.testing.assert_allclose(np.mean(means), 0.5, atol=0.05)
