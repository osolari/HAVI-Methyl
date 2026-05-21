"""Tests for variational family (Sec. 4)."""

from __future__ import annotations

import numpy as np
import pytest

from havi_methyl import (
    GaussianFactor,
    GaussianLocalPosterior,
    PopulationLayer,
    SampleShiftLayer,
    gaussian_observation_posterior,
)


def test_gaussian_factor_constructor_validates():
    with pytest.raises(ValueError):
        GaussianFactor(np.zeros(3), np.array([-1.0, 1.0, 1.0]))


def test_gaussian_factor_log_prob_normalized(rng):
    g = GaussianFactor(np.zeros(1), np.ones(1))
    grid = np.linspace(-8, 8, 8001)
    log_p = np.array([g.log_prob(np.array([x])).item() for x in grid])
    pdf = np.exp(log_p)
    integral = float(np.trapezoid(pdf, grid))
    np.testing.assert_allclose(integral, 1.0, atol=1e-3)


def test_population_layer_kl_to_prior_zero():
    pop = PopulationLayer.initialize(L=5, prior_mean=0.0, prior_var=2.0)
    kl = pop.kl_to_prior()
    np.testing.assert_allclose(kl, 0.0, atol=1e-12)


def test_sample_shift_recenter(rng):
    """Sec. 4.1: re-center sample shifts and absorb into population layer."""
    pop = PopulationLayer.initialize(L=8, prior_mean=0.0, prior_var=1.0)
    pop.mean = rng.normal(0, 1, size=8)
    sample = SampleShiftLayer.initialize(S=5)
    sample.mean = rng.normal(0, 1, size=5)
    eta_before = pop.mean[None, :] + sample.mean[:, None]
    shift, new_pop_mean = sample.recenter_(pop.mean)
    eta_after = new_pop_mean[None, :] + sample.mean[:, None]
    np.testing.assert_allclose(eta_after, eta_before, atol=1e-12)
    np.testing.assert_allclose(sample.mean.mean(), 0.0, atol=1e-12)


def test_gaussian_observation_posterior_combines_correctly(rng):
    """Conjugate prior-times-likelihood (Sec. 4.5)."""
    eta_obs = np.array([[1.0, -1.0], [0.5, 0.0]])
    obs_prec = np.array([[2.0, 1.0], [4.0, 2.0]])
    prior_mean = np.array([[0.0, 0.0], [0.0, 0.0]])
    prior_prec = 1.0
    post = gaussian_observation_posterior(eta_obs, obs_prec, prior_mean, prior_prec)
    expected_mean = (prior_prec * prior_mean + obs_prec * eta_obs) / (prior_prec + obs_prec)
    np.testing.assert_allclose(post.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(post.var, 1.0 / (prior_prec + obs_prec), atol=1e-12)


def test_local_posterior_beta_var_delta_method():
    local = GaussianLocalPosterior(mean=np.array([0.0]), var=np.array([0.25]))
    beta, beta_var = local.beta_mean_var()
    expected_beta = 0.5
    expected_var = (0.5 * 0.5) ** 2 * 0.25
    np.testing.assert_allclose(beta, expected_beta, atol=1e-10)
    np.testing.assert_allclose(beta_var, expected_var, atol=1e-10)
