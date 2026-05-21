"""Tests for ELBO components (Sec. 5, App. A)."""

from __future__ import annotations

import numpy as np

from havi_methyl import (
    GaussianLocalPosterior,
    PopulationLayer,
    SampleShiftLayer,
    compute_elbo_gaussian,
    effective_sample_size,
    iwae_log_mean_exp,
    kl_anneal_weight,
)
from havi_methyl import elbo as elbo_mod


def test_iwae_logmeanexp_matches_direct():
    log_w = np.log(np.array([1.0, 2.0, 4.0, 8.0]))
    log_w = log_w[:, None]  # K=4, batch=1
    out = iwae_log_mean_exp(log_w)
    expected = np.log(np.mean(np.exp(log_w)))
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_iwae_logmeanexp_numerically_stable():
    """Adding a large constant should not destabilize log-mean-exp (Sec. 5.4)."""
    log_w = np.array([[1000.0, 1001.0, 999.0]]).T
    out = iwae_log_mean_exp(log_w)
    # Equivalent to 1000 + log_mean_exp([0, 1, -1])
    expected = 1000 + np.log(np.mean(np.exp(np.array([0.0, 1.0, -1.0]))))
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_ess_bounds():
    """Sec. 6 ESS = (sum w)^2 / sum w^2, in [1, K]."""
    log_w = np.zeros((10, 5))  # all weights equal -> ESS = K
    np.testing.assert_allclose(effective_sample_size(log_w), 10.0)
    log_w_skewed = np.full((10, 1), -10.0)
    log_w_skewed[0, 0] = 0.0  # one dominant weight -> ESS ~= 1
    ess = effective_sample_size(log_w_skewed).flatten()
    assert 1.0 <= float(ess[0]) < 1.5


def test_kl_anneal_schedule():
    assert kl_anneal_weight(0, 100) == 0.0
    assert kl_anneal_weight(50, 100) == 0.5
    assert kl_anneal_weight(150, 100) == 1.0
    assert kl_anneal_weight(0, 0) == 1.0


def test_elbo_gaussian_decomposes(rng):
    """Sec. 5.0 decomposition: total = recon - kl_pop - kl_sample - kl_local."""
    S, L = 4, 8
    n_meth = rng.integers(0, 11, size=(S, L)).astype(float)
    n_cpg = np.full((S, L), 10.0)
    pop = PopulationLayer.initialize(L)
    pop.mean = rng.normal(0, 1, size=L)
    sample = SampleShiftLayer.initialize(S)
    sample.mean = rng.normal(0, 0.5, size=S)
    local = GaussianLocalPosterior(
        mean=rng.normal(0, 1, size=(S, L)),
        var=rng.uniform(0.1, 1.0, size=(S, L)),
    )
    terms = compute_elbo_gaussian(n_meth, n_cpg, pop, sample, local, sigma_eta=0.8, kappa=10.0)
    np.testing.assert_allclose(
        terms.total,
        terms.reconstruction - terms.kl_population - terms.kl_sample - terms.kl_local,
        atol=1e-10,
    )


def test_elbo_minibatch_rescaling_linearity(rng):
    """Sec. 5.6 / Table rescale: rescaling factors enter linearly."""
    S, L = 4, 8
    n_meth = rng.integers(0, 11, size=(S, L)).astype(float)
    n_cpg = np.full((S, L), 10.0)
    pop = PopulationLayer.initialize(L)
    sample = SampleShiftLayer.initialize(S)
    local = GaussianLocalPosterior(mean=np.zeros((S, L)), var=np.ones((S, L)))
    full = compute_elbo_gaussian(
        n_meth, n_cpg, pop, sample, local, sigma_eta=0.8, kappa=10.0, rescale=(1.0, 1.0)
    )
    half = compute_elbo_gaussian(
        n_meth, n_cpg, pop, sample, local, sigma_eta=0.8, kappa=10.0, rescale=(2.0, 1.0)
    )
    np.testing.assert_allclose(half.kl_sample, 2.0 * full.kl_sample, atol=1e-10)
    np.testing.assert_allclose(half.kl_population, full.kl_population, atol=1e-10)


def test_montecarlo_local_kl_zero_when_match(rng):
    """KL collapses to zero when q == p (Prop. 1)."""
    eta_sample = rng.normal(0, 1, size=10)
    log_q = -0.5 * (np.log(2 * np.pi * 1.0) + eta_sample**2 / 1.0)
    kl = elbo_mod.montecarlo_local_kl(eta_sample, log_q, np.zeros(10), 1.0)
    np.testing.assert_allclose(kl.sum(), 0.0, atol=1e-10)
