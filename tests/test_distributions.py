"""Tests for analytic densities and KL divergences (Sec. 5.3, App. A)."""

from __future__ import annotations

import havi_methyl as hm
import numpy as np
from havi_methyl import distributions as D
from scipy.stats import beta as scipy_beta
from scipy.stats import betabinom, dirichlet, nbinom, norm


def test_gaussian_log_pdf_matches_scipy(rng):
    x = rng.normal(0, 2, size=64)
    actual = D.gaussian_log_pdf(x, 0.5, 1.5**2)
    expected = norm.logpdf(x, 0.5, 1.5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_gaussian_kl_zero_when_equal():
    kl = D.gaussian_kl(0.0, 4.0, 0.0, 4.0)
    assert abs(float(kl)) < 1e-12


def test_gaussian_kl_known_value():
    """KL(N(0,1) || N(0,4)) = 0.5*(1/4 + 0/4 - 1 - log(1/4)) = 0.5*(0.25 - 1 + log 4)."""
    kl = D.gaussian_kl(0.0, 1.0, 0.0, 4.0)
    expected = 0.5 * (0.25 - 1 - np.log(0.25))
    np.testing.assert_allclose(kl, expected, rtol=1e-12)


def test_gaussian_kl_montecarlo_matches_closed_form(rng):
    mq, vq = 0.4, 0.5
    mp, vp = -0.1, 1.2
    closed = float(D.gaussian_kl(mq, vq, mp, vp))
    samples = mq + np.sqrt(vq) * rng.standard_normal(200_000)
    mc = float(np.mean(D.gaussian_log_pdf(samples, mq, vq) - D.gaussian_log_pdf(samples, mp, vp)))
    np.testing.assert_allclose(mc, closed, atol=1e-2)


def test_beta_binomial_matches_scipy(rng):
    n = np.arange(5, 12)
    a, c = 3.0, 7.0
    for k in range(0, 5):
        actual = D.beta_binomial_log_pmf(np.full_like(n, k), n, a, c)
        expected = betabinom.logpmf(k, n, a, c)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_beta_binomial_normalizes(rng):
    """Sum over k of pmf should be 1 (Sec. 5.2)."""
    n, a, c = 12, 4.0, 6.0
    pmf = np.exp(D.beta_binomial_log_pmf(np.arange(n + 1), n, a, c))
    np.testing.assert_allclose(pmf.sum(), 1.0, rtol=1e-12)


def test_beta_binomial_grad_matches_finite_difference(rng):
    eta = rng.normal(0, 1.5, size=8)
    n = np.full_like(eta, 12)
    k = rng.integers(0, 13, size=8).astype(float)
    kappa = 5.0
    eps = 1e-5
    fd = (
        D.beta_binomial_log_pmf_from_beta(k, n, hm.sigmoid(eta + eps), kappa)
        - D.beta_binomial_log_pmf_from_beta(k, n, hm.sigmoid(eta - eps), kappa)
    ) / (2 * eps)
    grad = D.beta_binomial_grad_eta(k, n, eta, kappa)
    np.testing.assert_allclose(grad, fd, atol=1e-4)


def test_bernoulli_log_pmf_matches_definition(rng):
    beta = rng.uniform(0.05, 0.95, size=20)
    y = rng.binomial(1, beta).astype(float)
    actual = D.bernoulli_log_pmf(y, beta)
    expected = y * np.log(beta) + (1 - y) * np.log(1 - beta)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_categorical_log_pmf_normalizes():
    K = 7
    logits = np.zeros(K)
    pmf = np.exp([D.categorical_log_pmf(k, logits) for k in range(K)])
    np.testing.assert_allclose(pmf.sum(), 1.0, rtol=1e-12)


def test_negative_binomial_matches_scipy(rng):
    r, p = 4.0, 0.6
    n_obs = np.arange(0, 20)
    actual = D.negative_binomial_log_pmf(n_obs, r, p)
    expected = nbinom.logpmf(n_obs, r, 1 - p)  # scipy uses success-prob, our p is failure-rate
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_dirichlet_log_pdf_matches_scipy(rng):
    alpha = np.array([2.0, 3.0, 1.5])
    pi = rng.dirichlet(alpha)
    actual = float(D.dirichlet_log_pdf(pi, alpha))
    expected = float(dirichlet.logpdf(pi, alpha))
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_dirichlet_kl_zero_when_equal():
    alpha = np.array([2.0, 3.0, 1.0])
    np.testing.assert_allclose(D.dirichlet_kl(alpha, alpha), 0.0, atol=1e-12)


def test_dirichlet_kl_montecarlo(rng):
    aq = np.array([3.0, 2.0, 1.0])
    ap = np.array([1.0, 1.0, 1.0])
    closed = float(D.dirichlet_kl(aq, ap))
    samples = rng.dirichlet(aq, size=20000)
    mc = float(np.mean(D.dirichlet_log_pdf(samples, aq) - D.dirichlet_log_pdf(samples, ap)))
    np.testing.assert_allclose(mc, closed, atol=0.05)


def test_logit_normal_density_normalizes(rng):
    """Integrating logit-Normal density over (0,1) should give 1 (Sec. 3.2)."""
    grid = np.linspace(1e-3, 1 - 1e-3, 4001)
    pdf = np.exp(D.logit_normal_log_pdf(grid, 0.0, 1.0))
    integral = np.trapezoid(pdf, grid)
    np.testing.assert_allclose(integral, 1.0, atol=2e-2)


def test_beta_log_pdf_matches_scipy(rng):
    a, c = 3.5, 2.7
    grid = np.linspace(0.05, 0.95, 50)
    actual = D.beta_log_pdf(grid, a, c)
    expected = scipy_beta.logpdf(grid, a, c)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
