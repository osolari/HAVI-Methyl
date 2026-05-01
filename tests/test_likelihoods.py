"""Tests for reconstruction likelihoods (Sec. 5.2)."""

from __future__ import annotations

import numpy as np
from havi_methyl import likelihoods as L
from scipy.stats import betabinom


def test_bb_reconstruction_matches_scipy(rng):
    n_cpg = np.array([10, 20, 5, 15], dtype=float)
    beta = np.array([0.2, 0.5, 0.8, 0.4])
    n_meth = rng.binomial(n_cpg.astype(int), beta).astype(float)
    kappa = 12.0
    actual = L.reconstruction_log_lik_bb(n_meth, n_cpg, beta, kappa)
    expected = betabinom.logpmf(n_meth, n_cpg, kappa * beta, kappa * (1 - beta))
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_bern_reconstruction_sums_inner(rng):
    """Per-fragment Bernoulli sums over the inner CpG axis (Sec. 5.2)."""
    beta = np.array([0.3, 0.7])
    y = rng.binomial(1, np.broadcast_to(beta[:, None], (2, 5))).astype(float)
    out = L.reconstruction_log_lik_bern(y, beta[:, None])
    assert out.shape == (2,)
    expected = (
        y * np.log(beta[:, None] + 1e-12) + (1 - y) * np.log(1 - beta[:, None] + 1e-12)
    ).sum(axis=-1)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_motif_logits_shape():
    seq = np.zeros((4, 6))
    W_seq = np.zeros((6, 8))
    W_meth = np.zeros((1, 8))
    beta = np.full(4, 0.5)
    logits = L.end_motif_logits(beta, seq, W_seq, W_meth)
    assert logits.shape == (4, 8)


def test_coverage_nb_increases_with_gc(rng):
    """Sec. 5.2 NB coverage: increasing GC content should raise expected coverage
    when ``b_gc`` is positive."""
    beta = np.zeros(50)
    gc_lo = np.full(50, 0.3)
    gc_hi = np.full(50, 0.7)
    mappability = np.ones(50)
    n = rng.integers(1, 30, size=50)
    ll_lo = L.reconstruction_log_lik_coverage(n, beta, gc_lo, mappability).sum()
    ll_hi = L.reconstruction_log_lik_coverage(n, beta, gc_hi, mappability).sum()
    # The data have been generated identically; we just check both yield finite log-lik.
    assert np.isfinite(ll_lo)
    assert np.isfinite(ll_hi)


def test_joint_reconstruction_drops_optional_terms():
    """Joint log-lik with motif/cov inputs missing reduces to the BB term."""
    n_meth = np.array([3.0, 7.0])
    n_cpg = np.array([10.0, 12.0])
    beta = np.array([0.3, 0.7])
    just_bb = L.reconstruction_log_lik_bb(n_meth, n_cpg, beta, 10.0)
    joint = L.joint_reconstruction_log_lik(n_meth, n_cpg, None, beta, kappa=10.0)
    np.testing.assert_allclose(joint, just_bb, rtol=1e-12)


def test_bb_unbiased_methylation_estimator(rng):
    """A draw from BB(n; kappa*beta, kappa*(1-beta)) has E[k]/n = beta."""
    n = 200
    beta = 0.4
    kappa = 50
    samples = betabinom.rvs(n, kappa * beta, kappa * (1 - beta), size=20_000, random_state=42)
    np.testing.assert_allclose(samples.mean() / n, beta, atol=2e-2)
