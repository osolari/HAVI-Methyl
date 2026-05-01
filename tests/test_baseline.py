"""Tests for baseline classifier and VB-HMM updates (App. B)."""

from __future__ import annotations

import havi_methyl as hm
import numpy as np
from havi_methyl.baseline import (
    NIWPosterior,
    dirichlet_geometric_mean,
    finaleme_baseline_predict,
    finaleme_bootstrap_intervals,
    hmm_forward_backward,
    vbhmm_dirichlet_init_update,
    vbhmm_dirichlet_transition_update,
    vbhmm_niw_update,
)


def test_finaleme_baseline_runs(rng):
    sim = hm.simulate_dataset(S=3, L=30, coverage=5.0, rng=rng)
    pred, fit = finaleme_baseline_predict(sim.bags, sim.n)
    assert pred.shape == (3, 30)
    assert ((pred >= 0) & (pred <= 1)).all()
    assert fit.mu_meth.shape == fit.mu_unmeth.shape


def test_bootstrap_intervals_in_unit_interval(rng):
    sim = hm.simulate_dataset(S=2, L=10, coverage=3.0, rng=rng)
    _, fit = finaleme_baseline_predict(sim.bags, sim.n)
    lo, hi = finaleme_bootstrap_intervals(sim.bags, fit, n_boot=10, rng=rng)
    assert lo.shape == sim.n.shape
    assert (hi >= lo).all()
    assert ((lo >= 0) & (hi <= 1)).all()


def test_dirichlet_init_update():
    """App. B: alpha_{s,k} = alpha_{0,k} + gamma_{s,1,k}."""
    alpha_0 = np.array([1.0, 1.0])
    gamma_t1 = np.array([0.7, 0.3])
    out = vbhmm_dirichlet_init_update(alpha_0, gamma_t1)
    np.testing.assert_allclose(out, np.array([1.7, 1.3]), atol=1e-12)


def test_dirichlet_transition_update():
    alpha_A = np.eye(2)
    xi_sum = np.array([[10.0, 1.0], [2.0, 8.0]])
    out = vbhmm_dirichlet_transition_update(alpha_A, xi_sum)
    np.testing.assert_allclose(out, alpha_A + xi_sum, atol=1e-12)


def test_dirichlet_geometric_mean_sums_below_one():
    """exp(E[log pi]) is bounded above by Dirichlet mean (Beal 2003)."""
    alpha = np.array([3.0, 5.0, 2.0])
    geom = dirichlet_geometric_mean(alpha)
    arith = alpha / alpha.sum()
    assert (geom <= arith + 1e-9).all()


def test_niw_update_consistent_for_zero_data():
    """No data should leave the NIW prior unchanged (App. B eqs.)."""
    prior = NIWPosterior(mu=np.zeros(2), kappa=2.0, Psi=np.eye(2), nu=3.0)
    out = vbhmm_niw_update(prior, N_k=0.0, y_bar=np.zeros(2), S_k=np.zeros((2, 2)))
    np.testing.assert_allclose(out.mu, prior.mu, atol=1e-12)
    np.testing.assert_allclose(out.kappa, prior.kappa, atol=1e-12)
    np.testing.assert_allclose(out.nu, prior.nu, atol=1e-12)


def test_forward_backward_marginalizes_to_one(rng):
    """Sum of gamma over states equals 1 for every t (App. B)."""
    K = 2
    T = 10
    log_init = np.log(np.array([0.5, 0.5]))
    log_trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    log_emit = rng.normal(0, 1, size=(T, K))
    gamma, xi, log_z = hmm_forward_backward(log_init, log_trans, log_emit)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-9)
    assert np.isfinite(log_z)


def test_forward_backward_known_chain():
    """Two-state chain with known stationary distribution: forward-backward
    should produce per-state marginals close to it."""
    K = 2
    T = 200
    log_init = np.log(np.array([0.5, 0.5]))
    log_trans = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    log_emit = np.zeros((T, K))  # uniform emissions
    gamma, _, _ = hmm_forward_backward(log_init, log_trans, log_emit)
    # All marginals equal 0.5 in this fully-symmetric model
    np.testing.assert_allclose(gamma, 0.5, atol=1e-8)
