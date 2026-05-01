"""Tests for the tissue-of-origin head (Sec. 9)."""

from __future__ import annotations

import numpy as np
from havi_methyl import (
    binarize_and_deconvolve,
    deconvolve_least_squares,
    dirichlet_alpha_from_logits,
    dirichlet_mean,
    hdp_truncated_pi,
    stick_breaking,
    too_loss_integrated,
)


def test_dirichlet_alpha_positive():
    logits = np.array([-3.0, 0.0, 4.0])
    alpha = dirichlet_alpha_from_logits(logits)
    assert (alpha > 0).all()


def test_dirichlet_mean_sums_to_one():
    alpha = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 1.0]])
    means = dirichlet_mean(alpha)
    np.testing.assert_allclose(means.sum(axis=-1), 1.0, atol=1e-10)


def test_stick_breaking_sums_to_one():
    """Sec. 9.2: stick-breaking with v_K = 1 yields sum_k pi_k = 1."""
    v = np.array([0.3, 0.4, 0.5, 1.0])
    pi = stick_breaking(v)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)


def test_hdp_truncated_sums_to_one(rng):
    pi = hdp_truncated_pi(alpha=2.0, T_max=8, rng=rng)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)


def test_too_integrated_loss_decreases_with_correct_pi(rng):
    """L_ToO should be highest at the true tissue mixture (Sec. 9.1)."""
    L = 50
    n_t = 3
    pi_true = np.array([0.5, 0.3, 0.2])[None, :]
    R = rng.uniform(0, 1, size=(n_t, L))
    pred_mu = pi_true @ R + rng.normal(0, 0.01, size=(1, L))
    pred_var = np.full_like(pred_mu, 0.01)
    correct = too_loss_integrated(pred_mu, pred_var, R, pi_true, sigma_R=0.1).item()
    wrong = too_loss_integrated(
        pred_mu, pred_var, R, np.array([0.1, 0.1, 0.8])[None, :], sigma_R=0.1
    ).item()
    assert correct > wrong


def test_deconvolve_recovers_pi(rng):
    """Continuous-beta deconvolution should approximately recover the true mix."""
    n_t = 3
    L = 200
    R = rng.uniform(0, 1, size=(n_t, L))
    pi_true = rng.dirichlet(np.ones(n_t), size=8)
    obs = pi_true @ R + rng.normal(0, 0.01, size=(8, L))
    pi_pred = deconvolve_least_squares(np.clip(obs, 0, 1), R)
    rmse = float(np.sqrt(np.mean((pi_true - pi_pred) ** 2)))
    assert rmse < 0.05


def test_binarize_loses_signal(rng):
    """Sec. 11.5: binarize-and-deconvolve has substantially higher RMSE."""
    n_t = 3
    L = 200
    R = rng.uniform(0, 1, size=(n_t, L))
    pi_true = rng.dirichlet(np.ones(n_t), size=8)
    obs = np.clip(pi_true @ R + rng.normal(0, 0.05, size=(8, L)), 0, 1)
    pi_continuous = deconvolve_least_squares(obs, R)
    pi_binary = binarize_and_deconvolve(obs, R)
    rmse_c = float(np.sqrt(np.mean((pi_true - pi_continuous) ** 2)))
    rmse_b = float(np.sqrt(np.mean((pi_true - pi_binary) ** 2)))
    assert rmse_b >= rmse_c
