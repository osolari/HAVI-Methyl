"""Tests for identifiability mechanisms (Sec. 7, Sec. 13)."""

from __future__ import annotations

import havi_methyl as hm
import numpy as np
from havi_methyl import identifiability as ident


def test_vib_kl_zero_at_unit_gaussian():
    mean = np.zeros((4, 8))
    var = np.ones((4, 8))
    kl = ident.vib_kl_to_unit_gaussian(mean, var)
    np.testing.assert_allclose(kl, 0.0, atol=1e-10)


def test_vib_kl_positive_when_off_unit():
    mean = np.full((4, 8), 1.0)
    var = np.full((4, 8), 0.25)
    kl = ident.vib_kl_to_unit_gaussian(mean, var)
    assert (kl > 0).all()


def test_counterfactual_invariance_zero_when_equal():
    """A swap that doesn't change predictions yields zero penalty (Sec. 7.2)."""
    pred = np.zeros((5, 10))
    loss = ident.counterfactual_invariance_loss(pred, pred, lam=1.0)
    assert loss == 0.0


def test_mqtl_loss_zero_at_truth():
    """The mQTL loss vanishes when (a, b) match exactly (Sec. 7.3)."""
    pop_mean = np.array([0.5, -0.2, 1.0])
    g = np.array([1.0, 0.0, 2.0])
    a = np.array([0.0, -0.2, 0.0])
    b = np.array([0.5, 0.0, 0.5])
    loss = ident.mqtl_anchor_loss(pop_mean, g, a, b, lam=1.0)
    assert loss == 0.0


def test_prior_attribution_decreases_with_regularization(rng):
    """Sec. 11.4 stress test: VIB and VIB+mQTL drive leakage down."""
    res = hm.run_identifiability_stress_test(S=12, L=200, coverage=5.0, rng=rng)
    assert res.leak_no_vib > res.leak_vib_only
    assert res.leak_vib_only > res.leak_vib_plus_mqtl
    # Final leakage should be small (matches Sec. 11.4 ~0.02% target order of magnitude)
    assert res.leak_vib_plus_mqtl < res.leak_no_vib / 5.0


def test_vib_finite_leakage_bound():
    """Corollary 1: eta_leak <= G(beta_VIB) / beta_VIB (Sec. 13)."""
    elbo_max = 100.0
    elbo_vib = 95.0
    bound = ident.vib_finite_leakage_bound(elbo_max, elbo_vib, beta_vib=2.0)
    np.testing.assert_allclose(bound, 2.5, atol=1e-12)
    # Negative gap clipped to zero
    assert ident.vib_finite_leakage_bound(0.0, 1.0, beta_vib=2.0) == 0.0
