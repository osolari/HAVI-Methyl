"""Tests for theoretical bounds (Sec. 13)."""

from __future__ import annotations

import numpy as np
from havi_methyl import (
    fano_error_lower_bound,
    fano_mse_lower_bound,
    hierarchical_pooling_shrinkage,
    hierarchical_pooling_variance,
    vib_information_upper_bound,
)


def test_pooling_variance_strict_reduction():
    """Prop. 5: pooled variance < obs variance whenever pop_var < inf."""
    pop = np.array([0.5, 1.0, 2.0])
    obs = np.array([1.0, 1.0, 1.0])
    pooled = hierarchical_pooling_variance(pop, obs)
    assert (pooled < obs).all()


def test_pooling_shrinkage_in_unit_interval():
    pop = np.array([0.0, 0.5, 1.0, 10.0])
    obs = np.array([1.0, 1.0, 1.0, 1.0])
    rho = hierarchical_pooling_shrinkage(pop, obs)
    assert (rho >= 0).all()
    assert (rho < 1).all()
    # Strong pooling at pop_var = 0 (rho = 0)
    assert rho[0] == 0.0


def test_pooling_diffuse_limit():
    """As pop_var -> inf, pooled posterior approaches obs (no shrinkage)."""
    pooled = hierarchical_pooling_variance(np.array([1e6]), np.array([1.0]))
    np.testing.assert_allclose(pooled, 1.0, atol=1e-3)


def test_fano_lower_bound_montone():
    """As mutual information grows, the bound on P_e shrinks."""
    M = 100
    bound_lo = fano_error_lower_bound(0.0, M)
    bound_hi = fano_error_lower_bound(np.log(M), M)
    assert bound_lo > bound_hi
    assert bound_lo > 0
    assert bound_hi <= bound_lo


def test_fano_clipped_to_zero():
    """If I exceeds log M, bound becomes vacuous (clipped to 0)."""
    assert fano_error_lower_bound(100.0, M=10) == 0.0


def test_fano_mse_lower_bound_decreases_with_mi():
    M = 50
    mse_lo = fano_mse_lower_bound(0.0, M)
    mse_hi = fano_mse_lower_bound(2.0, M)
    assert mse_lo >= mse_hi


def test_vib_information_upper_bound_average():
    kls = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(vib_information_upper_bound(kls), 2.5, atol=1e-12)
