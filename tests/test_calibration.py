"""Tests for conformal calibration wrappers (Sec. 8, Prop. 4)."""

from __future__ import annotations

import numpy as np
from havi_methyl import (
    ConformalRiskController,
    coverage_curve,
    cqr_intervals,
    gaussian_conformal_intervals,
    mondrian_conformal_intervals,
    split_conformal_threshold,
)


def test_split_conformal_marginal_coverage(rng):
    """Prop. 4: with exchangeable calibration/test, coverage >= 1 - alpha."""
    n_cal = 500
    n_test = 1000
    alpha = 0.1
    # All draws from N(0, 1); use |y - 0| / 1 as nonconformity, equivalent to abs(y).
    cal = np.abs(rng.standard_normal(n_cal))
    q_hat = split_conformal_threshold(cal, alpha)
    test = np.abs(rng.standard_normal(n_test))
    coverage = float((test <= q_hat).mean())
    assert coverage >= 1 - alpha - 0.05


def test_gaussian_conformal_coverage(rng):
    """Locally-adaptive Gaussian conformal achieves nominal coverage on the
    same generative process."""
    n_cal = 800
    n_test = 1500
    alpha = 0.1
    means_cal = rng.normal(0, 1, n_cal)
    stds_cal = rng.uniform(0.5, 1.5, n_cal)
    y_cal = means_cal + stds_cal * rng.standard_normal(n_cal)
    means_test = rng.normal(0, 1, n_test)
    stds_test = rng.uniform(0.5, 1.5, n_test)
    y_test = means_test + stds_test * rng.standard_normal(n_test)
    lo, hi = gaussian_conformal_intervals(means_cal, stds_cal, y_cal, means_test, stds_test, alpha)
    coverage = float(((y_test >= lo) & (y_test <= hi)).mean())
    assert coverage >= 1 - alpha - 0.03


def test_cqr_zero_residual_keeps_bounds(rng):
    """If quantile predictors are perfect, CQR adjustment q_hat is small."""
    n_cal = 500
    y = rng.uniform(-1.0, 1.0, n_cal)
    q_lo = y - 0.05
    q_hi = y + 0.05
    lo, hi = cqr_intervals(q_lo, q_hi, y, q_lo, q_hi, alpha=0.1)
    # Should be very close to [q_lo, q_hi]
    np.testing.assert_allclose(lo, q_lo, atol=0.1)
    np.testing.assert_allclose(hi, q_hi, atol=0.1)


def test_mondrian_per_stratum_coverage(rng):
    """Sec. 8.4: Mondrian calibration applies a per-stratum threshold."""
    n_cal = 600
    n_test = 800
    means_cal = rng.normal(0, 1, n_cal)
    stds_cal = rng.uniform(0.5, 1.5, n_cal)
    y_cal = means_cal + stds_cal * rng.standard_normal(n_cal)
    strata_cal = (stds_cal > 1.0).astype(int)
    means_test = rng.normal(0, 1, n_test)
    stds_test = rng.uniform(0.5, 1.5, n_test)
    y_test = means_test + stds_test * rng.standard_normal(n_test)
    strata_test = (stds_test > 1.0).astype(int)
    lo, hi = mondrian_conformal_intervals(
        means_cal, stds_cal, y_cal, strata_cal, means_test, stds_test, strata_test, alpha=0.1
    )
    coverage = float(((y_test >= lo) & (y_test <= hi)).mean())
    assert coverage >= 0.85


def test_conformal_risk_control(rng):
    """Sec. 8.5: smallest threshold whose corrected risk <= target."""
    y_cal = rng.uniform(0, 1, 500)
    thresholds = np.linspace(0, 1, 11)

    def loss(t, y):
        return (y > t).astype(float)

    crc = ConformalRiskController(loss_fn=loss, alpha=0.1)
    t_hat = crc.calibrate(y_cal, thresholds)
    # The selected threshold should be in the upper part of the range
    assert 0.7 <= t_hat <= 1.0


def test_coverage_curve_monotone():
    """coverage_curve evaluated on a Gaussian sample should be monotone."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500)
    centers = np.zeros_like(y)
    widths = np.full_like(y, 2 * 1.6448536269514722)  # exact 90% width
    nominal = np.linspace(0.1, 0.9, 9)
    out = coverage_curve(y, centers, widths, nominal)
    # Monotone non-decreasing across nominal levels
    assert np.all(np.diff(out) >= -0.05)
