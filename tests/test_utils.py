"""Tests for utility functions."""

from __future__ import annotations

import numpy as np
from havi_methyl import utils


def test_sigmoid_logit_inverse():
    eta = np.array([-3, -1, 0, 1, 3], dtype=float)
    np.testing.assert_allclose(utils.safe_logit(utils.sigmoid(eta)), eta, atol=1e-10)


def test_safe_logit_clips_extremes():
    out = utils.safe_logit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(out))


def test_get_rng_idempotent():
    gen = utils.get_rng(42)
    same = utils.get_rng(gen)
    assert gen is same


def test_pearson_constant_is_nan():
    out = utils.pearson_r(np.zeros(10), np.arange(10))
    assert np.isnan(out)


def test_rmse_zero_when_equal():
    x = np.linspace(0, 1, 50)
    assert utils.rmse(x, x) == 0.0


def test_empirical_coverage_known():
    true = np.array([0.5, 0.5, 0.5])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 0.4, 1.0])
    np.testing.assert_allclose(utils.empirical_coverage(true, lo, hi), 2.0 / 3.0, atol=1e-12)


def test_mean_interval_width():
    np.testing.assert_allclose(
        utils.mean_interval_width(np.zeros(4), np.full(4, 0.3)), 0.3, atol=1e-12
    )


def test_expected_calibration_error_zero_for_ideal():
    nominal = np.linspace(0.1, 0.9, 9)
    np.testing.assert_allclose(utils.expected_calibration_error(nominal, nominal), 0.0)


def test_icc_perfect_replicates():
    """When all raters give the exact same value, ICC -> 1."""
    arr = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    out = utils.icc_2_1(arr)
    np.testing.assert_allclose(out, 1.0, atol=1e-9)


def test_icc_pure_noise_low():
    """Pure noise should give a low ICC."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 3))
    out = utils.icc_2_1(arr)
    assert -0.3 <= out <= 0.3


def test_chunked_yields_complete_chunks():
    out = list(utils.chunked(range(7), 3))
    assert out == [[0, 1, 2], [3, 4, 5], [6]]
