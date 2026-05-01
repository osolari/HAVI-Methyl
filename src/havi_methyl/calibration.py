"""Conformal prediction wrappers (Sec. 8, Prop. 4).

Three constructions:
  1. Split-conformal with negative-log-density nonconformity (Sec. 8.2).
  2. Conformal quantile regression / CQR (Sec. 8.3).
  3. Conformal risk control (Sec. 8.5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def split_conformal_threshold(nonconformity_calib: NDArray[np.float64], alpha: float) -> float:
    """Empirical 1-alpha quantile with the finite-sample correction.

    Index = ceil((1-alpha) * (n+1)) per Sec. 8.2.
    """
    s = np.sort(np.asarray(nonconformity_calib, dtype=np.float64))
    n = len(s)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((1 - alpha) * (n + 1)))
    rank = min(max(rank, 1), n)
    return float(s[rank - 1])


def conformal_intervals_negloglik(
    log_density_test: NDArray[np.float64],
    log_density_calib: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.bool_]:
    """Return per-test indicator of inclusion in the conformal prediction set.

    Nonconformity score r_i = -log f_i(beta*_i). The set
    C(F_new) = { beta : -log f_new(beta) <= q_hat } has marginal coverage
    >= 1 - alpha (Prop. 4 / \\ref{prop:conformal}).
    """
    r_calib = -np.asarray(log_density_calib, dtype=np.float64)
    q_hat = split_conformal_threshold(r_calib, alpha)
    r_test = -np.asarray(log_density_test, dtype=np.float64)
    return r_test <= q_hat


def cqr_intervals(
    q_lo_calib: NDArray[np.float64],
    q_hi_calib: NDArray[np.float64],
    y_calib: NDArray[np.float64],
    q_lo_test: NDArray[np.float64],
    q_hi_test: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Conformal quantile regression intervals (Sec. 8.3).

    Score r_i = max(q_lo(F_i) - y_i, y_i - q_hi(F_i)). The conformal interval
    is [q_lo - q_hat, q_hi + q_hat] with q_hat the (1-alpha)-quantile of r.
    """
    r = np.maximum(q_lo_calib - y_calib, y_calib - q_hi_calib)
    q_hat = split_conformal_threshold(r, alpha)
    return q_lo_test - q_hat, q_hi_test + q_hat


def gaussian_conformal_intervals(
    mean_calib: NDArray[np.float64],
    std_calib: NDArray[np.float64],
    y_calib: NDArray[np.float64],
    mean_test: NDArray[np.float64],
    std_test: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Locally-adaptive conformal interval using studentized residuals.

    r_i = |y_i - mean_i| / std_i, then q_hat = (1-alpha) quantile of r,
    interval = [mean - q_hat * std, mean + q_hat * std].
    """
    r = np.abs(y_calib - mean_calib) / np.maximum(std_calib, 1e-9)
    q_hat = split_conformal_threshold(r, alpha)
    return mean_test - q_hat * std_test, mean_test + q_hat * std_test


def mondrian_conformal_intervals(
    mean_calib: NDArray[np.float64],
    std_calib: NDArray[np.float64],
    y_calib: NDArray[np.float64],
    strata_calib: NDArray[np.intp],
    mean_test: NDArray[np.float64],
    std_test: NDArray[np.float64],
    strata_test: NDArray[np.intp],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-stratum conformal intervals (Sec. 8.4).

    Each stratum has its own q_hat. Useful for coverage-decile reporting and
    worst-stratum stress tests.
    """
    strata_calib = np.asarray(strata_calib, dtype=np.intp)
    strata_test = np.asarray(strata_test, dtype=np.intp)
    lo = np.zeros_like(mean_test)
    hi = np.zeros_like(mean_test)
    for stratum in np.unique(strata_test):
        cal_mask = strata_calib == stratum
        if cal_mask.sum() == 0:
            r = np.abs(y_calib - mean_calib) / np.maximum(std_calib, 1e-9)
        else:
            r = np.abs(y_calib[cal_mask] - mean_calib[cal_mask]) / np.maximum(
                std_calib[cal_mask], 1e-9
            )
        q_hat = split_conformal_threshold(r, alpha)
        test_mask = strata_test == stratum
        lo[test_mask] = mean_test[test_mask] - q_hat * std_test[test_mask]
        hi[test_mask] = mean_test[test_mask] + q_hat * std_test[test_mask]
    return lo, hi


@dataclass
class ConformalRiskController:
    """Conformal risk control (Sec. 8.5).

    For a monotone loss ``loss(threshold, y)``, find the smallest threshold
    whose finite-sample-corrected calibration risk satisfies the target ``alpha``.
    """

    loss_fn: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
    alpha: float

    def calibrate(
        self,
        y_calib: NDArray[np.float64],
        thresholds: NDArray[np.float64],
    ) -> float:
        """Return the smallest threshold whose corrected risk <= alpha."""
        n = len(y_calib)
        for t in np.sort(np.asarray(thresholds, dtype=np.float64)):
            losses = self.loss_fn(t, y_calib)
            risk = float(losses.mean()) * (n / (n + 1)) + 1.0 / (n + 1)
            if risk <= self.alpha:
                return float(t)
        return float(thresholds.max())


def coverage_curve(
    true: NDArray[np.float64],
    centers: NDArray[np.float64],
    widths: NDArray[np.float64],
    nominal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Empirical coverage at a sequence of nominal levels under Gaussian scaling.

    ``widths`` is the 90% nominal width (matching the simplified-Gaussian
    variant of Sec. 11). For each requested nominal q, the interval is
    rescaled by the corresponding standard-normal-quantile ratio.
    """
    from scipy.special import ndtri  # standard normal inverse cdf

    nominal = np.asarray(nominal, dtype=np.float64)
    out = np.empty_like(nominal)
    z90 = 1.6448536269514722  # standard normal 95th percentile
    for i, q in enumerate(nominal):
        zq = ndtri(0.5 * (1.0 + q))
        scale = abs(zq) / z90
        lo = centers - scale * widths / 2.0
        hi = centers + scale * widths / 2.0
        out[i] = float(((true >= lo) & (true <= hi)).mean())
    return out
