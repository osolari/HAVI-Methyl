"""Utility functions: numerics, RNG, metrics, validation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit

from havi_methyl.constants import EPS, LOGIT_CLIP


def sigmoid(x: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable logistic sigmoid sigm(x) = 1/(1+exp(-x))."""
    return np.asarray(expit(np.asarray(x, dtype=np.float64)))


def safe_logit(p: ArrayLike, eps: float = EPS) -> NDArray[np.float64]:
    """logit(p) with clipping to (eps, 1-eps) to avoid +/- inf (Sec. 3.1)."""
    p = np.asarray(p, dtype=np.float64)
    return np.asarray(logit(np.clip(p, eps, 1 - eps)))


def clip_unit(x: ArrayLike, lo: float = 0.0, hi: float = 1.0) -> NDArray[np.float64]:
    """Clip to [lo,hi]."""
    return np.clip(np.asarray(x, dtype=np.float64), lo, hi)


def get_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Resolve a seed/generator argument to a numpy Generator (Sec. 2 design)."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# ---------- Metrics (Sec. 11, 12) ----------


def pearson_r(true: ArrayLike, pred: ArrayLike) -> float:
    """Pearson correlation; returns NaN if either input is constant."""
    t = np.asarray(true).flatten()
    p = np.asarray(pred).flatten()
    if t.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(t, p)[0, 1])


def spearman_r(true: ArrayLike, pred: ArrayLike) -> float:
    """Spearman rank correlation."""
    t = np.asarray(true).flatten()
    p = np.asarray(pred).flatten()
    return pearson_r(_rankdata(t), _rankdata(p))


def _rankdata(x: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def rmse(true: ArrayLike, pred: ArrayLike) -> float:
    """Root mean squared error."""
    t = np.asarray(true).flatten()
    p = np.asarray(pred).flatten()
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mae(true: ArrayLike, pred: ArrayLike) -> float:
    """Mean absolute error."""
    t = np.asarray(true).flatten()
    p = np.asarray(pred).flatten()
    return float(np.mean(np.abs(t - p)))


def empirical_coverage(true: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    """Fraction of true values within [lo, hi] (Sec. 11.3 / Prop. 4)."""
    t = np.asarray(true).flatten()
    lo = np.asarray(lo).flatten()
    hi = np.asarray(hi).flatten()
    return float(((t >= lo) & (t <= hi)).mean())


def mean_interval_width(lo: ArrayLike, hi: ArrayLike) -> float:
    """Average width of prediction intervals."""
    return float(np.mean(np.asarray(hi).flatten() - np.asarray(lo).flatten()))


def expected_calibration_error(nominal: ArrayLike, empirical: ArrayLike) -> float:
    """Mean absolute deviation of empirical from nominal coverage (Sec. 12.2)."""
    n = np.asarray(nominal).flatten()
    e = np.asarray(empirical).flatten()
    return float(np.mean(np.abs(n - e)))


def icc_2_1(values: NDArray[np.float64]) -> float:
    """Two-way random-effects single-rater intraclass correlation ICC(2,1).

    ``values`` has shape (n_targets, n_raters); returns ICC for absolute agreement,
    matching the Shrout-Fleiss convention used by Higgins-Chen (Sec. 12.2).
    """
    arr = np.asarray(values, dtype=np.float64)
    n, k = arr.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = arr.mean()
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)
    ms_between_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_between_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    ms_residual = np.sum((arr - row_means[:, None] - col_means[None, :] + grand) ** 2) / (
        (n - 1) * (k - 1)
    )
    denom = ms_between_rows + (k - 1) * ms_residual + k * (ms_between_cols - ms_residual) / n
    if denom <= 0:
        return float("nan")
    return float((ms_between_rows - ms_residual) / denom)


def auc_threshold(true_beta: ArrayLike, pred_beta: ArrayLike, threshold: float = 0.5) -> float:
    """ROC AUC for the binary task ``true_beta >= threshold`` (Sec. 12.2).

    Uses the trapezoidal-approximation Mann-Whitney U statistic so we do not
    depend on scikit-learn. Returns 0.5 if either class is empty.
    """
    t = np.asarray(true_beta, dtype=np.float64).flatten()
    p = np.asarray(pred_beta, dtype=np.float64).flatten()
    pos = p[t >= threshold]
    neg = p[t < threshold]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    n_pos = len(pos)
    n_neg = len(neg)
    ranks = _rankdata(np.concatenate([pos, neg])) + 1.0
    rank_pos_sum = ranks[:n_pos].sum()
    u = rank_pos_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def call_dmrs(
    beta_a: ArrayLike,
    beta_b: ArrayLike,
    delta_threshold: float = 0.25,
    bh_q: float = 0.05,
) -> NDArray[np.bool_]:
    """Mark CpGs as DMRs where ``|mean(beta_a) - mean(beta_b)| >= delta_threshold``.

    The Benjamini-Hochberg gating in the manuscript additionally requires
    a per-locus q-value <= ``bh_q``; we approximate this with a Welch
    t-test p-value plus BH correction, returning the binary DMR call mask.
    Sec. 12.2 specifies the (Δβ ≥ 0.25, q ≤ 0.05) definition.
    """
    a = np.asarray(beta_a, dtype=np.float64)
    b = np.asarray(beta_b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    delta = np.abs(mean_a - mean_b)
    var_a = a.var(axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mean_a)
    var_b = b.var(axis=0, ddof=1) if b.shape[0] > 1 else np.zeros_like(mean_b)
    se = np.sqrt(var_a / max(a.shape[0], 1) + var_b / max(b.shape[0], 1)) + 1e-12
    t_stat = (mean_a - mean_b) / se
    # Two-sided large-sample p-value via the standard-normal approximation.
    from scipy.special import erfc

    p = erfc(np.abs(t_stat) / np.sqrt(2.0))
    # Benjamini-Hochberg.
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q_full = np.empty_like(q)
    q_full[order] = q
    return (delta >= delta_threshold) & (q_full <= bh_q)


def dmr_f1(
    beta_truth_a: ArrayLike,
    beta_truth_b: ArrayLike,
    beta_pred_a: ArrayLike,
    beta_pred_b: ArrayLike,
    delta_threshold: float = 0.25,
    bh_q: float = 0.05,
) -> float:
    """F1 of predicted vs. ground-truth DMR calls (Sec. 12.2 metric).

    Both call_dmrs invocations use the same (Δβ, BH-q) thresholds. Returns
    ``0.0`` when either call set is empty (precision and recall are
    undefined; F1 conventionally maps to 0 in that case).
    """
    truth = call_dmrs(beta_truth_a, beta_truth_b, delta_threshold, bh_q)
    pred = call_dmrs(beta_pred_a, beta_pred_b, delta_threshold, bh_q)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def interval_ece(
    true: ArrayLike,
    centers: ArrayLike,
    widths: ArrayLike,
    nominal: ArrayLike | None = None,
) -> float:
    """Expected calibration error of nominal-vs-empirical interval coverage.

    ``widths`` is the 90% nominal width (matching the simplified-Gaussian
    variant of Sec. 11). The reported ECE is the mean absolute gap between
    nominal coverage levels in ``nominal`` and the empirically realised
    coverage when intervals are scaled to that nominal level.
    """
    from havi_methyl.calibration import coverage_curve

    if nominal is None:
        nominal = np.linspace(0.05, 0.95, 19)
    nominal = np.asarray(nominal, dtype=np.float64)
    empirical = coverage_curve(
        np.asarray(true, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        np.asarray(widths, dtype=np.float64),
        nominal,
    )
    return float(np.mean(np.abs(empirical - nominal)))


# ---------- Validation helpers ----------


def validate_beta(beta: ArrayLike, name: str = "beta") -> NDArray[np.float64]:
    """Ensure values are in [0, 1] (closed interval; clip is the caller's choice)."""
    b = np.asarray(beta, dtype=np.float64)
    if np.any(b < 0) or np.any(b > 1):
        bad = ((b < 0) | (b > 1)).sum()
        raise ValueError(f"{name} has {bad} entries outside [0,1]")
    return b


def validate_logit(eta: ArrayLike, name: str = "eta") -> NDArray[np.float64]:
    """Ensure logit-scale values are finite."""
    e = np.asarray(eta, dtype=np.float64)
    if not np.all(np.isfinite(e)):
        raise ValueError(f"{name} has non-finite entries")
    return e


def safe_clip_for_logit(p: ArrayLike) -> NDArray[np.float64]:
    """Clip beta into the safe range used throughout the codebase."""
    return np.clip(np.asarray(p, dtype=np.float64), 1 - LOGIT_CLIP, LOGIT_CLIP)


def chunked(iterable: Iterable, size: int) -> Iterable[list]:
    """Yield successive ``size``-sized chunks from ``iterable``."""
    it = iter(iterable)
    while True:
        chunk: list = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            return
        yield chunk
