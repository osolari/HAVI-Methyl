"""Tissue-of-origin head with HDP truncation (Sec. 9).

Two heads:
  - Joint Dirichlet head with posterior integration (Sec. 9.1).
  - Hierarchical-Dirichlet-process truncation for novel-tissue discovery
    (Sec. 9.2 with truncation level T_max from Appendix F).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from havi_methyl.constants import T_MAX_HDP
from havi_methyl.utils import get_rng


def softplus(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def dirichlet_alpha_from_logits(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """alpha = softplus(W_R @ E_q[beta] + b_R), eq. of Sec. 9.1.

    A small floor 1e-3 is added so the Dirichlet remains proper.
    """
    return softplus(np.asarray(logits, dtype=np.float64)) + 1e-3


def dirichlet_mean(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    """E[pi_k] = alpha_k / sum_j alpha_j."""
    a = np.asarray(alpha, dtype=np.float64)
    return a / a.sum(axis=-1, keepdims=True)


def dirichlet_variance(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    """Var[pi_k] = alpha_k (alpha_0 - alpha_k) / (alpha_0^2 (alpha_0 + 1))."""
    a = np.asarray(alpha, dtype=np.float64)
    a0 = a.sum(axis=-1, keepdims=True)
    return a * (a0 - a) / (a0**2 * (a0 + 1))


def too_loss_integrated(
    pred_beta_mean: NDArray[np.float64],
    pred_beta_var: NDArray[np.float64],
    reference: NDArray[np.float64],
    pi: NDArray[np.float64],
    sigma_R: float,
) -> NDArray[np.float64]:
    """Integrated tissue-of-origin loss (Sec. 9.1, closed form).

    L_ToO(s) = sum_l log N(hat_mu; R^T pi_s, sigma_R^2 + hat_sigma^2_l)
                - 0.5 log ((sigma_R^2 + hat_sigma^2_l) / sigma_R^2)
    """
    pred_mu = np.asarray(pred_beta_mean, dtype=np.float64)
    pred_v = np.asarray(pred_beta_var, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    pred_from_pi = pi @ ref  # shape (S, L)
    var_total = sigma_R**2 + pred_v
    log_norm = -0.5 * np.log(2.0 * np.pi * var_total)
    sq = -0.5 * (pred_mu - pred_from_pi) ** 2 / var_total
    correction = -0.5 * np.log(var_total / sigma_R**2)
    return (log_norm + sq + correction).sum(axis=-1)


def deconvolve_least_squares(
    beta: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve pi = arg min ||R^T pi - beta||^2 with non-negativity and simplex
    projection. Used as a baseline (FinaleMe binarize-and-deconvolve, Sec. 11.5).
    """
    beta = np.asarray(beta, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    out = np.zeros((beta.shape[0], ref.shape[0]), dtype=np.float64)
    for s in range(beta.shape[0]):
        pi, *_ = np.linalg.lstsq(ref.T, beta[s], rcond=None)
        pi = np.clip(pi, 0.0, None)
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            pi = np.ones(ref.shape[0]) / ref.shape[0]
        out[s] = pi
    return out


def binarize_and_deconvolve(
    beta: NDArray[np.float64], reference: NDArray[np.float64], threshold: float = 0.5
) -> NDArray[np.float64]:
    """FinaleMe-style baseline: threshold predictions before solving the QP."""
    return deconvolve_least_squares((np.asarray(beta) > threshold).astype(np.float64), reference)


# ---------- HDP truncation (Sec. 9.2) ----------


def stick_breaking(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Beta-stick-breaking weights pi_k = v_k * prod_{k'<k} (1 - v_{k'}).

    Sec. 9.2; truncation at len(v) yields pi summing to 1 if v[-1] = 1.
    """
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros_like(v)
    remaining = 1.0
    for k in range(len(v)):
        out[k] = v[k] * remaining
        remaining = remaining * (1.0 - v[k])
    return out


def hdp_truncated_pi(
    alpha: float,
    T_max: int = T_MAX_HDP,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample stick-breaking pi from Beta(1, alpha), truncated at T_max.

    Sec. 9.2: variational truncation following Blei-Jordan 2006.
    """
    gen = get_rng(rng)
    v = gen.beta(1.0, alpha, size=T_max)
    v[-1] = 1.0  # truncation: force the last stick to consume the remainder
    return stick_breaking(v)


def leave_one_tissue_out_stress(
    pi_true: NDArray[np.float64],
    reference: NDArray[np.float64],
    obs_beta: NDArray[np.float64],
    method: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Sec. 9.3 LOO stress test: drop one reference column at a time.

    For each held-out tissue ``k`` we re-deconvolve the observed ``beta``
    using only the remaining ``T-1`` reference columns and report the RMSE
    of the recovered ``pi`` against the true mixture (also masked to the
    remaining tissues, renormalised). Returns the per-tissue RMSE plus the
    average and worst values.

    ``method`` is the deconvolution callable ``(Y, R) -> pi`` evaluated for
    each LOO fold. Defaults to ``deconvolve_least_squares`` (current
    simplified head); pass ``binarize_and_deconvolve`` for the FinaleMe
    baseline or wrap ``dirichlet_head_predict`` to evaluate the
    posterior-variance-aware head.
    """
    pi_true = np.asarray(pi_true, dtype=np.float64)
    R = np.asarray(reference, dtype=np.float64)
    Y = np.asarray(obs_beta, dtype=np.float64)
    T = R.shape[0]
    fn = method if method is not None else deconvolve_least_squares
    rmses = np.zeros(T)
    for k in range(T):
        keep = np.array([j for j in range(T) if j != k])
        R_kept = R[keep]
        pi_kept_true = pi_true[:, keep]
        denom = pi_kept_true.sum(axis=1, keepdims=True)
        pi_kept_true = np.where(denom > 0, pi_kept_true / np.maximum(denom, 1e-12), pi_kept_true)
        pi_recovered = fn(Y, R_kept)
        rmses[k] = float(np.sqrt(((pi_kept_true - pi_recovered) ** 2).mean()))
    return {
        "per_tissue_rmse": rmses,
        "mean_rmse": float(rmses.mean()),  # type: ignore[dict-item]
        "worst_rmse": float(rmses.max()),  # type: ignore[dict-item]
    }


@dataclass
class TissueResults:
    """Bundle of tissue-fraction metrics."""

    rmse_baseline: float
    rmse_havi: float

    def as_dict(self) -> dict[str, float]:
        return {"rmse_baseline": self.rmse_baseline, "rmse_havi": self.rmse_havi}


# ---------- Dirichlet head consuming posterior (mean, var) (Phase 3 / IMPL-08) ----------


def dirichlet_head_predict(
    beta_mean: NDArray[np.float64],
    beta_var: NDArray[np.float64],
    reference: NDArray[np.float64],
    sigma_R: float = 0.05,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Dirichlet-head MAP estimate per sample given posterior (mean, var).

    Closed-form variance-weighted lstsq deconvolution under the Sec. 9.1
    integrated likelihood with weights ``w_l = 1 / (sigma_R^2 + var_l)``
    — i.e. each locus contributes inversely with its posterior variance,
    which is the posterior-aware refinement the simple ``deconvolve_least_squares``
    baseline does not have. Result is projected to the simplex.

    Returns ``(S, T)`` mixture estimates.
    """
    _ = get_rng(rng)  # accepted for API symmetry; deterministic closed form
    pred_mu = np.asarray(beta_mean, dtype=np.float64)
    pred_v = np.asarray(beta_var, dtype=np.float64)
    R = np.asarray(reference, dtype=np.float64)
    S = pred_mu.shape[0]
    T = R.shape[0]
    sigma_R2 = sigma_R**2
    out = np.zeros((S, T), dtype=np.float64)
    for s in range(S):
        w = 1.0 / (sigma_R2 + pred_v[s])
        Rw = R * w[None, :]  # weighted reference rows: (T, L)
        # solve (R W R^T) pi = R W mu in the least-squares sense
        try:
            pi, *_ = np.linalg.lstsq(Rw.T, pred_mu[s] * w, rcond=None)
        except np.linalg.LinAlgError:
            pi = np.ones(T) / T
        pi = np.clip(pi, 0.0, None)
        total = pi.sum()
        out[s] = pi / total if total > 0 else np.ones(T) / T
    return out


def hdp_truncated_deconvolve(
    beta_mean: NDArray[np.float64],
    reference: NDArray[np.float64],
    alpha: float = 1.0,
    T_max: int = T_MAX_HDP,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Stick-breaking truncated mixture deconvolution (Sec. 9.2).

    Truncates the HDP at level ``T_max``, samples a stick-breaking prior
    ``pi`` of length ``T_max``, then runs least-squares deconvolution
    against the augmented reference (original tissues + zero-padded
    novel-tissue placeholders to length ``T_max``). Returns the
    truncated mixture matrix of shape ``(S, T_max)``.
    """
    gen = get_rng(rng)
    R = np.asarray(reference, dtype=np.float64)
    T_orig, L = R.shape
    if T_max < T_orig:
        T_max = T_orig
    R_aug = np.vstack([R, np.full((T_max - T_orig, L), 0.5)])
    Y = np.asarray(beta_mean, dtype=np.float64)
    # Stick-breaking prior used as a soft regulariser: blend with lstsq.
    prior = hdp_truncated_pi(alpha=alpha, T_max=T_max, rng=gen)
    pi = deconvolve_least_squares(Y, R_aug)
    return 0.9 * pi + 0.1 * prior[None, :]
