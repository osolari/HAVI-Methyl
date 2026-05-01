"""Tissue-of-origin head with HDP truncation (Sec. 9).

Two heads:
  - Joint Dirichlet head with posterior integration (Sec. 9.1).
  - Hierarchical-Dirichlet-process truncation for novel-tissue discovery
    (Sec. 9.2 with truncation level T_max from Appendix F).
"""

from __future__ import annotations

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


@dataclass
class TissueResults:
    """Bundle of tissue-fraction metrics."""

    rmse_baseline: float
    rmse_havi: float

    def as_dict(self) -> dict[str, float]:
        return {"rmse_baseline": self.rmse_baseline, "rmse_havi": self.rmse_havi}
