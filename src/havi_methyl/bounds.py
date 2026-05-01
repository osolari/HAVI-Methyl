"""Theoretical bounds and propositions (Sec. 13).

- Proposition 5 (\\ref{prop:pooling}): hierarchical pooling variance reduction.
- Proposition 6 (\\ref{prop:fano}): Fano-style information-theoretic
  recoverability lower bound.
- Corollary 1 (\\ref{cor:vib-finite}): see ``identifiability.vib_finite_leakage_bound``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def hierarchical_pooling_variance(pop_var: ArrayLike, obs_var: ArrayLike) -> NDArray[np.float64]:
    """Posterior variance under conjugate Gaussian pooling.

    Var[hat_eta_HAVI] = pop_var * obs_var / (pop_var + obs_var) (Prop. 5).
    Strict reduction over independent estimator's ``obs_var`` whenever
    ``pop_var < inf``.
    """
    pv = np.asarray(pop_var, dtype=np.float64)
    ov = np.asarray(obs_var, dtype=np.float64)
    return pv * ov / (pv + ov)


def hierarchical_pooling_shrinkage(pop_var: ArrayLike, obs_var: ArrayLike) -> NDArray[np.float64]:
    """rho_l = pop_var / (pop_var + obs_var); 0 = strong, 1 = weak pooling."""
    pv = np.asarray(pop_var, dtype=np.float64)
    ov = np.asarray(obs_var, dtype=np.float64)
    return pv / (pv + ov)


# ---------- Fano lower bound ----------


def fano_error_lower_bound(mutual_info: float, M: int) -> float:
    """P_e >= 1 - (I(beta; F) + log 2) / log M (Prop. 6).

    ``mutual_info`` is the mutual information between the latent and the
    fragment bag in nats, ``M`` is the number of beta bins. Returns 0 when
    the upper-bound exceeds 1 (the bound is trivial).
    """
    if M <= 1:
        return 0.0
    bound = 1.0 - (float(mutual_info) + np.log(2.0)) / np.log(M)
    return float(max(0.0, bound))


def fano_mse_lower_bound(mutual_info: float, M: int) -> float:
    """Asymptotic MSE bound: E[(hat_beta - beta)^2] >= P_e / (2M)^2 (Prop. 6).

    Returns the lower bound on MSE assuming the worst-case bin spacing.
    """
    p_e = fano_error_lower_bound(mutual_info, M)
    return float(p_e / (2 * M) ** 2)


def estimate_mutual_information_gaussian_channel(snr_db: float, var_input: float = 1.0) -> float:
    """Closed-form Shannon capacity of a Gaussian channel: 0.5 log(1 + SNR) (nats).

    Useful as an analytical comparison against simulator-derived MI estimates.
    """
    snr = 10 ** (snr_db / 10.0)
    return 0.5 * float(np.log(1.0 + snr))


# ---------- VIB asymptotic bound (informational corollary) ----------


def vib_information_upper_bound(
    encoder_kl_to_proxy: NDArray[np.float64],
) -> float:
    """I(zeta; X_prior) <= E_{X_prior}[KL(q(zeta|X_prior) || r(zeta))] (Sec. 7.1).

    Returns the empirical mean KL across rows.
    """
    return float(np.mean(np.asarray(encoder_kl_to_proxy, dtype=np.float64)))
