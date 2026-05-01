"""FinaleMe-style baselines and conjugate VB-HMM updates (App. B).

Two baselines:
  - ``finaleme_baseline_predict``: per-fragment Gaussian classifier with a
    one-step EM pass, mirroring the harness in ``docs/report/code``.
  - ``vbhmm_update_*``: closed-form conjugate VB-HMM updates from App. B
    (Beal 2003 Ch. 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, logsumexp

from havi_methyl.utils import get_rng, sigmoid

# ---------- FinaleMe-style classifier baseline (Sec. 11) ----------


@dataclass
class FinaleMeFit:
    """Bundle of fitted Gaussian-emission classifier params."""

    mu_meth: NDArray[np.float64]
    mu_unmeth: NDArray[np.float64]
    sd: NDArray[np.float64]


def finaleme_baseline_predict(
    bags: list[list[NDArray[np.float64]]],
    n: NDArray[np.intp],
    n_em: int = 8,
) -> tuple[NDArray[np.float64], FinaleMeFit]:
    """Per-fragment Gaussian classifier with a one-step EM seed.

    Replicates the FinaleMe-style baseline in ``run_experiments.py``: pool all
    fragments into a single feature matrix, two-cluster EM seeded by the
    median of feature 0 (length), then per-locus prediction = mean p_meth
    over the locus's fragments.
    """
    S, L = n.shape
    pred = np.full((S, L), 0.5)
    pool: list[NDArray[np.float64]] = []
    for s in range(S):
        for ell in range(L):
            f = bags[s][ell]
            if f.shape[0] > 0:
                pool.append(f)
    if not pool:
        return pred, FinaleMeFit(np.zeros(0), np.zeros(0), np.ones(1))
    all_feats = np.concatenate(pool, axis=0)
    p_m = (all_feats[:, 0] > np.median(all_feats[:, 0])).astype(float)
    mu_m = np.zeros(all_feats.shape[1])
    mu_u = np.zeros(all_feats.shape[1])
    sd = np.ones(all_feats.shape[1])
    for _ in range(n_em):
        mu_m = (all_feats * p_m[:, None]).sum(0) / max(p_m.sum(), 1.0)
        mu_u = (all_feats * (1.0 - p_m)[:, None]).sum(0) / max((1.0 - p_m).sum(), 1.0)
        sd = all_feats.std(0) + 1e-3
        ll_m = -0.5 * (((all_feats - mu_m) / sd) ** 2).sum(1)
        ll_u = -0.5 * (((all_feats - mu_u) / sd) ** 2).sum(1)
        p_m = sigmoid(ll_m - ll_u)
    for s in range(S):
        for ell in range(L):
            f = bags[s][ell]
            if f.shape[0] == 0:
                pred[s, ell] = 0.5
                continue
            ll_m = -0.5 * (((f - mu_m) / sd) ** 2).sum(1)
            ll_u = -0.5 * (((f - mu_u) / sd) ** 2).sum(1)
            pred[s, ell] = sigmoid(ll_m - ll_u).mean()
    return pred, FinaleMeFit(mu_meth=mu_m, mu_unmeth=mu_u, sd=sd)


def finaleme_bootstrap_intervals(
    bags: list[list[NDArray[np.float64]]],
    fit: FinaleMeFit,
    n_boot: int = 50,
    alpha: float = 0.10,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fragment-bootstrap intervals at level ``1 - alpha`` (Sec. 11.3 baseline)."""
    gen = get_rng(rng)
    S = len(bags)
    L = len(bags[0]) if S else 0
    lo = np.zeros((S, L))
    hi = np.zeros((S, L))
    for s in range(S):
        for ell in range(L):
            f = bags[s][ell]
            if f.shape[0] == 0:
                lo[s, ell], hi[s, ell] = alpha / 2, 1 - alpha / 2
                continue
            ks = f.shape[0]
            preds = np.empty(n_boot)
            for b in range(n_boot):
                idx = gen.integers(0, ks, ks)
                fb = f[idx]
                ll_m = -0.5 * (((fb - fit.mu_meth) / fit.sd) ** 2).sum(1)
                ll_u = -0.5 * (((fb - fit.mu_unmeth) / fit.sd) ** 2).sum(1)
                preds[b] = sigmoid(ll_m - ll_u).mean()
            lo[s, ell] = np.quantile(preds, alpha / 2)
            hi[s, ell] = np.quantile(preds, 1 - alpha / 2)
    return lo, hi


# ---------- VB-HMM conjugate updates (App. B) ----------


def dirichlet_geometric_mean(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    """exp(E[log pi]) = exp(psi(alpha) - psi(sum alpha)). App. B eq."""
    a = np.asarray(alpha, dtype=np.float64)
    return np.exp(digamma(a) - digamma(a.sum(axis=-1, keepdims=True)))


def vbhmm_dirichlet_init_update(
    alpha_0: NDArray[np.float64], gamma_t1: NDArray[np.float64]
) -> NDArray[np.float64]:
    """alpha_{s,k} = alpha_{0,k} + gamma_{s,1,k} (App. B initial-state update)."""
    return np.asarray(alpha_0, dtype=np.float64) + np.asarray(gamma_t1, dtype=np.float64)


def vbhmm_dirichlet_transition_update(
    alpha_A: NDArray[np.float64], xi_sum: NDArray[np.float64]
) -> NDArray[np.float64]:
    """alpha^A_{s,k,k'} = alpha_{A,k,k'} + sum_t xi_{s,t,k,k'} (App. B)."""
    return np.asarray(alpha_A, dtype=np.float64) + np.asarray(xi_sum, dtype=np.float64)


@dataclass
class NIWPosterior:
    """Bundle of NIW posterior parameters (App. B emission update)."""

    mu: NDArray[np.float64]
    kappa: float
    Psi: NDArray[np.float64]
    nu: float


def vbhmm_niw_update(
    prior: NIWPosterior,
    N_k: float,
    y_bar: NDArray[np.float64],
    S_k: NDArray[np.float64],
) -> NIWPosterior:
    """Beal-2003 eqs. 3.65-3.68 emission NIW update (App. B)."""
    kappa_q = prior.kappa + N_k
    mu_q = (prior.kappa * prior.mu + N_k * y_bar) / kappa_q
    nu_q = prior.nu + N_k
    diff = (y_bar - prior.mu).reshape(-1, 1)
    Psi_q = prior.Psi + S_k + (prior.kappa * N_k / kappa_q) * (diff @ diff.T)
    return NIWPosterior(mu=mu_q, kappa=kappa_q, Psi=Psi_q, nu=nu_q)


def hmm_forward_backward(
    log_init: NDArray[np.float64],
    log_trans: NDArray[np.float64],
    log_emit: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Forward-backward algorithm in log-space (App. B forward-backward).

    Inputs are log-probabilities of shape:
      log_init: (K,)
      log_trans: (K, K)
      log_emit: (T, K)
    Returns gamma (T, K), xi summed over t (K, K), log evidence.
    """
    T, K = log_emit.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_init + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = log_emit[t] + logsumexp(log_alpha[t - 1, :, None] + log_trans, axis=0)
    log_z = float(logsumexp(log_alpha[-1]))
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_trans + log_emit[t + 1, None, :] + log_beta[t + 1, None, :], axis=1
        )
    log_gamma = log_alpha + log_beta - log_z
    gamma = np.exp(log_gamma)
    # Sum xi over t
    xi = np.zeros((K, K))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t, :, None]
            + log_trans
            + log_emit[t + 1, None, :]
            + log_beta[t + 1, None, :]
            - log_z
        )
        xi = xi + np.exp(log_xi_t)
    return gamma, xi, log_z
