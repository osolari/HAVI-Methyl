"""Analytical density functions and KL divergences (Sec. 5.3, App. A).

All formulas reference the equation numbers in the report. ``log_pdf`` and
``log_pmf`` functions are vectorized; closed-form KL divergences are between
parametric exponential-family pairs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaln, digamma, gammaln

from havi_methyl.utils import sigmoid

# ---------- Gaussian (logit-scale) ----------


def gaussian_log_pdf(x: ArrayLike, mean: ArrayLike, var: ArrayLike) -> NDArray[np.float64]:
    """log N(x; mean, var) per element. (Sec. 3.2 eqs. 1-3)."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def gaussian_kl(
    mean_q: ArrayLike, var_q: ArrayLike, mean_p: ArrayLike, var_p: ArrayLike
) -> NDArray[np.float64]:
    """Closed-form KL(N(m_q,v_q) || N(m_p,v_p)). (App. A eq. ``appA-gauss-kl``)."""
    mq = np.asarray(mean_q, dtype=np.float64)
    vq = np.asarray(var_q, dtype=np.float64)
    mp = np.asarray(mean_p, dtype=np.float64)
    vp = np.asarray(var_p, dtype=np.float64)
    return 0.5 * (vq / vp + (mq - mp) ** 2 / vp - 1.0 - np.log(vq / vp))


# ---------- Beta-Binomial (Sec. 5.2 eq. ``bb``) ----------


def beta_binomial_log_pmf(
    k: ArrayLike, n: ArrayLike, alpha: ArrayLike, gamma: ArrayLike
) -> NDArray[np.float64]:
    """log BB(k; n, alpha, gamma) = log C(n,k) + log B(k+a, n-k+g) - log B(a,g).

    See Sec. 5.2 of the report.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return log_choose + betaln(k + a, n - k + g) - betaln(a, g)


def beta_binomial_log_pmf_from_beta(
    k: ArrayLike, n: ArrayLike, beta: ArrayLike, kappa: float
) -> NDArray[np.float64]:
    """log BB with concentration kappa parameterization (eq. \\ref{eq:bb})."""
    b = np.asarray(beta, dtype=np.float64)
    return beta_binomial_log_pmf(k, n, kappa * b, kappa * (1.0 - b))


def beta_binomial_grad_eta(
    k: ArrayLike, n: ArrayLike, eta: ArrayLike, kappa: float
) -> NDArray[np.float64]:
    """d/deta log BB(k; n, kappa*sigm(eta), kappa*(1-sigm(eta))).

    Closed-form digamma identity from Sec. 5.2 eq. ``bb-grad``.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    beta = sigmoid(eta)
    one_m_beta = 1.0 - beta
    sig_prime = beta * one_m_beta
    bracket = (
        digamma(k + kappa * beta)
        - digamma(n - k + kappa * one_m_beta)
        - digamma(kappa * beta)
        + digamma(kappa * one_m_beta)
    )
    return kappa * sig_prime * bracket


# ---------- Bernoulli (Sec. 5.2 eq. ``bern``) ----------


def bernoulli_log_pmf(y: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """log Bernoulli(y; beta), elementwise."""
    y = np.asarray(y, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    return y * np.log(b + 1e-12) + (1.0 - y) * np.log(1.0 - b + 1e-12)


# ---------- Categorical end-motif (Sec. 5.2) ----------


def categorical_log_pmf(idx: ArrayLike, logits: ArrayLike) -> NDArray[np.float64]:
    """log Categorical(idx | softmax(logits)) along the last axis of logits."""
    logits = np.asarray(logits, dtype=np.float64)
    log_norm = np.log(
        np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1)
    ) + logits.max(axis=-1)
    idx = np.asarray(idx, dtype=int)
    return np.take_along_axis(logits, idx[..., None], axis=-1).squeeze(-1) - log_norm


# ---------- Negative-Binomial coverage (Sec. 5.2) ----------


def negative_binomial_log_pmf(n_obs: ArrayLike, r: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    """log NB(n_obs; r, p) using the failure-count parameterization.

    Mean = r * p / (1 - p), so when ``p`` increases the expected coverage rises.
    """
    n_obs = np.asarray(n_obs, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return (
        gammaln(n_obs + r)
        - gammaln(r)
        - gammaln(n_obs + 1)
        + r * np.log(1.0 - p + 1e-12)
        + n_obs * np.log(p + 1e-12)
    )


# ---------- Logit-Normal density (Sec. 3.2) ----------


def logit_normal_log_pdf(beta: ArrayLike, mean: ArrayLike, var: ArrayLike) -> NDArray[np.float64]:
    """log p(beta) where logit(beta) ~ N(mean, var) (Sec. 3.2)."""
    b = np.asarray(beta, dtype=np.float64)
    eta = np.log(b + 1e-12) - np.log(1.0 - b + 1e-12)
    return gaussian_log_pdf(eta, mean, var) - np.log(b + 1e-12) - np.log(1.0 - b + 1e-12)


# ---------- Beta distribution helpers (variational comparison, Sec. 4.6) ----------


def beta_log_pdf(beta: ArrayLike, a: ArrayLike, c: ArrayLike) -> NDArray[np.float64]:
    """log Beta(beta; a, c) = (a-1) log b + (c-1) log(1-b) - log B(a,c)."""
    b = np.asarray(beta, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    return (a - 1) * np.log(b + 1e-12) + (c - 1) * np.log(1 - b + 1e-12) - betaln(a, c)


# ---------- Dirichlet (Sec. 9, eq. ``alpha_s``) ----------


def dirichlet_log_pdf(pi: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
    """log Dir(pi; alpha) along the last axis."""
    pi = np.asarray(pi, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    norm = gammaln(alpha.sum(axis=-1)) - gammaln(alpha).sum(axis=-1)
    return norm + np.sum((alpha - 1.0) * np.log(pi + 1e-12), axis=-1)


def dirichlet_kl(alpha_q: ArrayLike, alpha_p: ArrayLike) -> NDArray[np.float64]:
    """Closed-form KL(Dir(alpha_q) || Dir(alpha_p)) along the last axis."""
    aq = np.asarray(alpha_q, dtype=np.float64)
    ap = np.asarray(alpha_p, dtype=np.float64)
    sq = aq.sum(axis=-1, keepdims=True)
    sp = ap.sum(axis=-1, keepdims=True)
    term1 = gammaln(sq.squeeze(-1)) - gammaln(sp.squeeze(-1))
    term2 = (gammaln(ap) - gammaln(aq)).sum(axis=-1)
    term3 = ((aq - ap) * (digamma(aq) - digamma(sq))).sum(axis=-1)
    return term1 + term2 + term3
