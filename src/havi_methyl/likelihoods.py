"""Observation-model likelihoods (Sec. 5.2: \\eqref{eq:bern}, \\eqref{eq:bb},
\\eqref{eq:joint}).

These functions compose the reconstruction term of \\elbo (eq.
\\ref{eq:reconstruction}) and are used both during ELBO evaluation and during
posterior-predictive checks (Sec. 8.1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from havi_methyl.distributions import (
    bernoulli_log_pmf,
    beta_binomial_log_pmf_from_beta,
    categorical_log_pmf,
    negative_binomial_log_pmf,
)
from havi_methyl.utils import sigmoid


def reconstruction_log_lik_bb(
    n_meth: ArrayLike,
    n_cpg: ArrayLike,
    beta: ArrayLike,
    kappa: float,
) -> NDArray[np.float64]:
    """Beta-Binomial aggregate reconstruction (Sec. 5.2 eq. ``bb``).

    Returns the per-(s, l) log-likelihood ``log BB(n_meth; n_cpg, kappa*beta, kappa*(1-beta))``.
    """
    return beta_binomial_log_pmf_from_beta(n_meth, n_cpg, beta, kappa)


def reconstruction_log_lik_bern(y: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """Per-fragment-CpG Bernoulli reconstruction (Sec. 5.2 eq. ``bern``).

    ``y`` is shape (..., K) of binary indicators; the result is summed over the
    inner CpG axis.
    """
    y = np.asarray(y, dtype=np.float64)
    return bernoulli_log_pmf(y, np.broadcast_to(beta, y.shape)).sum(axis=-1)


def end_motif_logits(
    beta: ArrayLike,
    seq_context: ArrayLike,
    W_seq: ArrayLike,
    W_meth: ArrayLike,
    bias: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute end-motif logits ``softmax^{-1}(W_seq * c + W_meth * beta) + b``.

    Sec. 5.2; small linear analog of the MLP ``h_theta`` used in production. The
    motif likelihood for fragment j is then ``Cat(motif_j; softmax(logits))``.
    """
    beta = np.asarray(beta, dtype=np.float64)
    seq_context = np.asarray(seq_context, dtype=np.float64)
    logits = seq_context @ W_seq + np.outer(beta, np.ones(W_seq.shape[1])) * W_meth
    if bias is not None:
        logits = logits + np.asarray(bias, dtype=np.float64)
    return logits


def reconstruction_log_lik_motif(
    motif_idx: ArrayLike,
    logits: ArrayLike,
) -> NDArray[np.float64]:
    """log Categorical(motif_idx | softmax(logits)) per fragment (Sec. 5.2)."""
    return categorical_log_pmf(motif_idx, logits)


def coverage_nb_params(
    beta: ArrayLike,
    gc: ArrayLike,
    mappability: ArrayLike,
    alpha_r: float,
    beta_r: float,
    gamma_r: float,
    b0: float,
    b_gc: float,
    b_beta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the (r, p) of the Negative-Binomial coverage model (Sec. 5.2).

    r_l = exp(alpha_r * GC + beta_r * map + gamma_r), and
    p_l = sigm(b0 + b_gc * GC + b_beta * beta).
    """
    gc = np.asarray(gc, dtype=np.float64)
    mappability = np.asarray(mappability, dtype=np.float64)
    r = np.exp(alpha_r * gc + beta_r * mappability + gamma_r)
    p = sigmoid(b0 + b_gc * gc + b_beta * np.asarray(beta, dtype=np.float64))
    return r, p


def reconstruction_log_lik_coverage(
    n_frag: ArrayLike,
    beta: ArrayLike,
    gc: ArrayLike,
    mappability: ArrayLike,
    alpha_r: float = 1.0,
    beta_r: float = 0.0,
    gamma_r: float = 0.0,
    b0: float = 0.0,
    b_gc: float = 1.5,
    b_beta: float = -0.5,
) -> NDArray[np.float64]:
    """Negative-Binomial coverage log-likelihood (Sec. 5.2).

    Defaults follow the qualitative direction of the Snyder-2016 fit: GC content
    raises expected coverage, methylation suppresses it (via fragmentation bias).
    """
    r, p = coverage_nb_params(beta, gc, mappability, alpha_r, beta_r, gamma_r, b0, b_gc, b_beta)
    return negative_binomial_log_pmf(n_frag, r, p)


def joint_reconstruction_log_lik(
    n_meth: ArrayLike,
    n_cpg: ArrayLike,
    n_frag: ArrayLike,
    beta: ArrayLike,
    *,
    kappa: float,
    motif_idx: ArrayLike | None = None,
    motif_logits: ArrayLike | None = None,
    gc: ArrayLike | None = None,
    mappability: ArrayLike | None = None,
    coverage_kwargs: dict | None = None,
) -> NDArray[np.float64]:
    """Sum of the three modality log-likelihoods (Sec. 5.2 eq. ``joint``).

    Coverage and motif terms are dropped when their inputs are absent (matching
    the simplified harness in the synthetic experiments of Sec. 11).
    """
    ll = reconstruction_log_lik_bb(n_meth, n_cpg, beta, kappa)
    if motif_idx is not None and motif_logits is not None:
        per_frag = reconstruction_log_lik_motif(motif_idx, motif_logits)
        ll = ll + per_frag.sum(axis=-1)
    if gc is not None and mappability is not None and n_frag is not None:
        kw = coverage_kwargs or {}
        ll = ll + reconstruction_log_lik_coverage(n_frag, beta, gc, mappability, **kw)
    return ll
