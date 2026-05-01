"""ELBO computation (Sec. 5, App. A).

Closed-form Gaussian-Gaussian KLs plus Monte-Carlo reconstruction term, with
the mini-batch rescaling factors of Table~\\ref{tab:rescale}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from havi_methyl.constants import KAPPA
from havi_methyl.distributions import gaussian_kl, gaussian_log_pdf
from havi_methyl.likelihoods import reconstruction_log_lik_bb
from havi_methyl.utils import sigmoid
from havi_methyl.varfamily import (
    GaussianLocalPosterior,
    PopulationLayer,
    SampleShiftLayer,
)


@dataclass
class ELBOTerms:
    """Decomposition of the ELBO into recon, KL_pop, KL_delta, KL_local.

    Mirrors eq. \\ref{eq:elbo-decomp}.
    """

    reconstruction: float
    kl_population: float
    kl_sample: float
    kl_local: float

    @property
    def total(self) -> float:
        return self.reconstruction - self.kl_population - self.kl_sample - self.kl_local


def reconstruction_term_bb(
    n_meth: NDArray[np.float64],
    n_cpg: NDArray[np.float64],
    eta_samples: NDArray[np.float64],
    kappa: float = KAPPA,
) -> NDArray[np.float64]:
    """Beta-Binomial reconstruction term per (s, l), evaluated at flow samples.

    eq. \\ref{eq:reconstruction}. ``eta_samples`` has shape (S, L) -> beta = sigm(eta).
    """
    beta = sigmoid(eta_samples)
    return reconstruction_log_lik_bb(n_meth, n_cpg, beta, kappa)


def gaussian_local_kl(
    local: GaussianLocalPosterior,
    prior_mean: ArrayLike,
    prior_var: float,
) -> NDArray[np.float64]:
    """Closed-form KL(q(eta_{s,l}) || p(eta_{s,l} | mu^pop, delta)) for the
    Gaussian local posterior of Sec. 11."""
    return gaussian_kl(local.mean, local.var, prior_mean, prior_var)


def montecarlo_local_kl(
    eta_sample: NDArray[np.float64],
    log_q_eta: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_var: float,
) -> NDArray[np.float64]:
    """Single-sample MC estimator of KL(q(eta|F) || p(eta | mu^pop, delta)).

    eq. \\ref{eq:kl-mc}: log q_phi(eta | c) - log N(eta; mu^pop+delta, sigma_eta^2).
    """
    log_p = gaussian_log_pdf(eta_sample, prior_mean, prior_var)
    return log_q_eta - log_p


def compute_elbo_gaussian(
    n_meth: NDArray[np.float64],
    n_cpg: NDArray[np.float64],
    population: PopulationLayer,
    sample_shift: SampleShiftLayer,
    local: GaussianLocalPosterior,
    *,
    sigma_eta: float,
    kappa: float = KAPPA,
    rescale: tuple[float, float] = (1.0, 1.0),
) -> ELBOTerms:
    """ELBO under the simplified-Gaussian posterior of Sec. 11.

    ``rescale`` = (S/|B_s|, L/|B_l|) applies the mini-batch rescaling of
    Table~\\ref{tab:rescale}; pass (1, 1) for full-batch.
    """
    s_factor, l_factor = rescale
    SL_factor = s_factor * l_factor

    eta = local.sample()
    recon = reconstruction_term_bb(n_meth, n_cpg, eta, kappa).sum()
    recon = float(recon * SL_factor)

    prior_mean = population.mean[None, :] + sample_shift.mean[:, None]
    kl_local = gaussian_local_kl(local, prior_mean, sigma_eta**2).sum()
    kl_local = float(kl_local * SL_factor)

    kl_pop = float(population.kl_to_prior().sum() * l_factor)
    kl_sample = float(sample_shift.kl_to_prior().sum() * s_factor)

    return ELBOTerms(
        reconstruction=recon,
        kl_population=kl_pop,
        kl_sample=kl_sample,
        kl_local=kl_local,
    )


def iwae_log_mean_exp(log_w: NDArray[np.float64]) -> NDArray[np.float64]:
    """log (1/K) sum_k exp(log_w_k) via the log-sum-exp trick (Sec. 5.4)."""
    log_w = np.asarray(log_w, dtype=np.float64)
    m = log_w.max(axis=0)
    return m + np.log(np.mean(np.exp(log_w - m), axis=0))


def effective_sample_size(log_w: NDArray[np.float64]) -> NDArray[np.float64]:
    """ESS = (sum w_i)^2 / sum w_i^2, used as a convergence diagnostic (Sec. 6).

    Operates on a stack of importance log-weights with shape (K, ...).
    """
    log_w = np.asarray(log_w, dtype=np.float64)
    log_w_norm = log_w - log_w.max(axis=0, keepdims=True)
    w = np.exp(log_w_norm)
    return (w.sum(axis=0)) ** 2 / (w**2).sum(axis=0)


def kl_anneal_weight(step: int, t_anneal: int) -> float:
    """Linear KL-annealing schedule (Sec. 6 line 7): min(t/T_anneal, 1)."""
    if t_anneal <= 0:
        return 1.0
    return float(min(step / t_anneal, 1.0))
