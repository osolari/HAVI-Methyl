"""Hierarchical generative model (Sec. 3, eqs. \\eqref{eq:pop}-\\eqref{eq:cell}).

This module describes the joint p(mu^pop, delta, eta, F) and provides
``sample_prior`` for drawing synthetic latents and ``log_joint`` for computing
the prior contribution to ELBO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from havi_methyl.constants import KAPPA, MU_0, SIGMA_DELTA, SIGMA_ETA, TAU_0
from havi_methyl.distributions import gaussian_log_pdf
from havi_methyl.utils import get_rng, sigmoid


@dataclass(frozen=True)
class HierarchicalModel:
    """Three-level hierarchical Bayesian model on logit-beta (Sec. 3.2).

    Parameters
    ----------
    mu_0, tau_0
        Population prior mean and std on logit scale: mu^pop_l ~ N(mu_0, tau_0^2).
    sigma_delta
        Per-sample shift std: delta_s ~ N(0, sigma_delta^2).
    sigma_eta
        Per-(sample, locus) latent std: eta_{s,l} ~ N(mu^pop_l + delta_s, sigma_eta^2).
    kappa
        Beta-Binomial concentration in the aggregate likelihood eq. \\ref{eq:bb}.
    """

    mu_0: float = MU_0
    tau_0: float = TAU_0
    sigma_delta: float = SIGMA_DELTA
    sigma_eta: float = SIGMA_ETA
    kappa: float = KAPPA

    def sample_prior(
        self,
        S: int,
        L: int,
        rng: int | np.random.Generator | None = None,
        enforce_sum_zero: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """Draw mu^pop, delta, eta from the prior (eqs. \\eqref{eq:pop}-\\eqref{eq:cell})."""
        gen = get_rng(rng)
        mu_pop = gen.normal(self.mu_0, self.tau_0, size=L)
        delta = gen.normal(0.0, self.sigma_delta, size=S)
        if enforce_sum_zero and S > 1:
            # eq. \ref{eq:sumzero}: re-center per-sample shifts
            shift = delta.mean()
            delta = delta - shift
            mu_pop = mu_pop + shift
        eta = mu_pop[None, :] + delta[:, None] + gen.normal(0.0, self.sigma_eta, size=(S, L))
        return {
            "mu_pop": mu_pop,
            "delta": delta,
            "eta": eta,
            "beta_pop": sigmoid(mu_pop),
            "beta_sample": sigmoid(eta),
        }

    def log_prior(
        self,
        mu_pop: ArrayLike,
        delta: ArrayLike,
        eta: ArrayLike,
    ) -> float:
        """Sum of the three prior log-densities (eqs. \\eqref{eq:pop}-\\eqref{eq:cell})."""
        mu_pop = np.asarray(mu_pop, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)
        eta = np.asarray(eta, dtype=np.float64)
        lp_pop = gaussian_log_pdf(mu_pop, self.mu_0, self.tau_0**2).sum()
        lp_delta = gaussian_log_pdf(delta, 0.0, self.sigma_delta**2).sum()
        # eta | mu_pop, delta ~ N(mu_pop + delta, sigma_eta^2)
        prior_mean = mu_pop[None, :] + delta[:, None]
        lp_eta = gaussian_log_pdf(eta, prior_mean, self.sigma_eta**2).sum()
        return float(lp_pop + lp_delta + lp_eta)

    def conditional_eta_mean(self, mu_pop: ArrayLike, delta: ArrayLike) -> NDArray[np.float64]:
        """E[eta_{s,l} | mu^pop, delta] = mu^pop_l + delta_s (Sec. 3.2)."""
        mu_pop = np.asarray(mu_pop, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)
        return mu_pop[None, :] + delta[:, None]


def enforce_sum_zero_constraint(
    mu_pop: NDArray[np.float64], delta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Enforce ``sum_s delta_s = 0`` by re-centering (Sec. 3.2 eq. \\ref{eq:sumzero}).

    Adds the mean shift to the population latent so eta_{s,l} is invariant.
    """
    shift = float(delta.mean())
    return mu_pop + shift, delta - shift
