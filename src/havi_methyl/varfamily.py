"""Variational family components (Sec. 4).

The full HAVI-Methyl variational family pairs mean-field Gaussian factors on
the population (eq. \\ref{eq:varfamily}) with an amortized normalizing flow on
the per-(sample, locus) layer. Production training uses PyTorch — see
``encoders.py`` and ``flow.py``. This module provides the *NumPy* analogues
used by the simplified Gaussian-posterior variant of Sec. 11 and by every
unit-testable closed-form expression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from havi_methyl.constants import SIGMA_DELTA, TAU_0
from havi_methyl.distributions import gaussian_kl, gaussian_log_pdf
from havi_methyl.utils import get_rng, sigmoid


@dataclass
class GaussianFactor:
    """Mean-field Gaussian factor q(x) = N(mean, var) (Sec. 4.1)."""

    mean: NDArray[np.float64]
    var: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.var = np.asarray(self.var, dtype=np.float64)
        if self.var.shape != self.mean.shape:
            raise ValueError("mean and var must have matching shapes")
        if np.any(self.var <= 0):
            raise ValueError("variance must be strictly positive")

    @classmethod
    def initialize(cls, shape, mean: float = 0.0, var: float = 1.0) -> GaussianFactor:
        return cls(np.full(shape, mean, dtype=np.float64), np.full(shape, var, dtype=np.float64))

    def log_prob(self, x: ArrayLike) -> NDArray[np.float64]:
        return gaussian_log_pdf(x, self.mean, self.var)

    def sample(self, rng: int | np.random.Generator | None = None) -> NDArray[np.float64]:
        gen = get_rng(rng)
        return self.mean + np.sqrt(self.var) * gen.standard_normal(self.mean.shape)

    def entropy(self) -> NDArray[np.float64]:
        """Differential entropy 0.5 log(2 pi e v)."""
        return 0.5 * (np.log(2 * np.pi * self.var) + 1.0)


@dataclass
class PopulationLayer:
    """Population factor lambda_l = (m_l, v_l), q(mu^pop_l) = N(m_l, v_l).

    Conjugate-exponential-family layer that admits closed-form natural-gradient
    SVI updates (Sec. 4.1, Sec. 5.6).
    """

    mean: NDArray[np.float64]
    var: NDArray[np.float64]
    prior_mean: float = 0.0
    prior_var: float = TAU_0**2

    @classmethod
    def initialize(
        cls,
        L: int,
        prior_mean: float = 0.0,
        prior_var: float = TAU_0**2,
    ) -> PopulationLayer:
        return cls(
            mean=np.full(L, prior_mean, dtype=np.float64),
            var=np.full(L, prior_var, dtype=np.float64),
            prior_mean=prior_mean,
            prior_var=prior_var,
        )

    def kl_to_prior(self) -> NDArray[np.float64]:
        """KL(q || p) closed form (Sec. 5.3)."""
        return gaussian_kl(self.mean, self.var, self.prior_mean, self.prior_var)

    def sample(self, rng: int | np.random.Generator | None = None) -> NDArray[np.float64]:
        gen = get_rng(rng)
        return self.mean + np.sqrt(self.var) * gen.standard_normal(self.mean.shape)


@dataclass
class SampleShiftLayer:
    """Per-sample shift factor q(delta_s) = N(m^delta_s, v^delta_s).

    Sec. 4.1: "the same form is used for the per-sample shift". The sum-to-zero
    constraint (eq. \\ref{eq:sumzero}) is enforced by re-centering after every
    natural-gradient step (Sec. 6.0 algorithm).
    """

    mean: NDArray[np.float64]
    var: NDArray[np.float64]
    prior_var: float = SIGMA_DELTA**2

    @classmethod
    def initialize(cls, S: int, prior_var: float = SIGMA_DELTA**2) -> SampleShiftLayer:
        return cls(
            mean=np.zeros(S, dtype=np.float64),
            var=np.full(S, prior_var, dtype=np.float64),
            prior_var=prior_var,
        )

    def kl_to_prior(self) -> NDArray[np.float64]:
        return gaussian_kl(self.mean, self.var, 0.0, self.prior_var)

    def recenter_(self, mu_pop: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """In-place: enforce sum_s delta_s = 0 by re-centering on the mean.

        Returns the applied shift and the updated ``mu_pop`` (which absorbs it).
        """
        shift = float(self.mean.mean())
        self.mean = self.mean - shift
        return shift, mu_pop + shift


@dataclass
class GaussianLocalPosterior:
    """Diagonal Gaussian posterior on eta_{s,l} (Sec. 4 ablation baseline).

    The full HAVI-Methyl uses a normalizing flow (Sec. 4.2). The Gaussian
    variant is the simplified posterior reported in the synthetic experiments
    of Sec. 11; it is also the closed-form prior-times-likelihood case under
    Gaussian observation noise.
    """

    mean: NDArray[np.float64]
    var: NDArray[np.float64]

    def sample(self, rng: int | np.random.Generator | None = None) -> NDArray[np.float64]:
        gen = get_rng(rng)
        return self.mean + np.sqrt(self.var) * gen.standard_normal(self.mean.shape)

    def kl_to_gaussian_prior(self, prior_mean: ArrayLike, prior_var: float) -> NDArray[np.float64]:
        return gaussian_kl(self.mean, self.var, prior_mean, prior_var)

    def beta_mean_var(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Delta-method approximation: beta = sigm(eta), Var[beta] ~= (sigm'(eta))^2 Var[eta]."""
        beta = sigmoid(self.mean)
        var_beta = (beta * (1.0 - beta)) ** 2 * self.var
        return beta, var_beta


def gaussian_observation_posterior(
    eta_obs: NDArray[np.float64],
    obs_prec: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_prec: float,
) -> GaussianLocalPosterior:
    """Conjugate posterior eta | obs ~ N(post_mean, 1/post_prec).

    Used by the simplified HAVI-Methyl in Sec. 11. Sec. 4.5: posterior =
    (prior_prec * prior_mean + like_prec * obs) / (prior_prec + like_prec).
    """
    post_prec = prior_prec + obs_prec
    post_mean = (prior_prec * prior_mean + obs_prec * eta_obs) / post_prec
    return GaussianLocalPosterior(mean=post_mean, var=1.0 / post_prec)
