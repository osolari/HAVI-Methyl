"""Stochastic variational inference loop (Sec. 6, Algorithm 1).

Implements the *simplified* HAVI-Methyl variant that replaces the Set
Transformer + flow encoder with a Gaussian observation-noise model and
empirical-Bayes hierarchical updates. This is the variant whose numbers
appear in Sec. 11 and ``docs/report/results.json``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from havi_methyl.constants import RHO_EXPONENT, SIGMA_DELTA, SIGMA_ETA, TAU_0
from havi_methyl.model import enforce_sum_zero_constraint
from havi_methyl.utils import safe_logit, sigmoid
from havi_methyl.varfamily import (
    GaussianLocalPosterior,
    PopulationLayer,
    SampleShiftLayer,
)


def robbins_monro_step(t: int, exponent: float = RHO_EXPONENT) -> float:
    """Step size rho_t = (t+1)^{-exponent} (Sec. 6.5, eq. \\ref{eq:nat-update}).

    Satisfies sum rho_t = inf, sum rho_t^2 < inf for ``exponent in (0.5, 1)``.
    """
    return float((t + 1) ** (-exponent))


def logit_observation_precision(
    pred_beta: NDArray[np.float64], n_frag: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Wald-style logit-scale observation precision proportional to coverage.

    Var[logit(p_hat)] ~= 1/(n p (1-p)), so precision = n p (1-p) (modulo a
    scale factor that we absorb into the user-tunable obs precision constant).
    """
    bp = np.clip(pred_beta, 1e-2, 1 - 1e-2)
    return (n_frag + 1.0) * bp * (1.0 - bp) * 4.0


def empirical_bayes_update_population(
    population: PopulationLayer,
    sample_shift: SampleShiftLayer,
    eta_obs: NDArray[np.float64],
    obs_prec: NDArray[np.float64],
    rho: float = 1.0,
) -> PopulationLayer:
    """Approximate non-conjugate natural-gradient population update.

    Eq. \\ref{eq:nat-target}: moment-match the encoder posterior means as if
    they were direct Gaussian observations of mu^pop. Eq. \\ref{eq:nat-update}
    blends the new natural parameter with the previous via the Robbins-Monro
    weight ``rho``.
    """
    S, L = eta_obs.shape
    weighted = obs_prec * (eta_obs - sample_shift.mean[:, None])
    num = weighted.sum(axis=0)
    den = obs_prec.sum(axis=0)
    prior_prec = 1.0 / population.prior_var
    new_mean = (num + prior_prec * population.prior_mean) / (den + prior_prec)
    new_var = 1.0 / (den + prior_prec)
    blended_mean = (1.0 - rho) * population.mean + rho * new_mean
    blended_var = (1.0 - rho) * population.var + rho * new_var
    return PopulationLayer(
        mean=blended_mean,
        var=blended_var,
        prior_mean=population.prior_mean,
        prior_var=population.prior_var,
    )


def empirical_bayes_update_sample_shift(
    population: PopulationLayer,
    sample_shift: SampleShiftLayer,
    eta_obs: NDArray[np.float64],
    obs_prec: NDArray[np.float64],
    rho: float = 1.0,
) -> SampleShiftLayer:
    """Conjugate Gaussian-Gaussian update for delta_s (Sec. 6 line 9)."""
    weighted = obs_prec * (eta_obs - population.mean[None, :])
    num = weighted.sum(axis=1)
    den = obs_prec.sum(axis=1)
    prior_prec = 1.0 / sample_shift.prior_var
    new_mean = (num + prior_prec * 0.0) / (den + prior_prec)
    new_var = 1.0 / (den + prior_prec)
    blended_mean = (1.0 - rho) * sample_shift.mean + rho * new_mean
    blended_var = (1.0 - rho) * sample_shift.var + rho * new_var
    return SampleShiftLayer(mean=blended_mean, var=blended_var, prior_var=sample_shift.prior_var)


def local_posterior_gaussian(
    eta_obs: NDArray[np.float64],
    obs_prec: NDArray[np.float64],
    population: PopulationLayer,
    sample_shift: SampleShiftLayer,
    sigma_eta: float = SIGMA_ETA,
) -> GaussianLocalPosterior:
    """Conjugate posterior on eta_{s,l} given the current variational means
    of the population and per-sample shift layers (Sec. 4.5 / Sec. 11)."""
    prior_mean = population.mean[None, :] + sample_shift.mean[:, None]
    prior_prec = 1.0 / sigma_eta**2
    post_prec = prior_prec + obs_prec
    post_mean = (prior_prec * prior_mean + obs_prec * eta_obs) / post_prec
    return GaussianLocalPosterior(mean=post_mean, var=1.0 / post_prec)


@dataclass
class SVIState:
    """Bundle of variational parameters and iteration history."""

    population: PopulationLayer
    sample_shift: SampleShiftLayer
    elbo_history: list[float] = field(default_factory=list)
    sse_history: list[float] = field(default_factory=list)


def fit_svi_simplified(
    pred_beta: NDArray[np.float64],
    n_frag: NDArray[np.float64],
    n_iter: int = 10,
    sigma_pop: float = TAU_0,
    sigma_delta: float = SIGMA_DELTA,
    sigma_eta: float = SIGMA_ETA,
    enforce_sum_zero: bool = True,
    callback: Callable[[SVIState, int], None] | None = None,
) -> SVIState:
    """Run the simplified SVI loop of Sec. 11 / ``run_experiments.py``.

    Each iteration performs:
      1. Population mean update conditional on current sample shifts.
      2. Sample shift update conditional on the new population.
      3. Sum-to-zero re-centering (eq. \\ref{eq:sumzero}).
      4. Population variance update from empirical residuals (with floor 0.05).
      5. ELBO/SSE diagnostic logging (Sec. 6 convergence diagnostics).
    """
    pred_beta = np.asarray(pred_beta, dtype=np.float64)
    n_frag = np.asarray(n_frag, dtype=np.float64)
    S, L = pred_beta.shape
    eta_obs = safe_logit(np.clip(pred_beta, 1e-2, 1 - 1e-2))
    obs_prec = logit_observation_precision(pred_beta, n_frag)

    state = SVIState(
        population=PopulationLayer.initialize(L, prior_mean=0.0, prior_var=sigma_pop**2),
        sample_shift=SampleShiftLayer.initialize(S, prior_var=sigma_delta**2),
    )
    for t in range(n_iter):
        state.population = empirical_bayes_update_population(
            state.population, state.sample_shift, eta_obs, obs_prec, rho=1.0
        )
        state.sample_shift = empirical_bayes_update_sample_shift(
            state.population, state.sample_shift, eta_obs, obs_prec, rho=1.0
        )
        if enforce_sum_zero:
            new_mu, new_delta = enforce_sum_zero_constraint(
                state.population.mean, state.sample_shift.mean
            )
            state.population.mean = new_mu
            state.sample_shift.mean = new_delta
        # update population variance from empirical residual (Sec. 11 SVI variant)
        prior_mean_grid = state.population.mean[None, :] + state.sample_shift.mean[:, None]
        resid = ((eta_obs - prior_mean_grid) ** 2).sum(axis=0) / max(S, 1)
        state.population.var = np.maximum(resid, 0.05)
        # ELBO surrogate: negative weighted SSE per pair (Sec. 11.6 trajectory plot)
        sse = float((obs_prec * (eta_obs - prior_mean_grid) ** 2).sum())
        state.sse_history.append(sse)
        state.elbo_history.append(-sse / max(S * L, 1))
        if callback is not None:
            callback(state, t)
    return state


def fit_svi_full(
    pred_beta: NDArray[np.float64],
    n_frag: NDArray[np.float64],
    n_iter: int = 30,
    batch_samples: int | None = None,
    batch_loci: int | None = None,
    rho_exponent: float = RHO_EXPONENT,
    sigma_pop: float = TAU_0,
    sigma_delta: float = SIGMA_DELTA,
    sigma_eta: float = SIGMA_ETA,
    rng: int | np.random.Generator | None = None,
) -> SVIState:
    """Mini-batch SVI with Robbins-Monro natural-gradient step sizes (Sec. 6).

    Differs from ``fit_svi_simplified`` in three ways:
      1. Sub-samples ``B_s`` samples and ``B_l`` loci uniformly at each step
         (Sec. 5.6 plate rescaling).
      2. Uses a Robbins-Monro step ``rho_t = (t+1)^{-rho_exponent}`` for the
         population/sample-shift updates rather than full overwrites; this
         is the surrogate-natural-gradient step of eq.~\\ref{eq:nat-update}.
      3. Tracks a global recentering residual every iteration so the
         sum-to-zero constraint is enforced *globally* rather than within a
         mini-batch (App. A correction A2).

    Returns the same ``SVIState`` as the simplified loop. Numbers from this
    routine are **not** the canonical Sec. 11 numbers — those still come from
    ``fit_svi_simplified``. This is the IMPL-05 path that future real-data
    runs should adopt.
    """
    from havi_methyl.utils import get_rng

    pred_beta = np.asarray(pred_beta, dtype=np.float64)
    n_frag = np.asarray(n_frag, dtype=np.float64)
    S, L = pred_beta.shape
    bs = S if batch_samples is None else min(batch_samples, S)
    bl = L if batch_loci is None else min(batch_loci, L)
    eta_obs_full = safe_logit(np.clip(pred_beta, 1e-2, 1 - 1e-2))
    obs_prec_full = logit_observation_precision(pred_beta, n_frag)
    gen = get_rng(rng)

    state = SVIState(
        population=PopulationLayer.initialize(L, prior_mean=0.0, prior_var=sigma_pop**2),
        sample_shift=SampleShiftLayer.initialize(S, prior_var=sigma_delta**2),
    )
    for t in range(n_iter):
        sample_idx = gen.choice(S, size=bs, replace=False)
        loci_idx = gen.choice(L, size=bl, replace=False)
        eta_batch = eta_obs_full[np.ix_(sample_idx, loci_idx)]
        prec_batch = obs_prec_full[np.ix_(sample_idx, loci_idx)]
        rho = robbins_monro_step(t, exponent=rho_exponent)

        # Population update on the locus mini-batch.
        weighted = prec_batch * (eta_batch - state.sample_shift.mean[sample_idx][:, None])
        num = weighted.sum(axis=0)
        den = prec_batch.sum(axis=0)
        prior_prec = 1.0 / state.population.prior_var
        new_mean_l = (num + prior_prec * state.population.prior_mean) / (den + prior_prec)
        new_var_l = 1.0 / (den + prior_prec)
        state.population.mean[loci_idx] = (1.0 - rho) * state.population.mean[
            loci_idx
        ] + rho * new_mean_l
        state.population.var[loci_idx] = (1.0 - rho) * state.population.var[
            loci_idx
        ] + rho * new_var_l

        # Sample shift update on the sample mini-batch.
        weighted_s = prec_batch * (eta_batch - state.population.mean[loci_idx][None, :])
        num_s = weighted_s.sum(axis=1)
        den_s = prec_batch.sum(axis=1)
        prior_prec_s = 1.0 / state.sample_shift.prior_var
        new_mean_s = num_s / (den_s + prior_prec_s)
        new_var_s = 1.0 / (den_s + prior_prec_s)
        state.sample_shift.mean[sample_idx] = (1.0 - rho) * state.sample_shift.mean[
            sample_idx
        ] + rho * new_mean_s
        state.sample_shift.var[sample_idx] = (1.0 - rho) * state.sample_shift.var[
            sample_idx
        ] + rho * new_var_s
        # Global recentering: enforce sum-zero across all samples (App. A A2).
        new_mu, new_delta = enforce_sum_zero_constraint(
            state.population.mean, state.sample_shift.mean
        )
        state.population.mean = new_mu
        state.sample_shift.mean = new_delta
        # Surrogate ELBO logging on the full set.
        prior_grid = state.population.mean[None, :] + state.sample_shift.mean[:, None]
        sse = float((obs_prec_full * (eta_obs_full - prior_grid) ** 2).sum())
        state.sse_history.append(sse)
        state.elbo_history.append(-sse / max(S * L, 1))
    return state


def recentering_residual(state: SVIState) -> float:
    """``sum_s delta_s`` — should be near zero after global recentering."""
    return float(state.sample_shift.mean.sum())


def predict_with_state(
    state: SVIState,
    pred_beta: NDArray[np.float64],
    n_frag: NDArray[np.float64],
    sigma_eta: float = SIGMA_ETA,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Posterior mean and std of beta = sigm(eta) (Sec. 11.1).

    Combines the prior (pop + delta) with the per-(s,l) observation evidence
    using a Gaussian-Gaussian conjugate update; transforms variance to beta
    space via the delta method (sigm'(eta))^2.
    """
    pred_beta = np.asarray(pred_beta, dtype=np.float64)
    n_frag = np.asarray(n_frag, dtype=np.float64)
    eta_obs = safe_logit(np.clip(pred_beta, 1e-2, 1 - 1e-2))
    obs_prec = logit_observation_precision(pred_beta, n_frag)
    local = local_posterior_gaussian(
        eta_obs, obs_prec, state.population, state.sample_shift, sigma_eta=sigma_eta
    )
    beta_mean = sigmoid(local.mean)
    beta_std = np.sqrt((beta_mean * (1.0 - beta_mean)) ** 2 * local.var)
    return beta_mean, beta_std
