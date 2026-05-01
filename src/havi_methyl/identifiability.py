"""Identifiability and de-confounding (Sec. 7).

The HAVI-Methyl identifiability story has three coordinated mechanisms:
  - VIB penalty (Sec. 7.1): bounds I(zeta; X_prior).
  - Counterfactual augmentation (Sec. 7.2): invariance to prior swaps.
  - mQTL anchor loss (Sec. 7.3): IV-style identification of mu^pop.

Each is provided here as a self-contained loss/diagnostic over numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from havi_methyl.constants import BETA_VIB, LAMBDA_CF, LAMBDA_MQTL


def vib_kl_to_unit_gaussian(
    mean: NDArray[np.float64], var: NDArray[np.float64]
) -> NDArray[np.float64]:
    """KL(N(mean, var) || N(0, I)) per row, summed over latent dims.

    Sec. 7.1 chooses r(zeta) = N(0, I) as the marginal proxy. This is the
    closed-form upper bound on I(zeta; X_prior) used in eq. \\ref{eq:vib}.
    """
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    return 0.5 * np.sum(var + mean**2 - 1.0 - np.log(var + 1e-12), axis=-1)


def vib_loss(
    encoder_mean: NDArray[np.float64],
    encoder_var: NDArray[np.float64],
    beta_vib: float = BETA_VIB,
) -> float:
    """Average VIB penalty beta_VIB * E[KL(q(zeta|X_prior) || r)] (Sec. 7.1)."""
    return float(beta_vib * vib_kl_to_unit_gaussian(encoder_mean, encoder_var).mean())


def counterfactual_invariance_loss(
    pred_beta_factual: NDArray[np.float64],
    pred_beta_counterfactual: NDArray[np.float64],
    lam: float = LAMBDA_CF,
) -> float:
    """Sec. 7.2 swap-invariance: lambda_cf * mean ||hat_beta - hat_beta_swap||^2."""
    a = np.asarray(pred_beta_factual, dtype=np.float64)
    b = np.asarray(pred_beta_counterfactual, dtype=np.float64)
    return float(lam * np.mean((a - b) ** 2))


def mqtl_anchor_loss(
    pop_mean: NDArray[np.float64],
    genotype: NDArray[np.float64],
    intercept: NDArray[np.float64],
    effect: NDArray[np.float64],
    anchor_idx: NDArray[np.intp] | None = None,
    lam: float = LAMBDA_MQTL,
) -> float:
    """Sec. 7.3: lambda_mQTL * sum_{l in A} (m_l - a_l - b_l * g_l)^2.

    ``intercept`` and ``effect`` are *fixed* (never co-trained) — see Sec. 7.3
    on the GoDMC catalogue.
    """
    pop_mean = np.asarray(pop_mean, dtype=np.float64)
    genotype = np.asarray(genotype, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    effect = np.asarray(effect, dtype=np.float64)
    if anchor_idx is None:
        anchor_idx = np.arange(len(pop_mean))
    pred = intercept[anchor_idx] + effect[anchor_idx] * genotype[anchor_idx]
    resid = pop_mean[anchor_idx] - pred
    return float(lam * np.sum(resid**2))


# ---------- Diagnostics ----------


def prior_attribution_partial_r2(
    pred_beta: NDArray[np.float64],
    true_beta: NDArray[np.float64],
    prior_input: NDArray[np.float64],
) -> float:
    """Partial R^2 of the buffy-coat prior in a regression of pred onto (prior, true).

    Sec. 11.4: defined as the share of explained variance attributable to the
    prior input above and beyond what ``true_beta`` already explains. Lower
    is better (less leakage). Reference values: 3.32% / 0.36% / 0.022% for
    no-VIB / VIB / VIB+mQTL respectively.
    """
    pred = np.asarray(pred_beta, dtype=np.float64).flatten()
    true = np.asarray(true_beta, dtype=np.float64).flatten()
    prior = np.asarray(prior_input, dtype=np.float64).flatten()
    if prior.size != pred.size:
        # Broadcast a per-locus prior across samples
        n = pred.size // prior.size
        prior = np.tile(prior, n)
    X = np.stack([prior, true], axis=1)
    # full R^2
    beta, *_ = np.linalg.lstsq(X, pred, rcond=None)
    yhat = X @ beta
    ss_res = float(((pred - yhat) ** 2).sum())
    ss_tot = float(((pred - pred.mean()) ** 2).sum())
    r2_full = 1.0 - ss_res / max(ss_tot, 1e-12)
    # R^2 of true-beta alone
    r2_true = float(np.corrcoef(true, pred)[0, 1] ** 2)
    return max(0.0, r2_full - r2_true) / max(r2_full, 1e-6)


@dataclass
class IdentifiabilityResults:
    """Bundle of leakage diagnostics across regularization regimes."""

    leak_no_vib: float
    leak_vib_only: float
    leak_vib_plus_mqtl: float

    def as_dict(self) -> dict[str, float]:
        return {
            "leak_no_vib": self.leak_no_vib,
            "leak_vib_only": self.leak_vib_only,
            "leak_vib_plus_mqtl": self.leak_vib_plus_mqtl,
        }


def vib_finite_leakage_bound(elbo_max: float, elbo_vib: float, beta_vib: float) -> float:
    """Corollary 1: eta_leak <= G(beta_VIB) / beta_VIB.

    Sec. 13 gives the corollary; ``elbo_max - elbo_vib >= 0`` is the gap of
    Corollary~\\ref{cor:vib-finite}.
    """
    if beta_vib <= 0:
        return float("inf")
    return float(max(0.0, elbo_max - elbo_vib) / beta_vib)
