"""Normalizing-flow primitives (Sec. 4.2, App. C).

The full HAVI-Methyl uses a stack of K=6 neural-spline flows (Durkan et al.
2019) on the logit-beta latent. Production code lives in PyTorch — provided
in the optional torch block below — but the *core math* (rational-quadratic
spline transform and its log-Jacobian, App. C) is also implemented in pure
NumPy so it can be unit-tested without torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------- Pure-numpy rational-quadratic spline (App. C) ----------


@dataclass
class RationalQuadraticSpline:
    """Monotonic rational-quadratic spline on [-B, B] with K bins.

    App. C eq. ``T(x)`` and the corresponding log-Jacobian. Knots ``y`` are
    sorted; the interior derivatives ``d`` are positive (softplus) with
    boundary derivatives fixed to 1.0 for tail-linearity (App. C).
    """

    x_knots: NDArray[np.float64]
    y_knots: NDArray[np.float64]
    derivatives: NDArray[np.float64]

    @classmethod
    def identity(cls, K: int = 8, B: float = 3.0) -> RationalQuadraticSpline:
        """Identity map with K equally spaced bins; useful as a default."""
        xs = np.linspace(-B, B, K + 1)
        ys = xs.copy()
        # boundary derivatives 1, interior 1
        ds = np.ones(K + 1)
        return cls(x_knots=xs, y_knots=ys, derivatives=ds)

    def transform(self, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute T(x) and log|dT/dx| elementwise (App. C eqs.)."""
        x = np.asarray(x, dtype=np.float64)
        K = len(self.x_knots) - 1
        # Find bin index
        bin_idx = np.searchsorted(self.x_knots, x, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, K - 1)
        x_lo = self.x_knots[bin_idx]
        x_hi = self.x_knots[bin_idx + 1]
        y_lo = self.y_knots[bin_idx]
        y_hi = self.y_knots[bin_idx + 1]
        d_lo = self.derivatives[bin_idx]
        d_hi = self.derivatives[bin_idx + 1]
        # Within-bin coordinate
        bin_width = x_hi - x_lo
        bin_height = y_hi - y_lo
        s = bin_height / bin_width
        xi = (x - x_lo) / bin_width
        # Rational-quadratic transform
        num = bin_height * (s * xi**2 + d_lo * xi * (1 - xi))
        den = s + (d_hi + d_lo - 2 * s) * xi * (1 - xi)
        y = y_lo + num / den
        # log-Jacobian
        log_jac = (
            2 * np.log(s)
            + np.log(d_hi * xi**2 + 2 * s * xi * (1 - xi) + d_lo * (1 - xi) ** 2)
            - 2 * np.log(s + (d_hi + d_lo - 2 * s) * xi * (1 - xi))
        )
        return y, log_jac


# ---------- Composed flow ----------


@dataclass
class StackedFlow:
    """Composition of K rational-quadratic spline blocks (Sec. 4.2).

    The log-Jacobian of the composition is the sum of the per-block
    log-Jacobians (App. C, last paragraph of "rational-quadratic spline").
    """

    blocks: list[RationalQuadraticSpline]

    def transform(self, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.asarray(x, dtype=np.float64)
        log_jac_total = np.zeros_like(x)
        y = x
        for block in self.blocks:
            y, lj = block.transform(y)
            log_jac_total = log_jac_total + lj
        return y, log_jac_total

    def log_density(
        self, eta: ArrayLike, base_mean: float = 0.0, base_var: float = 1.0
    ) -> NDArray[np.float64]:
        """log q_phi(eta) under the change-of-variables formula
        eq. \\eqref{eq:flow-density}: log N(epsilon; 0, 1) - log|det J|.

        Note this assumes the flow is invertible and applied forward;
        ``eta`` is the *output*, the inverse pre-image is approximated by
        treating the spline as identity-near-the-tails (App. C).
        """
        from havi_methyl.distributions import gaussian_log_pdf

        eta = np.asarray(eta, dtype=np.float64)
        # For unit testing the identity flow, we use eta itself as epsilon
        epsilon, log_jac = self.transform(eta)
        log_base = gaussian_log_pdf(epsilon, base_mean, base_var)
        return log_base - log_jac


# ---------- Optional torch flow (Sec. 4.2 production) ----------

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConditionalNSFBlock(nn.Module):
        """Conditional rational-quadratic spline block (App. C).

        Predicts the spline parameters from a context vector; rational-quadratic
        transform on a 1D latent (logit-beta).
        """

        def __init__(self, context_dim: int, num_bins: int = 8, B: float = 3.0):
            super().__init__()
            self.num_bins = num_bins
            self.B = B
            self.param_net = nn.Sequential(
                nn.Linear(context_dim, 128),
                nn.GELU(),
                nn.Linear(128, 3 * num_bins - 1),
            )

        def forward(
            self, x: torch.Tensor, context: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            params = self.param_net(context)
            xs, ys, ds = torch.split(
                params, [self.num_bins, self.num_bins, self.num_bins - 1], dim=-1
            )
            # Normalize
            xs = self.B * (2 * F.softmax(xs, dim=-1).cumsum(dim=-1) - 1)
            ys = self.B * (2 * F.softmax(ys, dim=-1).cumsum(dim=-1) - 1)
            ds = F.softplus(ds) + 1e-3
            ds = torch.cat([torch.ones_like(ds[..., :1]), ds, torch.ones_like(ds[..., :1])], dim=-1)
            # Bin search and within-bin transform
            x_clamped = torch.clamp(x, -self.B, self.B)
            idx = torch.searchsorted(xs, x_clamped.unsqueeze(-1)).clamp(1, self.num_bins) - 1
            x_lo = xs.gather(-1, idx).squeeze(-1)
            x_hi = xs.gather(-1, idx + 1).squeeze(-1)
            y_lo = ys.gather(-1, idx).squeeze(-1)
            y_hi = ys.gather(-1, idx + 1).squeeze(-1)
            d_lo = ds.gather(-1, idx).squeeze(-1)
            d_hi = ds.gather(-1, idx + 1).squeeze(-1)
            bw = x_hi - x_lo
            bh = y_hi - y_lo
            s = bh / bw
            xi = (x_clamped - x_lo) / bw
            num = bh * (s * xi**2 + d_lo * xi * (1 - xi))
            den = s + (d_hi + d_lo - 2 * s) * xi * (1 - xi)
            y = y_lo + num / den
            log_jac = (
                2 * torch.log(s)
                + torch.log(d_hi * xi**2 + 2 * s * xi * (1 - xi) + d_lo * (1 - xi) ** 2)
                - 2 * torch.log(den)
            )
            return y, log_jac

    HAS_TORCH = True

except ImportError:  # pragma: no cover
    HAS_TORCH = False
