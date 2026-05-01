"""Lightweight numpy encoders (Sec. 4.3, 4.4).

Production HAVI-Methyl uses a Set Transformer + dilated CNN/HyenaDNA in
PyTorch (App. D). Those modules require torch and are *optional* — see
``flow.py`` for the corresponding flow, and the comments below for how to
swap in the torch implementations.

The numpy encoders here implement the *interface* of the Set Transformer +
sequence encoder: permutation-invariant fragment-bag pooling and a fixed
projection of a sequence-context vector. They are deliberately simple so the
core library is fully testable without heavy ML deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mean_pool_fragment_bag(features: ArrayLike) -> NDArray[np.float64]:
    """Permutation-invariant mean pooling, the bare-minimum Set Transformer
    surrogate (Sec. 4.3 PMA pool with one query reduces to a soft mean).
    """
    f = np.asarray(features, dtype=np.float64)
    if f.size == 0:
        return np.zeros(f.shape[-1] if f.ndim > 1 else 0)
    return f.mean(axis=0)


def sum_pool_fragment_bag(features: ArrayLike) -> NDArray[np.float64]:
    """Permutation-invariant sum pooling. Used as an alternative to mean."""
    f = np.asarray(features, dtype=np.float64)
    return f.sum(axis=0)


@dataclass
class SetMLPEncoder:
    """A tiny set encoder: per-fragment MLP + mean pool + projection.

    A linear approximation of the Set Transformer (Sec. 4.3) sufficient for
    the simplified posterior of Sec. 11. Weights are random by default so
    test cases should pass an explicit RNG.
    """

    W_in: NDArray[np.float64]
    W_out: NDArray[np.float64]
    bias_in: NDArray[np.float64]
    bias_out: NDArray[np.float64]

    @classmethod
    def random(
        cls,
        in_dim: int,
        hidden: int,
        out_dim: int,
        rng: int | np.random.Generator | None = None,
    ) -> SetMLPEncoder:
        from havi_methyl.utils import get_rng

        gen = get_rng(rng)
        scale_in = 1.0 / np.sqrt(in_dim)
        scale_out = 1.0 / np.sqrt(hidden)
        return cls(
            W_in=gen.standard_normal((in_dim, hidden)) * scale_in,
            bias_in=np.zeros(hidden),
            W_out=gen.standard_normal((hidden, out_dim)) * scale_out,
            bias_out=np.zeros(out_dim),
        )

    def encode(self, features: ArrayLike) -> NDArray[np.float64]:
        f = np.asarray(features, dtype=np.float64)
        if f.size == 0:
            return np.zeros(self.W_out.shape[1])
        h = np.maximum(0.0, f @ self.W_in + self.bias_in)  # ReLU
        pooled = h.mean(axis=0)
        return pooled @ self.W_out + self.bias_out


def make_context_vector(
    c_frag: NDArray[np.float64],
    c_seq: NDArray[np.float64],
    pop_mean: float,
    sample_shift_mean: float,
    log1p_coverage: float,
) -> NDArray[np.float64]:
    """Concatenate the encoder context (Sec. 4.4 eq. ``c_{s,l}``).

    Returns ``[c_frag || c_seq || m_l || m_s || log(1 + n_frag)]``.
    """
    return np.concatenate(
        [
            np.asarray(c_frag, dtype=np.float64).flatten(),
            np.asarray(c_seq, dtype=np.float64).flatten(),
            [pop_mean, sample_shift_mean, log1p_coverage],
        ]
    )


# ---------- Optional PyTorch Set Transformer (Sec. 4.3) ----------

try:  # pragma: no cover
    import torch
    import torch.nn as nn

    class ISAB(nn.Module):
        """Induced-set attention block of Lee et al. (Sec. 4.3, App. D).

        Implements the m-inducing-point attention reduction so memory is
        O(n*m) rather than O(n^2). Only available when ``torch`` is installed.
        """

        def __init__(self, dim: int, num_heads: int = 4, num_inducing: int = 64):
            super().__init__()
            self.inducing = nn.Parameter(torch.randn(1, num_inducing, dim))
            self.mab1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.mab2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            inducing = self.inducing.expand(B, -1, -1)
            H, _ = self.mab1(inducing, x, x)
            H = self.ln1(H + inducing)
            X, _ = self.mab2(x, H, H)
            X = self.ln2(X + self.ff(X))
            return X

    class PMA(nn.Module):
        """Pooling by multi-head attention with a single learned seed (Sec. 4.3)."""

        def __init__(self, dim: int, num_heads: int = 4):
            super().__init__()
            self.seed = nn.Parameter(torch.randn(1, 1, dim))
            self.mab = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.ln = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            S = self.seed.expand(B, -1, -1)
            out, _ = self.mab(S, x, x)
            return self.ln(out).squeeze(1)

    class SetTransformerEncoder(nn.Module):
        """Two-layer ISAB stack + PMA pool (Sec. 4.3 / App. D Table 1).

        Replaces ``SetMLPEncoder`` for production runs where torch is
        available.
        """

        def __init__(
            self,
            in_dim: int,
            hidden: int = 128,
            num_layers: int = 2,
            num_inducing: int = 64,
        ):
            super().__init__()
            self.proj_in = nn.Linear(in_dim, hidden)
            self.isabs = nn.ModuleList(
                [ISAB(hidden, num_heads=4, num_inducing=num_inducing) for _ in range(num_layers)]
            )
            self.pma = PMA(hidden, num_heads=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.proj_in(x)
            for isab in self.isabs:
                h = isab(h)
            return self.pma(h)

    HAS_TORCH = True

except ImportError:  # pragma: no cover
    HAS_TORCH = False
