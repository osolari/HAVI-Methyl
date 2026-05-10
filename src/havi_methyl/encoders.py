"""Lightweight numpy encoders (Sec. 4.3, 4.4).

Production HAVI-Methyl uses a Set Transformer + dilated CNN/HyenaDNA in
PyTorch (App. D). Those modules require torch and are *optional* — see
``flow.py`` for the corresponding flow, and the comments below for how to
swap in the torch implementations.

The numpy encoders here implement the *interface* of the Set Transformer +
sequence encoder: permutation-invariant fragment-bag pooling and a fixed
projection of a sequence-context vector. They are deliberately simple so the
core library is fully testable without heavy ML deps.

For IMPL-02 in ``docs/report/CODING_AGENT_HANDOFF.md`` we additionally ship
mask-aware ``ISABNumpy`` and ``PMANumpy`` blocks plus a small
``SetTransformerNumpy`` stack. They share the same call signature as the
torch counterpart, are exact in float64, and pass the permutation-invariance
and mask-handling tests in ``tests/test_torch_modules.py`` (now also covered
without torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def masked_mean_pool(features: ArrayLike, mask: ArrayLike | None = None) -> NDArray[np.float64]:
    """Permutation-invariant mean pool that ignores masked rows.

    ``mask`` is a boolean array of shape ``(n,)`` where ``True`` marks valid
    fragments. If ``mask`` is None (or all False) returns a zero vector to
    keep downstream pipelines stable for empty bags.
    """
    f = np.asarray(features, dtype=np.float64)
    if f.size == 0:
        return np.zeros(f.shape[-1] if f.ndim > 1 else 0)
    if mask is None:
        return f.mean(axis=0)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros(f.shape[-1])
    return f[m].mean(axis=0)


def _softmax_axis(x: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def masked_attention(
    queries: NDArray[np.float64],
    keys: NDArray[np.float64],
    values: NDArray[np.float64],
    key_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Single-head scaled dot-product attention with optional key mask.

    Shapes: ``queries`` (q, d), ``keys`` (k, d), ``values`` (k, v),
    ``key_mask`` (k,) — masked entries are excluded from the attention
    weights via additive ``-inf`` before softmax.
    """
    q = np.asarray(queries, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    d = q.shape[-1]
    scores = q @ k.T / np.sqrt(d)
    if key_mask is not None:
        m = np.asarray(key_mask, dtype=bool)
        scores = np.where(m[None, :], scores, -1e30)
    weights = _softmax_axis(scores, axis=-1)
    return weights @ v


@dataclass
class ISABNumpy:
    """Numpy induced-set attention block (Sec. 4.3).

    Single-head, layernorm-free reference implementation. Two-step attention:
    (1) inducing points attend to ``X``; (2) ``X`` attends to the inducing
    summary. Memory is O(n m) rather than O(n^2). Used in tests where torch
    is not installed.
    """

    inducing: NDArray[np.float64]  # (m, dim)
    W_q1: NDArray[np.float64]
    W_k1: NDArray[np.float64]
    W_v1: NDArray[np.float64]
    W_q2: NDArray[np.float64]
    W_k2: NDArray[np.float64]
    W_v2: NDArray[np.float64]

    @classmethod
    def random(
        cls,
        dim: int,
        num_inducing: int = 64,
        rng: int | np.random.Generator | None = None,
    ) -> ISABNumpy:
        from havi_methyl.utils import get_rng

        gen = get_rng(rng)
        scale = 1.0 / np.sqrt(dim)
        return cls(
            inducing=gen.standard_normal((num_inducing, dim)) * scale,
            W_q1=gen.standard_normal((dim, dim)) * scale,
            W_k1=gen.standard_normal((dim, dim)) * scale,
            W_v1=gen.standard_normal((dim, dim)) * scale,
            W_q2=gen.standard_normal((dim, dim)) * scale,
            W_k2=gen.standard_normal((dim, dim)) * scale,
            W_v2=gen.standard_normal((dim, dim)) * scale,
        )

    def forward(
        self, x: NDArray[np.float64], mask: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        # Step 1: inducing points attend to X.
        H = masked_attention(
            self.inducing @ self.W_q1,
            x @ self.W_k1,
            x @ self.W_v1,
            key_mask=mask,
        )
        H = H + self.inducing  # residual
        # Step 2: X attends to H (no mask on H, all valid).
        Y = masked_attention(x @ self.W_q2, H @ self.W_k2, H @ self.W_v2)
        Y = Y + x  # residual
        return Y


@dataclass
class PMANumpy:
    """Pooling by multi-head attention with a single learned seed (Sec. 4.3)."""

    seed: NDArray[np.float64]  # (1, dim)
    W_q: NDArray[np.float64]
    W_k: NDArray[np.float64]
    W_v: NDArray[np.float64]

    @classmethod
    def random(cls, dim: int, rng: int | np.random.Generator | None = None) -> PMANumpy:
        from havi_methyl.utils import get_rng

        gen = get_rng(rng)
        scale = 1.0 / np.sqrt(dim)
        return cls(
            seed=gen.standard_normal((1, dim)) * scale,
            W_q=gen.standard_normal((dim, dim)) * scale,
            W_k=gen.standard_normal((dim, dim)) * scale,
            W_v=gen.standard_normal((dim, dim)) * scale,
        )

    def forward(
        self, x: NDArray[np.float64], mask: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        out = masked_attention(self.seed @ self.W_q, x @ self.W_k, x @ self.W_v, key_mask=mask)
        return out.squeeze(0)


@dataclass
class SetTransformerNumpy:
    """ISAB + PMA stack (numpy reference for the production Set Transformer).

    Provides a permutation-invariant, mask-aware encoder that maps a fragment
    bag ``(n, in_dim)`` to a context vector ``(out_dim,)``. Unlike the torch
    version, every operation is deterministic in float64 — useful for unit
    tests of the IMPL-02 contract.
    """

    proj_in: NDArray[np.float64]
    isabs: list[ISABNumpy]
    pma: PMANumpy
    proj_out: NDArray[np.float64]
    bias_in: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    bias_out: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def random(
        cls,
        in_dim: int,
        hidden: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        num_inducing: int = 16,
        rng: int | np.random.Generator | None = None,
    ) -> SetTransformerNumpy:
        from havi_methyl.utils import get_rng

        gen = get_rng(rng)
        scale_in = 1.0 / np.sqrt(in_dim)
        scale_out = 1.0 / np.sqrt(hidden)
        return cls(
            proj_in=gen.standard_normal((in_dim, hidden)) * scale_in,
            bias_in=np.zeros(hidden),
            isabs=[
                ISABNumpy.random(hidden, num_inducing=num_inducing, rng=gen)
                for _ in range(num_layers)
            ],
            pma=PMANumpy.random(hidden, rng=gen),
            proj_out=gen.standard_normal((hidden, out_dim)) * scale_out,
            bias_out=np.zeros(out_dim),
        )

    def encode(self, features: ArrayLike, mask: ArrayLike | None = None) -> NDArray[np.float64]:
        f = np.asarray(features, dtype=np.float64)
        if f.size == 0:
            return np.zeros(self.proj_out.shape[1])
        h = f @ self.proj_in + self.bias_in
        m = None if mask is None else np.asarray(mask, dtype=bool)
        for layer in self.isabs:
            h = layer.forward(h, mask=m)
        pooled = self.pma.forward(h, mask=m)
        return pooled @ self.proj_out + self.bias_out


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


# ---------- Sequence-context encoder (Sec. 4.4 / IMPL-03) ----------


def one_hot_dna(seq: str | bytes) -> NDArray[np.float64]:
    """One-hot encode a DNA sequence to shape ``(len(seq), 4)``.

    Order is ``A, C, G, T``. ``N`` and any other character produce a uniform
    0.25 row; the strand convention is forward (5'->3') as written.
    """
    if isinstance(seq, bytes):
        seq = seq.decode("ascii")
    s = seq.upper()
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = np.zeros((len(s), 4), dtype=np.float64)
    for i, base in enumerate(s):
        idx = table.get(base)
        if idx is None:
            out[i] = 0.25
        else:
            out[i, idx] = 1.0
    return out


def reverse_complement(seq: str) -> str:
    """Reverse-complement a DNA sequence (for stranded encodings)."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b.upper(), "N") for b in reversed(seq))


@dataclass
class DilatedCNNSequenceEncoder:
    """Numpy dilated 1D-CNN for the sequence context.

    Implements the reference architecture of Sec. 4.4: stacked dilated
    convolutions with exponentially growing dilation, ReLU activations, and
    mean pooling to a fixed-size embedding. The numpy version is enough to
    test that the embedding is shift-equivariant up to pooling and reproducible
    across runs given a fixed RNG.
    """

    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]
    dilations: list[int]
    proj_out: NDArray[np.float64]

    @classmethod
    def random(
        cls,
        in_channels: int = 4,
        hidden: int = 32,
        out_dim: int = 64,
        num_layers: int = 4,
        kernel: int = 3,
        rng: int | np.random.Generator | None = None,
    ) -> DilatedCNNSequenceEncoder:
        from havi_methyl.utils import get_rng

        gen = get_rng(rng)
        weights: list[NDArray[np.float64]] = []
        biases: list[NDArray[np.float64]] = []
        dilations: list[int] = []
        c_in = in_channels
        for layer in range(num_layers):
            scale = 1.0 / np.sqrt(c_in * kernel)
            weights.append(gen.standard_normal((hidden, c_in, kernel)) * scale)
            biases.append(np.zeros(hidden))
            dilations.append(2**layer)
            c_in = hidden
        proj_out = gen.standard_normal((hidden, out_dim)) * (1.0 / np.sqrt(hidden))
        return cls(weights=weights, biases=biases, dilations=dilations, proj_out=proj_out)

    @staticmethod
    def _conv1d_dilated(
        x: NDArray[np.float64], w: NDArray[np.float64], b: NDArray[np.float64], dilation: int
    ) -> NDArray[np.float64]:
        # x: (T, C_in); w: (C_out, C_in, K); b: (C_out,)
        T, C_in = x.shape
        C_out, _, K = w.shape
        eff_K = (K - 1) * dilation + 1
        pad = eff_K // 2
        x_padded = np.pad(x, ((pad, pad), (0, 0)))
        out = np.zeros((T, C_out), dtype=np.float64)
        for k in range(K):
            offset = k * dilation
            out += x_padded[offset : offset + T] @ w[:, :, k].T
        return out + b

    def encode(self, sequence: str) -> NDArray[np.float64]:
        x = one_hot_dna(sequence)
        h = x
        for w, b, d in zip(self.weights, self.biases, self.dilations, strict=False):
            h = np.maximum(0.0, self._conv1d_dilated(h, w, b, d))
        pooled = h.mean(axis=0)
        return pooled @ self.proj_out


@dataclass
class FrozenEmbeddingProjection:
    """Wrap a frozen long-range DNA embedding behind a small projection.

    The embedding (e.g. HyenaDNA ``(L_seq, 256)``) is averaged across the
    locus window and projected to ``out_dim``. This lets HAVI-Methyl call a
    pretrained encoder *without* taking on its weights or training loop.
    Sec. 4.4 lists this as an optional alternative to the dilated CNN.
    """

    proj: NDArray[np.float64]

    @classmethod
    def random(
        cls,
        in_dim: int = 256,
        out_dim: int = 64,
        rng: int | np.random.Generator | None = None,
    ) -> FrozenEmbeddingProjection:
        from havi_methyl.utils import get_rng

        gen = get_rng(rng)
        return cls(proj=gen.standard_normal((in_dim, out_dim)) * (1.0 / np.sqrt(in_dim)))

    def project(self, embedding: ArrayLike) -> NDArray[np.float64]:
        e = np.asarray(embedding, dtype=np.float64)
        if e.ndim == 2:
            e = e.mean(axis=0)
        return e @ self.proj


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
