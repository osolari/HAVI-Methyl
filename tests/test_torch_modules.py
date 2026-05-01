"""Tests for the optional PyTorch encoder + flow modules (Sec. 4.2-4.3, App. C-D).

Skip cleanly when torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from havi_methyl.encoders import HAS_TORCH  # noqa: E402

if not HAS_TORCH:  # pragma: no cover
    pytest.skip("torch encoders not available", allow_module_level=True)

from havi_methyl.encoders import ISAB, PMA, SetTransformerEncoder  # noqa: E402
from havi_methyl.flow import ConditionalNSFBlock  # noqa: E402


@pytest.fixture
def torch_rng():
    return torch.Generator().manual_seed(20260429)


# ---------- Set Transformer (Sec. 4.3, App. D) ----------


def test_isab_output_shape(torch_rng):
    """ISAB preserves the (B, n, dim) shape of the input fragment bag."""
    B, n, dim = 2, 17, 32
    x = torch.randn(B, n, dim, generator=torch_rng)
    isab = ISAB(dim=dim, num_heads=4, num_inducing=8)
    out = isab(x)
    assert out.shape == (B, n, dim)
    assert torch.isfinite(out).all()


def test_pma_pools_to_one_query(torch_rng):
    """PMA(k=1) reduces (B, n, dim) -> (B, dim)."""
    B, n, dim = 3, 11, 16
    x = torch.randn(B, n, dim, generator=torch_rng)
    pma = PMA(dim=dim, num_heads=4)
    out = pma(x)
    assert out.shape == (B, dim)


def test_set_transformer_permutation_invariance(torch_rng):
    """Sec. 4.3: pooled output is invariant to fragment ordering."""
    B, n, in_dim = 1, 12, 7
    x = torch.randn(B, n, in_dim, generator=torch_rng)
    perm = torch.randperm(n, generator=torch_rng)
    encoder = SetTransformerEncoder(in_dim=in_dim, hidden=24, num_layers=2, num_inducing=8)
    encoder.eval()
    with torch.no_grad():
        out_a = encoder(x)
        out_b = encoder(x[:, perm, :])
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_set_transformer_param_count_reasonable():
    """App. D: ~200K params at d_c=128, L_e=2 ISAB layers, m=64 inducing."""
    encoder = SetTransformerEncoder(in_dim=7, hidden=128, num_layers=2, num_inducing=64)
    n_params = sum(p.numel() for p in encoder.parameters())
    # Sanity: same order of magnitude (100K-1M) as the App. D estimate
    assert 50_000 < n_params < 5_000_000


# ---------- Conditional NSF block (Sec. 4.2, App. C) ----------


def test_nsf_block_forward_shape(torch_rng):
    """Conditional NSF block returns (y, log|J|) of the same shape as the input."""
    B = 5
    ctx_dim = 16
    x = torch.randn(B, generator=torch_rng) * 1.5
    ctx = torch.randn(B, ctx_dim, generator=torch_rng)
    block = ConditionalNSFBlock(context_dim=ctx_dim, num_bins=8)
    y, log_jac = block(x, ctx)
    assert y.shape == x.shape
    assert log_jac.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(log_jac).all()


def test_nsf_block_jacobian_matches_finite_difference(torch_rng):
    """Empirical d/dx T(x) should match exp(log_jac) (App. C)."""
    ctx_dim = 8
    block = ConditionalNSFBlock(context_dim=ctx_dim, num_bins=8)
    block.eval()
    eps = 1e-3
    B = 6
    x = torch.linspace(-2.0, 2.0, B)
    ctx = torch.zeros(B, ctx_dim)  # constant context across batch
    with torch.no_grad():
        y, log_jac = block(x, ctx)
        y_plus, _ = block(x + eps, ctx)
        y_minus, _ = block(x - eps, ctx)
    fd_jac = (y_plus - y_minus) / (2 * eps)
    np.testing.assert_allclose(torch.exp(log_jac).numpy(), fd_jac.numpy(), rtol=1e-2, atol=1e-2)


def test_nsf_block_monotonic(torch_rng):
    """Rational-quadratic spline is monotonic in x for fixed context (App. C)."""
    ctx_dim = 4
    block = ConditionalNSFBlock(context_dim=ctx_dim, num_bins=8)
    block.eval()
    x = torch.linspace(-2.5, 2.5, 50)
    ctx = torch.zeros(50, ctx_dim)
    with torch.no_grad():
        y, _ = block(x, ctx)
    assert torch.all(torch.diff(y) > 0)


def test_isab_gradient_flows(torch_rng):
    """Backprop through the ISAB stack works."""
    B, n, dim = 2, 8, 16
    x = torch.randn(B, n, dim, generator=torch_rng, requires_grad=True)
    isab = ISAB(dim=dim, num_heads=4, num_inducing=4)
    out = isab(x).sum()
    out.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None for p in isab.parameters())
