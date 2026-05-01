"""Tests for normalizing flow primitives (App. C)."""

from __future__ import annotations

import numpy as np
from havi_methyl.flow import RationalQuadraticSpline, StackedFlow


def test_identity_spline_is_identity():
    rqs = RationalQuadraticSpline.identity(K=8, B=3.0)
    x = np.linspace(-2.5, 2.5, 11)
    y, log_jac = rqs.transform(x)
    np.testing.assert_allclose(y, x, atol=1e-9)
    np.testing.assert_allclose(log_jac, 0.0, atol=1e-9)


def test_stacked_identity_flow_matches_base():
    flow = StackedFlow(blocks=[RationalQuadraticSpline.identity(K=4) for _ in range(3)])
    x = np.array([-1.0, 0.0, 0.5, 1.5])
    y, log_jac = flow.transform(x)
    np.testing.assert_allclose(y, x, atol=1e-9)
    np.testing.assert_allclose(log_jac, 0.0, atol=1e-9)


def test_flow_log_density_normalizes():
    """Identity flow density on x ~ N(0,1) should integrate to 1."""
    flow = StackedFlow(blocks=[RationalQuadraticSpline.identity(K=4)])
    grid = np.linspace(-6, 6, 12001)
    log_dens = flow.log_density(grid)
    integral = float(np.trapezoid(np.exp(log_dens), grid))
    np.testing.assert_allclose(integral, 1.0, atol=1e-3)
