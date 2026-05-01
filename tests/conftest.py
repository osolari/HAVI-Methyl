"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG so tests are exactly reproducible."""
    return np.random.default_rng(20260429)


@pytest.fixture(params=[(4, 50), (8, 100)])
def small_shape(request) -> tuple[int, int]:
    """Parametrize over small (S, L) shapes for fast tests."""
    return request.param
