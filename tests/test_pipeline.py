"""Tests for the end-to-end synthetic pipeline (Sec. 11)."""

from __future__ import annotations

import havi_methyl as hm
import numpy as np
from havi_methyl.pipeline import (
    cpg_poor_mask,
    evaluate_with_intervals,
    run_one_coverage,
)


def test_cpg_poor_mask_returns_bool(rng):
    beta = rng.uniform(0, 1, size=200)
    mask = cpg_poor_mask(beta, q=0.3)
    assert mask.dtype == bool
    # Approximately 30% of loci should be flagged
    frac = float(mask.mean())
    assert 0.2 <= frac <= 0.4


def test_evaluate_with_intervals_shapes():
    true = np.linspace(0, 1, 10)
    pred = true + 0.05
    metrics = evaluate_with_intervals(true, pred)
    assert metrics.pearson > 0.99
    assert metrics.rmse < 0.1


def test_run_one_coverage_returns_full_block(rng):
    """Sec. 11.1: per-coverage block should report all metrics."""
    result, _ = run_one_coverage(S=4, L=40, coverage=5.0, rng=rng, n_iter=3, bootstrap_subset=2)
    assert "pearson" in result.baseline
    assert "pearson" in result.havi
    assert "rmse" in result.havi
    assert result.elbo_final == result.elbo_final  # not NaN
    assert "true" in result.plot_data


def test_run_synthetic_experiment_smoke(rng):
    """Smoke test the top-level entry point on a tiny config."""
    out = hm.run_synthetic_experiment(coverages=(1.0, 5.0), S=3, L=30, rng=rng, n_iter=3)
    assert "cov_1.0" in out
    assert "cov_5.0" in out
    assert "identifiability" in out
    assert "tissue" in out
    # leakage should at least be ordered
    leak = out["identifiability"]
    assert leak["leak_no_vib"] >= leak["leak_vib_only"] >= leak["leak_vib_plus_mqtl"]
