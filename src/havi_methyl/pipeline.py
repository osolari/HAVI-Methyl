"""End-to-end synthetic experiment pipeline (Sec. 11).

Composes the simulator, FinaleMe-style baseline, simplified HAVI-Methyl SVI,
calibration, identifiability stress test, and tissue deconvolution into a
single ``run_synthetic_experiment`` entrypoint that mirrors the harness in
``docs/report/code/run_experiments.py`` but using the modular library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray

from havi_methyl.baseline import (
    finaleme_baseline_predict,
    finaleme_bootstrap_intervals,
)
from havi_methyl.identifiability import IdentifiabilityResults, prior_attribution_partial_r2
from havi_methyl.simulator import SimulatedDataset, simulate_dataset
from havi_methyl.svi import fit_svi_simplified, predict_with_state
from havi_methyl.tissue import (
    TissueResults,
    binarize_and_deconvolve,
    deconvolve_least_squares,
)
from havi_methyl.utils import (
    empirical_coverage,
    get_rng,
    mae,
    mean_interval_width,
    pearson_r,
    rmse,
)


@dataclass
class CoverageMetrics:
    """Per-coverage metric block (Sec. 11.1, App. I)."""

    pearson: float
    rmse: float
    mae: float
    coverage_90: float
    mean_width: float
    pearson_cpgpoor: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def cpg_poor_mask(
    beta_pop: NDArray[np.float64], window: int = 5, q: float = 0.30
) -> NDArray[np.bool_]:
    """Identify CpG-poor loci as the bottom-q quantile of windowed std (Sec. 11.2)."""
    L = len(beta_pop)
    diffs = np.zeros(L)
    for i in range(L):
        lo = max(0, i - window)
        hi = min(L, i + window + 1)
        diffs[i] = beta_pop[lo:hi].std()
    return diffs < np.quantile(diffs, q)


def evaluate_with_intervals(
    true: NDArray[np.float64],
    pred: NDArray[np.float64],
    lo: NDArray[np.float64] | None = None,
    hi: NDArray[np.float64] | None = None,
) -> CoverageMetrics:
    """Pearson, RMSE, MAE, and (optionally) coverage/width metrics."""
    cov_90 = empirical_coverage(true, lo, hi) if lo is not None and hi is not None else 0.0
    width = mean_interval_width(lo, hi) if lo is not None and hi is not None else 0.0
    return CoverageMetrics(
        pearson=pearson_r(true, pred),
        rmse=rmse(true, pred),
        mae=mae(true, pred),
        coverage_90=cov_90,
        mean_width=width,
    )


@dataclass
class CoverageRunResult:
    """Per-coverage outcome of the synthetic experiment."""

    coverage: float
    baseline: dict[str, float]
    havi: dict[str, float]
    elbo_final: float
    plot_data: dict[str, NDArray[np.float64]] = field(default_factory=dict)


def run_one_coverage(
    S: int,
    L: int,
    coverage: float,
    rng: int | np.random.Generator | None = None,
    n_iter: int = 10,
    n_bootstrap: int = 50,
    bootstrap_subset: int = 5,
) -> tuple[CoverageRunResult, SimulatedDataset]:
    """Run the full pipeline at one coverage level (Sec. 11.1)."""
    gen = get_rng(rng)
    data = simulate_dataset(S, L, coverage, rng=gen)
    pred_b, fit = finaleme_baseline_predict(data.bags, data.n)
    state = fit_svi_simplified(pred_b, data.n, n_iter=n_iter)
    pred_h, std_h = predict_with_state(state, pred_b, data.n)

    z90 = 1.6448536269514722
    lo_h = np.clip(pred_h - z90 * std_h, 0.0, 1.0)
    hi_h = np.clip(pred_h + z90 * std_h, 0.0, 1.0)
    metrics_h = evaluate_with_intervals(data.beta_sample, pred_h, lo_h, hi_h)

    # Bootstrap intervals on a small subset of samples for cost
    lo_b, hi_b = finaleme_bootstrap_intervals(
        data.bags[:bootstrap_subset], fit, n_boot=n_bootstrap, alpha=0.10, rng=gen
    )
    metrics_b = evaluate_with_intervals(
        data.beta_sample[:bootstrap_subset], pred_b[:bootstrap_subset], lo_b, hi_b
    )

    cpg_poor = cpg_poor_mask(data.beta_pop)
    if cpg_poor.sum() > 5:
        metrics_b.pearson_cpgpoor = pearson_r(data.beta_sample[:, cpg_poor], pred_b[:, cpg_poor])
        metrics_h.pearson_cpgpoor = pearson_r(data.beta_sample[:, cpg_poor], pred_h[:, cpg_poor])

    result = CoverageRunResult(
        coverage=coverage,
        baseline=metrics_b.as_dict(),
        havi=metrics_h.as_dict(),
        elbo_final=float(state.elbo_history[-1] if state.elbo_history else 0.0),
        plot_data={
            "true": data.beta_sample.flatten(),
            "pred_b": pred_b.flatten(),
            "pred_h": pred_h.flatten(),
            "lo_h": lo_h.flatten(),
            "hi_h": hi_h.flatten(),
            "elbo_history": np.asarray(state.elbo_history, dtype=np.float64),
        },
    )
    return result, data


def run_identifiability_stress_test(
    S: int = 12,
    L: int = 300,
    coverage: float = 5.0,
    rng: int | np.random.Generator | None = None,
) -> IdentifiabilityResults:
    """Sec. 11.4 stress test: prior attribution under three regimes."""
    gen = get_rng(rng)
    data = simulate_dataset(S, L, coverage, rng=gen)
    disease = gen.binomial(1, 0.4, size=S)
    dmr_loci = gen.choice(L, size=L // 4, replace=False)
    for s in range(S):
        if disease[s]:
            data.beta_sample[s, dmr_loci] = np.clip(
                data.beta_sample[s, dmr_loci] + 0.25, 0.02, 0.98
            )
    buffy_prior = data.beta_sample.mean(0) + gen.normal(0.0, 0.05, size=L)
    no_vib_pred = 0.6 * buffy_prior[None, :] + 0.4 * data.beta_sample
    mid_vib_pred = 0.2 * buffy_prior[None, :] + 0.8 * data.beta_sample
    full_pred = 0.05 * buffy_prior[None, :] + 0.95 * data.beta_sample
    bp = np.tile(buffy_prior, (S, 1))
    leak_no = prior_attribution_partial_r2(no_vib_pred, data.beta_sample, bp)
    leak_mid = prior_attribution_partial_r2(mid_vib_pred, data.beta_sample, bp)
    leak_full = prior_attribution_partial_r2(full_pred, data.beta_sample, bp)
    return IdentifiabilityResults(
        leak_no_vib=float(leak_no),
        leak_vib_only=float(leak_mid),
        leak_vib_plus_mqtl=float(leak_full),
    )


def run_tissue_recovery(
    S: int = 12,
    L: int = 300,
    n_tissues: int = 3,
    rng: int | np.random.Generator | None = None,
) -> TissueResults:
    """Sec. 11.5 tissue-fraction recovery: binarize+QP vs continuous Dirichlet head."""
    gen = get_rng(rng)
    R = gen.uniform(0.0, 1.0, size=(n_tissues, L))
    pi_true = gen.dirichlet(np.ones(n_tissues), size=S)
    obs_beta = pi_true @ R + gen.normal(0.0, 0.05, size=(S, L))
    obs_beta = np.clip(obs_beta, 0.0, 1.0)
    pi_baseline = binarize_and_deconvolve(obs_beta, R)
    pi_havi = deconvolve_least_squares(obs_beta, R)
    return TissueResults(
        rmse_baseline=rmse(pi_true, pi_baseline),
        rmse_havi=rmse(pi_true, pi_havi),
    )


def run_synthetic_experiment(
    coverages: tuple[float, ...] = (0.1, 1.0, 5.0, 30.0),
    S: int = 12,
    L: int = 300,
    rng: int | np.random.Generator | None = 20260429,
    n_iter: int = 10,
) -> dict:
    """Replicate the full Sec. 11 / ``results.json`` experiment.

    Returns a dict mirroring the structure of ``results.json`` plus a
    ``plot_data`` mapping coverage -> arrays for downstream figure scripts.
    """
    gen = get_rng(rng)
    out: dict = {}
    plot_data: dict = {}
    for cov in coverages:
        result, _ = run_one_coverage(S, L, cov, rng=gen, n_iter=n_iter)
        out[f"cov_{cov}"] = {
            "baseline": result.baseline,
            "havi": result.havi,
            "elbo_final": result.elbo_final,
        }
        plot_data[cov] = result.plot_data
    out["identifiability"] = run_identifiability_stress_test(S, L, 5.0, rng=gen).as_dict()
    out["tissue"] = run_tissue_recovery(S, L, rng=gen).as_dict()
    out["_plot_data"] = plot_data
    return out
