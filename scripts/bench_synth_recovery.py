"""Canonical synthetic-recovery benchmark (Sec. 11 / ``results.json``).

Single source of truth for every Sec. 11 artifact. Runs the simplified-harness
pipeline once and writes:

  - ``outputs/results.json`` (mirrors ``docs/report/results.json``)
  - ``outputs/plot_data.npz`` (per-coverage true / pred / interval / elbo arrays
    consumed by ``fig_*.py``)
  - ``outputs/tables/bench_synth_recovery.csv`` (long-form CSV)

All downstream ``tab_*.py`` (recovery, extended_recovery, identifiability,
tissue) and ``fig_*.py`` scripts read these artifacts rather than re-running
the pipeline themselves, so every figure and table in the manuscript is
internally consistent (IMPL-10 in ``docs/report/CODING_AGENT_HANDOFF.md``).
"""

from __future__ import annotations

from pathlib import Path

import _common  # type: ignore
import numpy as np

import havi_methyl as hm

PLOT_DATA_PATH = Path("outputs/plot_data.npz")
REPORT_RESULTS_PATH = Path("docs/report/results.json")
REPORT_CODE_RESULTS_PATH = Path("docs/report/code/results.json")


def _flatten_plot_data(plot_data: dict[float, dict]) -> dict[str, np.ndarray]:
    """Pack per-coverage arrays into an ``np.savez``-compatible mapping."""
    out: dict[str, np.ndarray] = {"coverages": np.array(sorted(plot_data.keys()))}
    for cov, arrays in plot_data.items():
        for key, value in arrays.items():
            out[f"{cov}__{key}"] = np.asarray(value, dtype=np.float64)
    return out


def main() -> None:
    parser = _common.base_parser("Synthetic recovery benchmark (Sec. 11).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    args = parser.parse_args()

    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    n_iter = _common.small_or_full(args.fast, full=(10,), fast_values=(3,))[0]
    rng = np.random.default_rng(args.seed)

    out = hm.run_synthetic_experiment(
        coverages=tuple(args.coverages), S=S, L=L, rng=rng, n_iter=n_iter
    )
    plot_data = out.pop("_plot_data")
    out.pop("_substream_seeds", None)

    _common.write_json("outputs/results.json", out)
    print("Wrote outputs/results.json")

    PLOT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PLOT_DATA_PATH, **_flatten_plot_data(plot_data))
    print(f"Wrote {PLOT_DATA_PATH}")

    # Mirror the JSON into docs/report/ so the LaTeX-side artifacts stay in
    # sync with the repo pipeline.
    REPORT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_RESULTS_PATH.write_bytes(Path("outputs/results.json").read_bytes())
    REPORT_CODE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_CODE_RESULTS_PATH.write_bytes(Path("outputs/results.json").read_bytes())
    print(f"Mirrored to {REPORT_RESULTS_PATH} and {REPORT_CODE_RESULTS_PATH}")

    rows = []
    for cov in args.coverages:
        block = out[f"cov_{cov}"]
        for which in ("baseline", "havi"):
            method_label = "FinaleMe-style HMM" if which == "baseline" else "HAVI-Methyl simplified"
            for metric, value in block[which].items():
                rows.append(
                    {
                        "coverage": cov,
                        "method": method_label,
                        "metric": metric,
                        "value": value,
                    }
                )
        rows.append(
            {
                "coverage": cov,
                "method": "surrogate",
                "metric": "elbo_final",
                "value": block["elbo_final"],
            }
        )

    csv = _common.write_csv("outputs/tables/bench_synth_recovery.csv", rows)
    _common.copy_to_report_tables(csv)
    print(f"Wrote {csv}")
    print("\nHeadline metrics:")
    for cov in args.coverages:
        b = out[f"cov_{cov}"]["baseline"]
        h = out[f"cov_{cov}"]["havi"]
        print(
            f"  {cov:>5}x  baseline r={b['pearson']:.3f} rmse={b['rmse']:.3f}"
            f" | HAVI r={h['pearson']:.3f} rmse={h['rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
