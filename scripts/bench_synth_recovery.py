"""Synthetic-data benchmark mirroring Sec. 11 / ``results.json``.

Runs the full ``run_synthetic_experiment`` and emits two outputs:
  - ``outputs/results.json`` (numerical results, exactly matching the
    structure of ``docs/report/code/results.json``).
  - ``outputs/tables/bench_synth_recovery.csv`` (long-form CSV).
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


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
    out.pop("_plot_data")  # drop numpy arrays before JSON serialization

    _common.write_json("outputs/results.json", out)
    print("Wrote outputs/results.json")

    rows = []
    for cov in args.coverages:
        block = out[f"cov_{cov}"]
        for which in ("baseline", "havi"):
            method_label = "FinaleMe-style HMM" if which == "baseline" else "HAVI-Methyl"
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
                "method": "ELBO_final",
                "metric": "elbo_final",
                "value": block["elbo_final"],
            }
        )
    for k, v in out["identifiability"].items():
        rows.append(
            {"coverage": "identifiability", "method": k, "metric": "partial_r2", "value": v}
        )
    for k, v in out["tissue"].items():
        rows.append({"coverage": "tissue", "method": k, "metric": "rmse", "value": v})

    csv = _common.write_csv("outputs/tables/bench_synth_recovery.csv", rows)
    _common.copy_to_report_tables(csv)
    print(f"Wrote {csv}")
    # Echo the headline numbers
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
