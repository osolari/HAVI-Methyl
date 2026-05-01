"""Regenerate Appendix~I Table~\\ref{tab:ext-recovery} — extended per-coverage
detail with Pearson, RMSE, MAE, 90% coverage and width, and CpG-poor Pearson.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Extended recovery table (App. I).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    args = parser.parse_args()

    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    n_iter = _common.small_or_full(args.fast, full=(10,), fast_values=(3,))[0]
    rng = np.random.default_rng(args.seed)

    rows = []
    for cov in args.coverages:
        result, _ = hm.run_one_coverage(S, L, cov, rng=rng, n_iter=n_iter)
        for method_key, method_label in (("baseline", "FinaleMe-style"), ("havi", "HAVI-Methyl")):
            d = getattr(result, method_key)
            rows.append(
                {
                    "coverage": cov,
                    "method": method_label,
                    "pearson": d.get("pearson"),
                    "rmse": d.get("rmse"),
                    "mae": d.get("mae"),
                    "coverage_90": d.get("coverage_90"),
                    "mean_width": d.get("mean_width"),
                    "pearson_cpgpoor": d.get("pearson_cpgpoor"),
                }
            )

    out = _common.write_csv("outputs/tables/tab_extended_recovery.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
