"""Regenerate Table~\\ref{tab:recovery} (Sec. 11.1) — Pearson r and RMSE
of HAVI-Methyl vs. FinaleMe-style HMM across coverages.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Per-CpG beta recovery table (Sec. 11.1).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    args = parser.parse_args()

    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    n_iter = _common.small_or_full(args.fast, full=(10,), fast_values=(3,))[0]
    rng = np.random.default_rng(args.seed)

    rows = []
    for cov in args.coverages:
        result, _ = hm.run_one_coverage(S, L, cov, rng=rng, n_iter=n_iter)
        rows.append(
            {
                "coverage": cov,
                "method": "FinaleMe-style HMM",
                "pearson": result.baseline["pearson"],
                "rmse": result.baseline["rmse"],
                "mae": result.baseline["mae"],
            }
        )
        rows.append(
            {
                "coverage": cov,
                "method": "HAVI-Methyl",
                "pearson": result.havi["pearson"],
                "rmse": result.havi["rmse"],
                "mae": result.havi["mae"],
            }
        )

    out = _common.write_csv("outputs/tables/tab_recovery.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    print(f"Rows: {len(rows)} (numeric values from S={S}, L={L}, n_iter={n_iter}).")


if __name__ == "__main__":
    main()
