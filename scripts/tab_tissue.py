"""Regenerate Appendix~I Table~\\ref{tab:ext-tissue} — tissue-fraction RMSE
on synthetic three-tissue mixtures (Sec. 11.5).
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Tissue-fraction recovery (App. I).")
    parser.add_argument("--n-tissues", type=int, default=3)
    args = parser.parse_args()

    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    rng = np.random.default_rng(args.seed)
    res = hm.run_tissue_recovery(S=S, L=L, n_tissues=args.n_tissues, rng=rng)
    rows = [
        {
            "method": "FinaleMe-binarized + QP deconvolution",
            "tissue_fraction_rmse": res.rmse_baseline,
        },
        {"method": "HAVI-Methyl joint Dirichlet head", "tissue_fraction_rmse": res.rmse_havi},
    ]
    out = _common.write_csv("outputs/tables/tab_tissue.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['method']:<45s}  {r['tissue_fraction_rmse']:.4f}")


if __name__ == "__main__":
    main()
