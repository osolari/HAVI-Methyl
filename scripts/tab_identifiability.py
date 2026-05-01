"""Regenerate Appendix~I Table~\\ref{tab:ext-ident} — prior-leakage attribution
across regularization regimes (Sec. 11.4).
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Identifiability ablation (App. I).")
    args = parser.parse_args()

    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    rng = np.random.default_rng(args.seed)

    res = hm.run_identifiability_stress_test(S=S, L=L, coverage=5.0, rng=rng)
    rows = [
        {
            "configuration": "No regularization (VIB off, mQTL off)",
            "partial_r2_prior": res.leak_no_vib,
        },
        {
            "configuration": "VIB only (beta_VIB=0.3)",
            "partial_r2_prior": res.leak_vib_only,
        },
        {
            "configuration": "VIB + mQTL anchors",
            "partial_r2_prior": res.leak_vib_plus_mqtl,
        },
    ]
    out = _common.write_csv("outputs/tables/tab_identifiability.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['configuration']:<45s}  {r['partial_r2_prior']:.5f}")


if __name__ == "__main__":
    main()
