"""Regenerate Table~\\ref{tab:ablations} (Sec. 12.3) — six-row ablation
factorial. Each row adds one component on top of the previous configuration.
"""

from __future__ import annotations

import _common  # type: ignore


def main() -> None:
    parser = _common.base_parser("Ablation factorial table (Sec. 12.3).")
    parser.parse_args()

    yes = "yes"
    no = ""
    rows = [
        {
            "configuration": "1. VB-HMM only",
            "VB_HMM": yes,
            "BetaBin": no,
            "Hierarchy": no,
            "Flow": no,
            "VIB": no,
            "Conformal": no,
        },
        {
            "configuration": "2. + Beta-Binomial likelihood",
            "VB_HMM": yes,
            "BetaBin": yes,
            "Hierarchy": no,
            "Flow": no,
            "VIB": no,
            "Conformal": no,
        },
        {
            "configuration": "3. + hierarchical SVI",
            "VB_HMM": yes,
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": no,
            "VIB": no,
            "Conformal": no,
        },
        {
            "configuration": "4. + amortized flow posterior",
            "VB_HMM": yes,
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": yes,
            "VIB": no,
            "Conformal": no,
        },
        {
            "configuration": "5. + VIB de-confounding",
            "VB_HMM": yes,
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": yes,
            "VIB": yes,
            "Conformal": no,
        },
        {
            "configuration": "6. + conformal wrapper (full)",
            "VB_HMM": yes,
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": yes,
            "VIB": yes,
            "Conformal": yes,
        },
    ]
    out = _common.write_csv("outputs/tables/tab_ablations.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
