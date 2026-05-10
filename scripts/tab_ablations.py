"""Regenerate Table~\\ref{tab:ablations} (Sec. 12.3) — planned nested
HAVI-Methyl ablation matrix.

Schema matches ``docs/report/tables/tab_ablations.csv``. Every row is flagged
``planned`` because the released harness contains only the simplified Sec. 11
variant; the nested ablations are part of the real-data benchmark in
Sec. 12 and IMPL-06..09 of CODING_AGENT_HANDOFF.md.
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
            "configuration": "A0. Feature regression scaffold",
            "BetaBin": no,
            "Hierarchy": no,
            "Flow": no,
            "VIB_mQTL": no,
            "Conformal": no,
            "status": "planned",
        },
        {
            "configuration": "A1. + Beta-Binomial pseudo-likelihood",
            "BetaBin": yes,
            "Hierarchy": no,
            "Flow": no,
            "VIB_mQTL": no,
            "Conformal": no,
            "status": "planned",
        },
        {
            "configuration": "A2. + hierarchical SVI",
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": no,
            "VIB_mQTL": no,
            "Conformal": no,
            "status": "planned",
        },
        {
            "configuration": "A3. + amortized flow posterior",
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": yes,
            "VIB_mQTL": no,
            "Conformal": no,
            "status": "planned",
        },
        {
            "configuration": "A4. + leakage-control terms",
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": yes,
            "VIB_mQTL": yes,
            "Conformal": no,
            "status": "planned",
        },
        {
            "configuration": "A5. + conformal wrapper (full)",
            "BetaBin": yes,
            "Hierarchy": yes,
            "Flow": yes,
            "VIB_mQTL": yes,
            "Conformal": yes,
            "status": "planned",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_ablations.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
