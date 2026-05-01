"""Regenerate Table~\\ref{tab:datasets} (Sec. 12.1) — public datasets used
in the benchmarking plan.
"""

from __future__ import annotations

import _common  # type: ignore


def main() -> None:
    parser = _common.base_parser("Datasets table (Sec. 12.1).")
    parser.parse_args()

    rows = [
        {
            "dataset": "Liu et al. 2024 (FinaleMe)",
            "S": 80,
            "mean_coverage": "~1x",
            "paired_WGBS": "yes",
            "access": "EGA",
        },
        {
            "dataset": "Sun 2015",
            "S": 32,
            "mean_coverage": "~35x WGBS",
            "paired_WGBS": "WGBS-only",
            "access": "SRA",
        },
        {
            "dataset": "Moss 2018",
            "S": 25,
            "mean_coverage": "methylation array",
            "paired_WGBS": "N/A",
            "access": "GEO",
        },
        {
            "dataset": "Loyfer 2023",
            "S": "39 cell types from 205 samples",
            "mean_coverage": "WGBS atlas",
            "paired_WGBS": "N/A",
            "access": "GSE186458",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_datasets.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
