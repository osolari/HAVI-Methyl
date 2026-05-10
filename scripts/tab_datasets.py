"""Regenerate Table~\\ref{tab:datasets} (Sec. 12.1) — public datasets used
in the planned real-data benchmark.

Schema matches ``docs/report/tables/tab_datasets.csv``: every accession,
sample count, and modality is flagged ``verify ...`` because external dataset
verification is one of the open technical questions in
``docs/report/CODING_AGENT_HANDOFF.md``.
"""

from __future__ import annotations

import _common  # type: ignore


def main() -> None:
    parser = _common.base_parser("Datasets table (Sec. 12.1).")
    parser.parse_args()

    rows = [
        {
            "dataset": "Liu et al. 2024 (FinaleMe)",
            "scope": "80 planned comparison",
            "modality_or_coverage": "low-pass WGS/WGBS",
            "paired_WGBS": "yes",
            "verification_status": "verify controlled-access details",
        },
        {
            "dataset": "Sun 2015",
            "scope": "reference/planned",
            "modality_or_coverage": "WGBS",
            "paired_WGBS": "WGBS-only",
            "verification_status": "verify accession and sample count",
        },
        {
            "dataset": "Moss 2018",
            "scope": "reference/planned",
            "modality_or_coverage": "methylation array or atlas",
            "paired_WGBS": "N/A",
            "verification_status": "verify ground truth and accession",
        },
        {
            "dataset": "Loyfer 2023",
            "scope": "atlas-scale reference",
            "modality_or_coverage": "WGBS atlas",
            "paired_WGBS": "N/A",
            "verification_status": "verify accession and tissue labels",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_datasets.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
