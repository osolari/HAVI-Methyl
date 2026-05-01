"""Regenerate Table~\\ref{tab:vfamily} (Sec. 4.6) — variational family
trade-offs across the per-(sample, locus) layer.

This is a *protocol* table: each row documents an architectural choice and its
qualitative trade-offs as stated in the manuscript. We emit it as CSV so the
LaTeX build can re-render it from a single source of truth.
"""

from __future__ import annotations

import _common  # type: ignore


def main() -> None:
    parser = _common.base_parser("Variational family table (Sec. 4.6).")
    parser.parse_args()

    rows = [
        {
            "family": "Mean-field Gaussian (logit scale)",
            "expressiveness": "low (unimodal)",
            "cost_per_step": "low",
            "gradient_variance": "low",
        },
        {
            "family": "Mean-field Beta",
            "expressiveness": "low (unimodal)",
            "cost_per_step": "low",
            "gradient_variance": "moderate (Beta reparam)",
        },
        {
            "family": "Normalizing flow (NSF, K blocks)",
            "expressiveness": "high (multi-modal)",
            "cost_per_step": "moderate (O(K))",
            "gradient_variance": "low (pathwise)",
        },
        {
            "family": "IWAE-tightened flow (K samples)",
            "expressiveness": "high",
            "cost_per_step": "high (Kx)",
            "gradient_variance": "moderate (DReG; Tucker 2019)",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_vfamily.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
