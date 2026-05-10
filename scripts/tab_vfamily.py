"""Regenerate Table~\\ref{tab:vfamily} (Sec. 4.6) — variational family /
objective trade-offs across the per-(sample, locus) layer.

Schema matches ``docs/report/tables/tab_vfamily.csv``: ``status`` flags which
choice is exercised by the released simplified harness vs. the planned full
model. IWAE is labelled as an objective tightening, not a separate family,
matching the corrected manuscript wording.
"""

from __future__ import annotations

import _common  # type: ignore


def main() -> None:
    parser = _common.base_parser("Variational family table (Sec. 4.6).")
    parser.parse_args()

    rows = [
        {
            "family_or_objective": "Mean-field Gaussian (logit scale)",
            "expressiveness": "low",
            "cost_per_step": "low",
            "gradient_behavior": "low variance",
            "status": "implemented in simplified harness",
        },
        {
            "family_or_objective": "Mean-field Beta",
            "expressiveness": "low",
            "cost_per_step": "low",
            "gradient_behavior": "moderate",
            "status": "planned ablation",
        },
        {
            "family_or_objective": "Conditional scalar NSF flow",
            "expressiveness": "moderate-high",
            "cost_per_step": "moderate",
            "gradient_behavior": "pathwise gradients",
            "status": "planned full model",
        },
        {
            "family_or_objective": "IWAE-tightened flow objective",
            "expressiveness": "tighter bound",
            "cost_per_step": "high",
            "gradient_behavior": "DReG recommended",
            "status": "planned training objective",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_vfamily.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
