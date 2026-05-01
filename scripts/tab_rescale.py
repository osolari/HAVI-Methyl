"""Regenerate Table~\\ref{tab:rescale} (Sec. 5.6) — mini-batch rescaling
factors per ELBO term in the two-plate model.
"""

from __future__ import annotations

import _common  # type: ignore


def main() -> None:
    parser = _common.base_parser("Mini-batch rescaling table (Sec. 5.6).")
    parser.parse_args()

    rows = [
        {
            "elbo_term": "Reconstruction log p(F | beta) summed over (s, l)",
            "indexed_by": "S x L plate",
            "rescaling_factor": "(S * L) / (|B_s| * |B_l|)",
        },
        {
            "elbo_term": "Population KL on mu_pop",
            "indexed_by": "L plate",
            "rescaling_factor": "L / |B_l|",
        },
        {
            "elbo_term": "Sample-shift KL on delta",
            "indexed_by": "S plate",
            "rescaling_factor": "S / |B_s|",
        },
        {
            "elbo_term": "Local KL on eta summed over (s, l)",
            "indexed_by": "S x L plate",
            "rescaling_factor": "(S * L) / (|B_s| * |B_l|)",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_rescale.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
