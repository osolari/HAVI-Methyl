"""Real-data FinaleMe benchmark stub (Sec. 12.1, requires EGA access).

The 80 paired WGS/WGBS samples of Liu et al. 2024 are gated behind the
European Genome-Phenome Archive. Direct numbers therefore cannot be filled
in by this script — we emit a CSV with the expected schema and an XX
placeholder per row, plus a ``_status`` column documenting why.
"""

from __future__ import annotations

import _common  # type: ignore


def main() -> None:
    parser = _common.base_parser("FinaleMe real-data benchmark (Sec. 12).")
    parser.parse_args()

    rows = []
    for method in [
        "HAVI-Methyl (full)",
        "HAVI-Methyl (no flow)",
        "HAVI-Methyl (no VIB)",
        "FinaleMe",
        "DeepCpG",
        "Elastic-net regression",
        "MethylBERT",
    ]:
        rows.append(
            {
                "method": method,
                "pearson_r": "XX",
                "spearman_r": "XX",
                "auc_meth_at_0p5": "XX",
                "ece_credible": "XX",
                "icc_2_1": "XX",
                "dmr_f1": "XX",
                "_status": "Requires EGA-controlled FinaleMe dataset; placeholder per repo.md",
            }
        )
    out = _common.write_csv("outputs/tables/bench_finaleme_realdata.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out} (XX placeholder rows; see _status column).")


if __name__ == "__main__":
    main()
