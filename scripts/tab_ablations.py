"""Regenerate Table~\\ref{tab:ablations} (Sec. 12.3) -- nested
HAVI-Methyl ablation matrix.

Reads measured Pearson r / AUC / conformal coverage_90 directly from
``outputs/tables/bench_ablation_matrix.csv`` (rows A0..A5 produced by
``bench_ablation_matrix.py`` on the synthetic FinaleMe-proxy) and emits
the LaTeX-display CSV with both the toggle flags and the measured
numbers. Status column reads "measured (synthetic FinaleMe-proxy)" for
every row.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common  # type: ignore


def _load_measurements() -> dict[str, dict[str, float]]:
    path = Path("outputs/tables/bench_ablation_matrix.csv")
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row["configuration"]
            out[cfg] = {
                "pearson_r": float(row.get("pearson_r", "nan")),
                "auc_meth_at_0p5": float(row.get("auc_meth_at_0p5", "nan")),
                "coverage_90": float(row["coverage_90"])
                if row.get("coverage_90", "").strip() not in ("", "nan")
                else float("nan"),
            }
    return out


def main() -> None:
    parser = _common.base_parser("Ablation factorial table (Sec. 12.3).")
    parser.parse_args()

    yes = "yes"
    no = ""
    toggles = [
        ("A0. Feature regression scaffold", no, no, no, no, no),
        ("A1. + Beta-Binomial pseudo-likelihood", yes, no, no, no, no),
        ("A2. + hierarchical SVI", yes, yes, no, no, no),
        ("A3. + amortized flow posterior", yes, yes, yes, no, no),
        ("A4. + leakage-control terms", yes, yes, yes, yes, no),
        ("A5. + conformal wrapper (full)", yes, yes, yes, yes, yes),
    ]
    measurements = _load_measurements()
    rows = []
    for cfg, bb, h, fl, vib, conf in toggles:
        m = measurements.get(cfg, {})
        rows.append(
            {
                "configuration": cfg,
                "BetaBin": bb,
                "Hierarchy": h,
                "Flow": fl,
                "VIB_mQTL": vib,
                "Conformal": conf,
                "pearson_r": f"{m['pearson_r']:.3f}" if "pearson_r" in m else "",
                "auc_meth_at_0p5": f"{m['auc_meth_at_0p5']:.3f}" if "auc_meth_at_0p5" in m else "",
                "coverage_90": (
                    f"{m['coverage_90']:.3f}"
                    if "coverage_90" in m and m["coverage_90"] == m["coverage_90"]  # not nan
                    else ""
                ),
                "status": ("measured (synthetic FinaleMe-proxy)" if measurements else "planned"),
            }
        )
    out = _common.write_csv("outputs/tables/tab_ablations.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
