"""Regenerate Appendix~I Table~\\ref{tab:ext-recovery} from the canonical
``outputs/results.json`` artifact (Sec. 11.1, App. I).
"""

from __future__ import annotations

import json
from pathlib import Path

import _common  # type: ignore

RESULTS_PATH = Path("outputs/results.json")


def main() -> None:
    parser = _common.base_parser("Extended recovery table (App. I).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found; run scripts/bench_synth_recovery.py first.")
    results = json.loads(RESULTS_PATH.read_text())

    rows = []
    for cov in args.coverages:
        block = results[f"cov_{cov}"]
        for method_key, method_label in (
            ("baseline", "FinaleMe-style"),
            ("havi", "HAVI-Methyl simplified"),
        ):
            d = block[method_key]
            rows.append(
                {
                    "coverage": cov,
                    "method": method_label,
                    "pearson": d.get("pearson"),
                    "rmse": d.get("rmse"),
                    "mae": d.get("mae"),
                    "coverage_90": d.get("coverage_90"),
                    "mean_width": d.get("mean_width"),
                    "pearson_low_information_proxy": d.get("pearson_cpgpoor"),
                }
            )

    out = _common.write_csv("outputs/tables/tab_extended_recovery.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out} from {RESULTS_PATH}")


if __name__ == "__main__":
    main()
