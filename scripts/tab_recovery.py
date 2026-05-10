"""Regenerate Table~\\ref{tab:recovery} (Sec. 11.1) from the canonical
``outputs/results.json`` artifact.

Reads the JSON produced by ``scripts/bench_synth_recovery.py`` so the table is
guaranteed consistent with the rest of Sec. 11.
"""

from __future__ import annotations

import json
from pathlib import Path

import _common  # type: ignore

RESULTS_PATH = Path("outputs/results.json")


def main() -> None:
    parser = _common.base_parser("Per-CpG beta recovery table (Sec. 11.1).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found; run scripts/bench_synth_recovery.py first.")
    results = json.loads(RESULTS_PATH.read_text())

    rows = []
    for cov in args.coverages:
        block = results[f"cov_{cov}"]
        rows.append(
            {
                "coverage": cov,
                "method": "FinaleMe-style HMM",
                "pearson": block["baseline"]["pearson"],
                "rmse": block["baseline"]["rmse"],
                "mae": block["baseline"]["mae"],
            }
        )
        rows.append(
            {
                "coverage": cov,
                "method": "HAVI-Methyl simplified",
                "pearson": block["havi"]["pearson"],
                "rmse": block["havi"]["rmse"],
                "mae": block["havi"]["mae"],
            }
        )

    out = _common.write_csv("outputs/tables/tab_recovery.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out} from {RESULTS_PATH}")


if __name__ == "__main__":
    main()
