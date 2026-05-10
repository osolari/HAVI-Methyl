"""Regenerate Appendix~I Table~\\ref{tab:ext-tissue} from the canonical
``outputs/results.json`` artifact (Sec. 11.5, App. I).

Both rows are flagged ``simplified synthetic proxy`` because the released
harness does not implement the full Dirichlet/HDP tissue head; that work is
listed as IMPL-08 in ``docs/report/CODING_AGENT_HANDOFF.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import _common  # type: ignore

RESULTS_PATH = Path("outputs/results.json")
STATUS = "simplified synthetic proxy"


def main() -> None:
    parser = _common.base_parser("Tissue-fraction recovery (App. I).")
    parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found; run scripts/bench_synth_recovery.py first.")
    results = json.loads(RESULTS_PATH.read_text())
    tissue = results["tissue"]
    rows = [
        {
            "method": "FinaleMe-binarized + QP deconvolution",
            "tissue_fraction_rmse": tissue["rmse_baseline"],
            "status": STATUS,
        },
        {
            "method": "HAVI-Methyl simplified continuous proxy",
            "tissue_fraction_rmse": tissue["rmse_havi"],
            "status": STATUS,
        },
    ]
    out = _common.write_csv("outputs/tables/tab_tissue.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out} from {RESULTS_PATH}")
    for r in rows:
        print(f"  {r['method']:<45s}  {r['tissue_fraction_rmse']:.4f}")


if __name__ == "__main__":
    main()
