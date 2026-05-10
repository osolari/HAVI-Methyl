"""Regenerate Appendix~I Table~\\ref{tab:ext-ident} from the canonical
``outputs/results.json`` artifact (Sec. 11.4, App. I).
"""

from __future__ import annotations

import json
from pathlib import Path

import _common  # type: ignore

RESULTS_PATH = Path("outputs/results.json")


def main() -> None:
    parser = _common.base_parser("Identifiability ablation (App. I).")
    parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found; run scripts/bench_synth_recovery.py first.")
    results = json.loads(RESULTS_PATH.read_text())
    ident = results["identifiability"]
    rows = [
        {
            "configuration": "No regularization (VIB off, mQTL off)",
            "partial_r2_prior": ident["leak_no_vib"],
        },
        {
            "configuration": "VIB only (beta_VIB=0.3)",
            "partial_r2_prior": ident["leak_vib_only"],
        },
        {
            "configuration": "VIB + mQTL anchors",
            "partial_r2_prior": ident["leak_vib_plus_mqtl"],
        },
    ]
    out = _common.write_csv("outputs/tables/tab_identifiability.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out} from {RESULTS_PATH}")
    for r in rows:
        print(f"  {r['configuration']:<45s}  {r['partial_r2_prior']:.5f}")


if __name__ == "__main__":
    main()
