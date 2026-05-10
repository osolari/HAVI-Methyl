"""App. H simulator-validation table (Sec. 10 / App. E targets).

Now (post IMPL-09) emits two CSVs:

  - ``outputs/tables/bench_simulator_validation.csv`` — schema matching
    ``docs/report/tables/bench_simulator_validation.csv`` with planning
    targets and publication actions. Status remains ``preliminary`` because
    the released harness still uses the compact simulator described in
    ``docs/report/code/run_experiments.py``; the chromatin-aware simulator
    is IMPL-09 in CODING_AGENT_HANDOFF.md.
  - ``outputs/tables/bench_simulator_metrics.csv`` — the actually-computed
    metrics from ``simulator_validation_metrics`` so future runs can compare
    against the reference targets without re-running the simulator.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Simulator validation against published cfDNA distributions.")
    parser.add_argument("--n-frag", type=int, default=100_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    metrics = hm.simulator_validation_metrics(n_frag=args.n_frag, rng=rng)

    rows = [
        {
            "axis": "Fragment-length modes",
            "current_status": "planning/preliminary",
            "expected_output": "histogram and distance to reference",
            "publication_action": "regenerate from final simulator",
        },
        {
            "axis": "End-motif frequencies",
            "current_status": "planning/preliminary",
            "expected_output": "top 4-mers and KL/distance",
            "publication_action": "regenerate from final simulator",
        },
        {
            "axis": "Periodicity spectrum",
            "current_status": "planning/preliminary",
            "expected_output": "autocorrelation peak near helical periodicity",
            "publication_action": "regenerate from final simulator",
        },
        {
            "axis": "Methylation-conditioned cut bias",
            "current_status": "planning/preliminary",
            "expected_output": "stratified motif/cut-bias effect size",
            "publication_action": "regenerate from final simulator",
        },
    ]
    out = _common.write_csv("outputs/tables/bench_simulator_validation.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")

    metric_rows = [
        {
            "axis": "Fragment-length primary mode (bp)",
            "value": f"{metrics['length_primary_mode_bp']:.2f}",
            "target": "~167 (Snyder 2016)",
        },
        {
            "axis": "Fragment-length 320-350 bp peak height (per bp)",
            "value": f"{metrics['length_secondary_height']:.6f}",
            "target": "~0.005",
        },
        {
            "axis": "10.4 bp periodicity autocorrelation peak",
            "value": f"{metrics['length_periodicity_amplitude']:.4f}",
            "target": "non-trivially positive",
        },
        {
            "axis": "Top-4 4-mer fraction at 5' cuts",
            "value": f"{metrics['top4_motif_fraction']:.4f}",
            "target": "~0.20 (Zhou 2022)",
        },
        {
            "axis": "Methylation-conditioned GC effect size",
            "value": f"{metrics['meth_cut_bias_effect_size']:.4f}",
            "target": "non-zero (compact simulator: ~0.10)",
        },
    ]
    metric_csv = _common.write_csv("outputs/tables/bench_simulator_metrics.csv", metric_rows)
    _common.copy_to_report_tables(metric_csv)
    print(f"Wrote {metric_csv}")
    for r in metric_rows:
        print(f"  {r['axis']:<55s}  value={r['value']}  target={r['target']}")


if __name__ == "__main__":
    main()
