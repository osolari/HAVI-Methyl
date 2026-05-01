"""Compute Sec. 12.5 wafer-scale FLOP / step / sample numbers analytically.

The numbers are derived from ``havi_methyl.constants`` so the LaTeX table and
the library cannot drift apart.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm


def main() -> None:
    parser = _common.base_parser("Compute budget on wafer-scale accelerator (Sec. 12.5).")
    parser.add_argument("--mean-frags-per-locus", type=int, default=20)
    parser.add_argument("--sustained-tflops", type=float, default=100.0)
    parser.add_argument("--total-loci", type=int, default=30_000_000)
    args = parser.parse_args()

    h = hm.DEFAULT_HPARAMS
    bs = h.batch_samples
    bl = h.batch_loci
    dc = h.hidden_dim
    K = h.flow_blocks
    m = h.inducing_points
    n_bar = args.mean_frags_per_locus

    isab_flops = 4 * n_bar * m * dc * bs * bl
    flow_flops = 2 * K * dc * dc * bs * bl
    total_flops = isab_flops + flow_flops
    steps_per_sec = (args.sustained_tflops * 1e12) / total_flops
    steps_per_epoch = args.total_loci / bl
    seconds_per_epoch = steps_per_epoch / steps_per_sec

    rows = [
        {"item": "ISAB FLOPs per step", "value": f"{isab_flops:.3e}"},
        {"item": "Flow forward FLOPs per step", "value": f"{flow_flops:.3e}"},
        {"item": "Total FLOPs per step", "value": f"{total_flops:.3e}"},
        {"item": "Sustained throughput (TFLOPs/s)", "value": f"{args.sustained_tflops:.1f}"},
        {"item": "Steps per second", "value": f"{steps_per_sec:.0f}"},
        {"item": "Steps per epoch (L=30M, |B_l|=4096)", "value": f"{steps_per_epoch:.0f}"},
        {"item": "Wall time per epoch (s)", "value": f"{seconds_per_epoch:.0f}"},
        {"item": "Wall time per epoch (min)", "value": f"{seconds_per_epoch / 60:.1f}"},
    ]
    out = _common.write_csv("outputs/tables/bench_compute_budget.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['item']:<35s} {r['value']}")


if __name__ == "__main__":
    main()
