"""Sec. 12.5 compute-budget table.

Two parts:

  1. **Analytical FLOPs** for the planned full architecture, derived from
     ``havi_methyl.constants`` so the LaTeX table and the library cannot
     drift apart. The full architecture (ISAB + flow) is not exercised end
     to end yet (IMPL-02..05 in CODING_AGENT_HANDOFF.md); these are
     planning estimates based on the documented dimensions.

  2. **Measured wall-time** on local hardware for the simplified harness
     (``fit_svi_simplified``). This is a real reproducible artifact: every
     run records what it actually saw on the machine that ran the script.

Both sections are emitted to ``outputs/tables/bench_compute_budget.csv``;
the ``status`` column makes clear which rows are planning estimates and
which are measured.
"""

from __future__ import annotations

import platform
import time

import _common  # type: ignore
import numpy as np

import havi_methyl as hm


def measure_simplified_step(S: int, L: int, n_iter: int, seed: int) -> dict[str, float]:
    """Time ``fit_svi_simplified`` and report seconds-per-iteration."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.05, 0.95, size=(S, L))
    n_frag = rng.integers(1, 8, size=(S, L))
    # Warm up JIT/numpy paths.
    hm.fit_svi_simplified(pred, n_frag, n_iter=2)
    start = time.perf_counter()
    state = hm.fit_svi_simplified(pred, n_frag, n_iter=n_iter)
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": float(elapsed),
        "iters_per_s": float(n_iter / elapsed) if elapsed > 0 else float("inf"),
        "final_surrogate": float(state.elbo_history[-1]) if state.elbo_history else 0.0,
    }


def main() -> None:
    parser = _common.base_parser("Compute budget on local hardware + planning estimates.")
    parser.add_argument("--measure-S", type=int, default=12)
    parser.add_argument("--measure-L", type=int, default=4096)
    parser.add_argument("--measure-iter", type=int, default=20)
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

    measured = measure_simplified_step(args.measure_S, args.measure_L, args.measure_iter, args.seed)
    sys_label = f"{platform.system()} {platform.machine()} {platform.processor() or 'cpu'}"

    rows = [
        {
            "item": "Trainable neural parameters (planned full model)",
            "value": "~1.0M",
            "status": "planning estimate",
        },
        {
            "item": "Variational parameters",
            "value": "data-scale (2L + 2S)",
            "status": "planning estimate",
        },
        {
            "item": "Batch samples |B_s|",
            "value": str(bs),
            "status": "planning default",
        },
        {
            "item": "Batch loci |B_l|",
            "value": str(bl),
            "status": "planning default",
        },
        {
            "item": "Mean fragments per locus",
            "value": str(n_bar),
            "status": "planning assumption",
        },
        {
            "item": "ISAB FLOPs / step (analytical)",
            "value": f"{isab_flops:.3e}",
            "status": "analytical from constants",
        },
        {
            "item": "Flow FLOPs / step (analytical)",
            "value": f"{flow_flops:.3e}",
            "status": "analytical from constants",
        },
        {
            "item": "Total FLOPs / step (analytical)",
            "value": f"{total_flops:.3e}",
            "status": "analytical from constants",
        },
        {
            "item": f"Sustained throughput target ({args.sustained_tflops} TFLOPs/s)",
            "value": f"{steps_per_sec:.0f} steps/s",
            "status": "planning estimate; verify on target hardware",
        },
        {
            "item": "Wall time per epoch (planned 30M loci)",
            "value": f"{seconds_per_epoch:.1f} s",
            "status": "planning estimate; verify on target hardware",
        },
        {
            "item": (
                f"Measured fit_svi_simplified seconds (S={args.measure_S}, "
                f"L={args.measure_L}, iter={args.measure_iter})"
            ),
            "value": f"{measured['elapsed_s']:.4f}",
            "status": f"measured on {sys_label}",
        },
        {
            "item": "Measured fit_svi_simplified iters/sec",
            "value": f"{measured['iters_per_s']:.2f}",
            "status": f"measured on {sys_label}",
        },
        {
            "item": "Measured final surrogate ELBO/pair",
            "value": f"{measured['final_surrogate']:.6f}",
            "status": f"measured on {sys_label}",
        },
    ]
    out = _common.write_csv("outputs/tables/bench_compute_budget.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['item']:<60s} {r['value']}  [{r['status']}]")


if __name__ == "__main__":
    main()
