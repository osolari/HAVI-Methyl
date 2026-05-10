"""Phase 6 multi-seed bootstrap CI for the Sec. 11 simplified harness.

Runs ``run_synthetic_experiment`` across N seeds and emits
``outputs/tables/tab_recovery_multiseed.csv`` with median, 5th, and 95th
percentile of Pearson r, RMSE, MAE, raw 90%-interval coverage and
width, plus low-information-proxy r per coverage and method.

Does **not** overwrite the canonical seed=20260429 ``results.json`` or
``tab_recovery.csv``: those remain the single-seed point estimates that
Sec. 11 / App. I cite. The multiseed sidecar is meant to be cited
alongside the canonical run as the variability-of-the-harness check
called for in Sec. 11's caveats subsection (``CODING_AGENT_HANDOFF.md``
"multi-seed synthetic sensitivity").
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Multi-seed bootstrap CI for the Sec. 11 harness.")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of seeds to run (default: 20).",
    )
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 5.0, 30.0],
    )
    args = parser.parse_args()

    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    n_iter = _common.small_or_full(args.fast, full=(10,), fast_values=(3,))[0]
    print(
        f"Multi-seed runner: n_seeds={args.n_seeds}, coverages={args.coverages}, "
        f"S={S}, L={L}, n_iter={n_iter}"
    )

    base_seeds = list(range(args.seed, args.seed + args.n_seeds))
    metric_keys = ("pearson", "rmse", "mae", "coverage_90", "mean_width", "pearson_cpgpoor")
    # Collect (n_seeds,) values per coverage / method / metric.
    accum: dict[tuple[float, str, str], list[float]] = {}
    for seed in base_seeds:
        out = hm.run_synthetic_experiment(
            coverages=tuple(args.coverages), S=S, L=L, rng=seed, n_iter=n_iter
        )
        for cov in args.coverages:
            for method_key in ("baseline", "havi"):
                block = out[f"cov_{cov}"][method_key]
                for metric in metric_keys:
                    if metric not in block:
                        continue
                    accum.setdefault((cov, method_key, metric), []).append(float(block[metric]))

    rows = []
    for cov in args.coverages:
        for method_key, method_label in (
            ("baseline", "FinaleMe-style HMM"),
            ("havi", "HAVI-Methyl simplified"),
        ):
            for metric in metric_keys:
                values = accum.get((cov, method_key, metric))
                if not values:
                    continue
                arr = np.asarray(values, dtype=np.float64)
                rows.append(
                    {
                        "coverage": cov,
                        "method": method_label,
                        "metric": metric,
                        "n_seeds": len(arr),
                        "median": float(np.median(arr)),
                        "p5": float(np.percentile(arr, 5)),
                        "p95": float(np.percentile(arr, 95)),
                        "mean": float(arr.mean()),
                        "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                    }
                )

    out_path = _common.write_csv("outputs/tables/tab_recovery_multiseed.csv", rows)
    _common.copy_to_report_tables(out_path)
    print(f"Wrote {out_path} ({len(rows)} rows from {args.n_seeds} seeds)")
    # Emit a short summary of per-coverage Pearson r medians + 90% CI.
    print("\nHeadline Pearson r medians (5th/95th percentile across seeds):")
    for cov in args.coverages:
        for method_key, label in (("baseline", "FinaleMe-style"), ("havi", "HAVI-Methyl")):
            values = accum.get((cov, method_key, "pearson"))
            if values:
                arr = np.asarray(values)
                print(
                    f"  cov={cov:>5}x  {label:<14s}  median={np.median(arr):.3f}  "
                    f"(p5={np.percentile(arr, 5):.3f}, p95={np.percentile(arr, 95):.3f})"
                )


if __name__ == "__main__":
    main()
