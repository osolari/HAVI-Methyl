"""Phase 1.4 verification artifact: end-to-end torch SVI on the synthetic
simulator with the same coverages used by the simplified harness.

Emits ``outputs/tables/bench_torch_svi.csv`` with one row per coverage
recording the recovery Pearson r, RMSE, MAE, final surrogate ELBO, and
wall-time per iteration. Status column always says ``measured (full
torch on local CPU)`` — this is a reproducible artifact, not a planning
estimate.

Canonical Sec. 11 numbers stay with ``fit_svi_simplified``; this script
is the IMPL-02..05 integration witness.
"""

from __future__ import annotations

import platform
import time

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Torch end-to-end SVI verification (Phase 1).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--loci", type=int, default=80)
    parser.add_argument("--iter", type=int, default=80)
    args = parser.parse_args()

    if hm.fit_svi_torch is None:  # pragma: no cover
        raise SystemExit("torch is required; install with `pip install torch`")

    sys_label = f"{platform.system()} {platform.machine()} {platform.processor() or 'cpu'}"
    rows = []
    for cov in args.coverages:
        sim = hm.simulate_dataset(args.samples, args.loci, cov, rng=args.seed)
        cfg = hm.TorchSVIConfig(
            in_dim=5,
            hidden=16,
            num_inducing=8,
            num_layers=1,
            batch_samples=args.samples,
            batch_loci=min(args.loci, 32),
        )
        start = time.perf_counter()
        state = hm.fit_svi_torch(
            bags=sim.bags,
            n_frag=sim.n,
            n_meth=sim.n_meth,
            n_iter=args.iter,
            config=cfg,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - start
        mean, std = hm.predict_with_torch_state(state, sim.bags, sim.n, n_samples=8)
        r = float(np.corrcoef(sim.beta_sample.flatten(), mean.flatten())[0, 1])
        rmse = float(np.sqrt(((sim.beta_sample - mean) ** 2).mean()))
        mae = float(np.abs(sim.beta_sample - mean).mean())
        rows.append(
            {
                "coverage": cov,
                "pearson_r": r,
                "rmse": rmse,
                "mae": mae,
                "elbo_first": state.elbo_history[0],
                "elbo_last": state.elbo_history[-1],
                "wall_seconds": f"{elapsed:.3f}",
                "iters": args.iter,
                "iters_per_s": f"{args.iter / elapsed:.2f}" if elapsed > 0 else "inf",
                "status": f"measured (full torch on {sys_label})",
            }
        )
        print(
            f"  cov={cov:>5}x  r={r:.3f}  rmse={rmse:.3f}  ELBO {state.elbo_history[0]:.3f} -> "
            f"{state.elbo_history[-1]:.3f}  ({elapsed:.2f}s)"
        )

    out = _common.write_csv("outputs/tables/bench_torch_svi.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
