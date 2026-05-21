"""Phase 1 / IMPL-04 verification artifact: end-to-end torch SVI on the
synthetic simulator with the same coverages used by the simplified harness.

Emits ``outputs/tables/bench_torch_svi.csv`` with one row per (head,
coverage) combination — comparing the stable Gaussian posterior head
against the rewritten conditional NSF flow head — recording the
recovery Pearson r, RMSE, MAE, final surrogate ELBO, and wall-time per
iteration. Status column always says ``measured (full torch on local
CPU)`` — these are reproducible artifacts, not planning estimates.

Canonical Sec. 11 numbers stay with ``fit_svi_simplified``.
"""

from __future__ import annotations

import platform
import time

import _common  # type: ignore
import numpy as np

import havi_methyl as hm


def main() -> None:
    parser = _common.base_parser("Torch end-to-end SVI verification (Phase 1).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--loci", type=int, default=80)
    parser.add_argument("--iter", type=int, default=80)
    parser.add_argument(
        "--heads",
        type=str,
        nargs="+",
        default=["gaussian", "flow"],
        help="Posterior heads to evaluate (gaussian, flow).",
    )
    parser.add_argument(
        "--k-iwae",
        type=int,
        nargs="+",
        default=[1, 4],
        help="K values for the IWAE bound (1 = standard ELBO).",
    )
    parser.add_argument(
        "--include-dreg",
        action="store_true",
        help="Also run the simplified DReG-IWAE variant for each K>1.",
    )
    args = parser.parse_args()

    if hm.fit_svi_torch is None:  # pragma: no cover
        raise SystemExit("torch is required; install with `pip install torch`")

    # --fast: shrink for CI smoke (samples/loci/iter/coverages).
    if getattr(args, "fast", False):
        args.iter = 10
        args.coverages = [1.0]
        args.samples = 4
        args.loci = 40
        args.heads = args.heads[:1]
        args.iwae_ks = [1]

    sys_label = f"{platform.system()} {platform.machine()} {platform.processor() or 'cpu'}"
    rows = []
    # Build the (objective, K, dreg) sweep. K=1 always uses standard ELBO; K>1
    # adds IWAE and optionally the simplified DReG variant.
    objectives = []
    for k in args.k_iwae:
        if k <= 1:
            objectives.append(("ELBO", 1, False))
        else:
            objectives.append((f"IWAE K={k}", k, False))
            if args.include_dreg:
                objectives.append((f"DReG K={k}", k, True))

    for head in args.heads:
        for cov in args.coverages:
            sim = hm.simulate_dataset(args.samples, args.loci, cov, rng=args.seed)
            for obj_label, k, dreg in objectives:
                cfg = hm.TorchSVIConfig(
                    in_dim=5,
                    hidden=16,
                    num_inducing=8,
                    num_layers=1,
                    batch_samples=args.samples,
                    batch_loci=min(args.loci, 32),
                    posterior=head,
                    k_iwae=k,
                    iwae_dreg=dreg,
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
                        "posterior_head": head,
                        "objective": obj_label,
                        "coverage": cov,
                        "pearson_r": r,
                        "rmse": rmse,
                        "mae": mae,
                        "elbo_first": state.elbo_history[0],
                        "elbo_last": state.elbo_history[-1],
                        "wall_seconds": f"{elapsed:.3f}",
                        "iters": args.iter,
                        "iters_per_s": (f"{args.iter / elapsed:.2f}" if elapsed > 0 else "inf"),
                        "status": f"measured (full torch on {sys_label})",
                    }
                )
                print(
                    f"  head={head:<8s} {obj_label:<10s} cov={cov:>5}x  "
                    f"r={r:.3f}  rmse={rmse:.3f}  "
                    f"ELBO {state.elbo_history[0]:.3f} -> {state.elbo_history[-1]:.3f}  "
                    f"({elapsed:.2f}s)"
                )

    out = _common.write_csv("outputs/tables/bench_torch_svi.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
