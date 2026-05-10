"""Real-data FinaleMe benchmark runner (Sec. 12).

The 80 paired WGS/WGBS samples of Liu et al. 2024 are gated behind the
European Genome-Phenome Archive. Until the EGA-controlled dataset is
available, this script runs the *same* benchmark protocol against a
synthetic FinaleMe-proxy at S=80 / L=1000 / coverage=1x.

Every metric (Pearson, Spearman, AUC at beta=0.5, interval ECE, ICC(2,1),
DMR F1) is computed by the actual pipeline. The CSV ``_status`` column
makes clear that the underlying data is synthetic; nothing is fabricated.
When EGA access is secured, replace the simulator block with the real
loader and rerun.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("FinaleMe real-data benchmark scaffolding (Sec. 12).")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--loci", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = hm.simulate_dataset(args.samples, args.loci, args.coverage, rng=rng)
    # Inject a synthetic case/control DMR signal so the DMR F1 metric is
    # well defined: half of the samples have a +0.4 beta shift at 10% of
    # loci. This is a deliberately strong signal — the proxy is meant to
    # exercise the metric, not to estimate FinaleMe-vs-HAVI effect sizes.
    n_dmr = max(1, args.loci // 10)
    dmr_idx = rng.choice(args.loci, size=n_dmr, replace=False)
    case_idx = np.arange(args.samples // 2)
    sim.beta_sample[case_idx[:, None], dmr_idx] = np.clip(
        sim.beta_sample[case_idx[:, None], dmr_idx] + 0.4, 0.02, 0.98
    )
    # Re-simulate the affected fragment bags so the encoder sees the shift.
    for s in case_idx:
        for ell in dmr_idx:
            n = int(sim.n[s, ell])
            feats, z = hm.sample_fragment_bag(float(sim.beta_sample[s, ell]), n, rng=rng)
            sim.bags[s][ell] = feats
            sim.n_meth[s, ell] = int(z.sum())
    print(
        f"Running benchmark on synthetic FinaleMe-proxy: S={args.samples}, "
        f"L={args.loci}, coverage={args.coverage}x, DMR loci={n_dmr}"
    )
    truth_split = (case_idx, np.arange(args.samples // 2, args.samples))
    results = hm.evaluate_real_data_benchmark(
        bags=sim.bags,
        n_frag=sim.n,
        truth_beta=sim.beta_sample,
        truth_split=truth_split,
        rng=rng,
    )

    status = (
        f"synthetic FinaleMe-proxy (S={args.samples}, L={args.loci}, "
        f"coverage={args.coverage}x); replace with EGA Liu 2024 loader for "
        "real-data values"
    )
    rows = [r.as_row(name, status) for name, r in results.items()]
    # Append placeholders for external baselines we cannot run without their
    # codebases; keep the schema parity with the report table.
    for ext in ("FinaleMe", "DeepCpG", "Elastic-net regression", "MethylBERT"):
        rows.append(
            {
                "method": ext,
                "pearson_r": "XX",
                "spearman_r": "XX",
                "auc_meth_at_0p5": "XX",
                "ece_credible": "XX",
                "icc_2_1": "XX",
                "dmr_f1": "XX",
                "_status": (f"external baseline {ext} requires its own codebase; placeholder"),
            }
        )

    out = _common.write_csv("outputs/tables/bench_finaleme_realdata.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    for r in rows:
        if isinstance(r["pearson_r"], str):
            print(f"  {r['method']:<40s}  XX (external baseline)")
        else:
            print(
                f"  {r['method']:<40s}  r={r['pearson_r']:.3f}  "
                f"AUC={r['auc_meth_at_0p5']:.3f}  ECE={r['ece_credible']:.3f}  "
                f"DMR F1={r['dmr_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
