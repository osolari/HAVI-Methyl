"""FinaleMe benchmark runner with optional real-data path (Phase 5).

Two modes:

  - Default: run the synthetic FinaleMe-proxy at S=80 / L=1000 /
    coverage=1x with an injected case/control DMR signal so the DMR F1
    metric is exercised. Every row's ``_status`` column says
    ``synthetic FinaleMe-proxy (...)``.
  - ``--data-dir <path>``: load the real Liu 2024 paired WGS+WGBS
    dataset from that directory using ``havi_methyl.io.load_finaleme_dataset``.
    The metric stack is identical; ``_status`` rows say
    ``Liu 2024 (data-dir=..., n_samples=..., n_loci=...)``.

Every value in the resulting CSV is the actual pipeline output for the
selected mode — nothing is fabricated.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def _run_synthetic_proxy(args) -> tuple[dict, np.ndarray, np.ndarray, str, dict[str, int]]:
    rng = np.random.default_rng(args.seed)
    sim = hm.simulate_dataset(args.samples, args.loci, args.coverage, rng=rng)
    n_dmr = max(1, args.loci // 10)
    dmr_idx = rng.choice(args.loci, size=n_dmr, replace=False)
    case_idx = np.arange(args.samples // 2)
    sim.beta_sample[case_idx[:, None], dmr_idx] = np.clip(
        sim.beta_sample[case_idx[:, None], dmr_idx] + 0.4, 0.02, 0.98
    )
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
        f"coverage={args.coverage}x); replace with --data-dir for real Liu 2024 numbers"
    )
    return results, sim.n, sim.beta_sample, status, {"S": args.samples, "L": args.loci}


def _run_real_data(args) -> tuple[dict, np.ndarray, np.ndarray, str, dict[str, int]]:
    print(f"Loading FinaleMe paired WGS/WGBS from {args.data_dir} ...")
    ds = hm.load_finaleme_dataset(
        args.data_dir,
        locus_panel=args.locus_panel,
        max_samples=args.samples,
        max_loci=args.loci,
        manifest=args.manifest,
        buffy_coat_bw=args.buffy_coat_bw,
    )
    S, L = ds.beta_sample.shape
    print(f"Loaded {S} paired samples x {L} loci")
    # Two-group split: even/odd sample index. Without explicit case/control
    # labels we report by the natural sample partition for DMR F1.
    idx_a = np.arange(0, S, 2)
    idx_b = np.arange(1, S, 2)
    if len(idx_a) == 0 or len(idx_b) == 0:
        idx_a = np.arange(max(1, S // 2))
        idx_b = np.arange(max(1, S // 2), S)
    rng = np.random.default_rng(args.seed)
    results = hm.evaluate_real_data_benchmark(
        bags=ds.bags,
        n_frag=ds.n,
        truth_beta=ds.beta_sample,
        truth_split=(idx_a, idx_b),
        rng=rng,
    )

    # Re-run predictions (cheap) to dump per-locus per-sample for the scatter
    # figure. evaluate_real_data_benchmark returns metrics only; the
    # underlying predictors are the same ones it calls internally.
    from havi_methyl.baseline import finaleme_baseline_predict
    from havi_methyl.pipeline import _ablation_predict

    pred_baseline, _fit = finaleme_baseline_predict(ds.bags, ds.n)
    pred_full, _ = _ablation_predict(pred_baseline, ds.n, use_hierarchy=True, use_full_iter=True)
    pred_no_flow, _ = _ablation_predict(
        pred_baseline, ds.n, use_hierarchy=True, use_full_iter=False
    )
    pred_no_hier, _ = _ablation_predict(
        pred_baseline, ds.n, use_hierarchy=False, use_full_iter=False
    )

    pred_torch_mean = None
    pred_torch_std = None
    if args.torch_svi:
        from havi_methyl.torch_svi import (
            TorchSVIConfig,
            fit_svi_torch,
            predict_with_torch_state,
        )

        # Infer in_dim from the first non-empty bag (Liu 2024 has 3 features
        # per fragment: length, strand_signed, buffy_coat_prior).
        in_dim = 3
        for sample_bags in ds.bags:
            for bag in sample_bags:
                if bag.shape[0] > 0:
                    in_dim = bag.shape[1]
                    break
            if in_dim != 3 or any(b.shape[0] > 0 for b in sample_bags):
                break

        cfg = TorchSVIConfig(
            in_dim=in_dim,
            hidden=args.torch_hidden,
            posterior=args.torch_posterior,
            k_iwae=args.torch_iwae_k,
            batch_samples=min(args.torch_batch_samples, S),
            batch_loci=min(args.torch_batch_loci, L),
            device=args.torch_device,
        )
        print(
            f"Training fit_svi_torch on real Liu 2024: in_dim={in_dim}, "
            f"posterior={cfg.posterior}, k_iwae={cfg.k_iwae}, "
            f"n_iter={args.torch_iter}, S={S}, L={L} ..."
        )
        # Critical: for real cfDNA data the Beta-Binomial trials are the
        # WGBS read coverage (ds.n_total), NOT the WGS fragment count
        # (ds.n). They are two separate observation streams; conflating
        # them would make the BB likelihood interpret "k WGBS-methylated
        # reads out of N WGS-fragments" which is semantically broken and
        # causes the model to collapse toward β ≈ 0.
        state = fit_svi_torch(
            ds.bags,
            ds.n,
            ds.n_meth,
            n_iter=args.torch_iter,
            config=cfg,
            seed=args.seed,
            n_obs=ds.n_total,
            snapshot_every=args.torch_snapshot_every,
        )
        pred_torch_mean, pred_torch_std = predict_with_torch_state(state, ds.bags, ds.n)
        print(
            f"  trained; pred mean range [{pred_torch_mean.min():.3f}, "
            f"{pred_torch_mean.max():.3f}], "
            f"std mean {pred_torch_std.mean():.3f}"
        )

        # Add the torch row to the metric table.
        from havi_methyl.pipeline import BenchmarkResult
        from havi_methyl.utils import (
            auc_threshold,
            dmr_f1,
            interval_ece,
        )
        from havi_methyl.utils import (
            icc_2_1 as _icc,
        )
        from havi_methyl.utils import (
            pearson_r as _pearson,
        )
        from havi_methyl.utils import (
            spearman_r as _spearman,
        )

        z90 = 1.6448536269514722
        widths = 2 * z90 * pred_torch_std
        truth = ds.beta_sample
        results["HAVI-Methyl (full torch)"] = BenchmarkResult(
            pearson_r=float(_pearson(truth, pred_torch_mean)),
            spearman_r=float(_spearman(truth, pred_torch_mean)),
            auc_meth_at_0p5=float(auc_threshold(truth, pred_torch_mean, 0.5)),
            ece_credible=float(interval_ece(truth, pred_torch_mean, widths)),
            icc_2_1=float(_icc(np.stack([truth.flatten(), pred_torch_mean.flatten()], axis=1))),
            dmr_f1=float(
                dmr_f1(
                    truth[idx_a],
                    truth[idx_b],
                    pred_torch_mean[idx_a],
                    pred_torch_mean[idx_b],
                )
            ),
        )

    pred_npz = "outputs/finaleme_realdata_predictions.npz"
    npz_payload = dict(
        truth=ds.beta_sample,
        pred_finaleme_hmm=pred_baseline,
        pred_havi_full=pred_full,
        pred_havi_no_flow=pred_no_flow,
        pred_havi_no_hier=pred_no_hier,
        n_frag=ds.n,
    )
    if pred_torch_mean is not None:
        npz_payload["pred_havi_torch"] = pred_torch_mean
        npz_payload["std_havi_torch"] = pred_torch_std
        # Save the per-iteration ELBO trajectory so the training-curve
        # figure can replace the flat numpy-SVI placeholder.
        if hasattr(state, "elbo_history") and state.elbo_history:
            npz_payload["torch_elbo_history"] = np.asarray(state.elbo_history, dtype=np.float64)
        if hasattr(state, "recentering_history") and state.recentering_history:
            npz_payload["torch_recentering_history"] = np.asarray(
                state.recentering_history, dtype=np.float64
            )
        if hasattr(state, "snapshots") and state.snapshots:
            # state.snapshots: list of (iter, pred_mean) tuples saved during training.
            iters_, preds_ = zip(*state.snapshots, strict=False)
            npz_payload["torch_snapshot_iters"] = np.asarray(iters_, dtype=np.int64)
            npz_payload["torch_snapshot_preds"] = np.stack(preds_, axis=0)
    np.savez(pred_npz, **npz_payload)
    print(f"Wrote {pred_npz} for the per-locus scatter figure")

    status = (
        f"Liu 2024 (data-dir={args.data_dir}, n_samples={S}, n_loci={L}, "
        f"locus_panel={args.locus_panel or 'default-grid'}, "
        f"manifest={args.manifest or 'filename-stripping'}, "
        f"buffy_coat_prior={'on' if args.buffy_coat_bw else 'off'})"
    )
    return results, ds.n, ds.beta_sample, status, {"S": S, "L": L}


def main() -> None:
    parser = _common.base_parser("FinaleMe benchmark (Sec. 12).")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--loci", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Path to the Liu 2024 FinaleMe directory containing frag_wgs/ "
            "and meth_wgbs/. When set, runs against real data instead of "
            "the synthetic proxy."
        ),
    )
    parser.add_argument(
        "--locus-panel",
        type=str,
        default=None,
        help="Optional BED file giving the panel of (chrom, start, end) loci to score.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Path to the Liu 2024 Supplementary Table 1 CSV pairing "
            "WGS_library_id, WGBS_library_id, and patient_id. Required to "
            "pair the on-disk frag_wgs/ and meth_wgbs/ samples on the lab "
            "drive (filename-only stripping does not pair these)."
        ),
    )
    parser.add_argument(
        "--buffy-coat-bw",
        type=str,
        default=None,
        help=(
            "Path to wgbs_buffyCoat_jensen2015GB.methy.hg19.bw. When set, "
            "each fragment gets an extra ``buffy_coat_prior`` feature equal "
            "to the per-locus mean methylation in the buffy-coat reference, "
            "matching FinaleMe's methylation-prior input."
        ),
    )
    parser.add_argument(
        "--torch-svi",
        action="store_true",
        help=(
            "Run the FULL HAVI-Methyl torch SVI loop "
            "(Set Transformer + posterior head + Beta-Binomial reconstruction) "
            "on top of the simplified pipeline, adding a "
            "'HAVI-Methyl (full torch)' row to the metric table. Requires "
            "torch installed (`make install-torch`)."
        ),
    )
    parser.add_argument("--torch-iter", type=int, default=80)
    parser.add_argument(
        "--torch-posterior",
        type=str,
        choices=("gaussian", "flow"),
        default="gaussian",
        help="Posterior head: gaussian (stable) or flow (Conditional NSF stack).",
    )
    parser.add_argument("--torch-iwae-k", type=int, default=4)
    parser.add_argument("--torch-hidden", type=int, default=32)
    parser.add_argument("--torch-batch-samples", type=int, default=8)
    parser.add_argument("--torch-batch-loci", type=int, default=64)
    parser.add_argument(
        "--torch-device",
        type=str,
        default="auto",
        help="Device for fit_svi_torch: auto, cpu, cuda, cuda:0, mps, ...",
    )
    parser.add_argument(
        "--torch-snapshot-every",
        type=int,
        default=0,
        help=(
            "If > 0, snapshot per-locus predictions every K iters during "
            "fit_svi_torch and save them to the predictions npz so the "
            "training-curve figure can compute per-iteration Pearson r."
        ),
    )
    args = parser.parse_args()

    if not args.data_dir:
        existing = "outputs/tables/bench_finaleme_realdata.csv"
        try:
            with open(existing) as f:
                head = f.read(4096)
            if "Liu 2024" in head:
                print(
                    f"Real-data CSV at {existing} already carries Liu 2024 numbers; "
                    f"skipping synthetic-proxy overwrite. Re-run with --data-dir to refresh."
                )
                return
        except FileNotFoundError:
            pass

    if args.data_dir:
        results, _n, _beta, status, _ = _run_real_data(args)
    else:
        results, _n, _beta, status, _ = _run_synthetic_proxy(args)

    rows = [r.as_row(name, status) for name, r in results.items()]
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
                "_status": f"external baseline {ext} requires its own codebase; placeholder",
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
