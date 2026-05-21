"""Sec. 12.3 nested ablation matrix (Phase 2 in IMPLEMENTATION_ROADMAP.md).

Runs every configuration A0..A5 from ``tab_ablations.csv`` against the
synthetic FinaleMe-proxy dataset and emits real metrics for each row:

  A0. Feature regression scaffold       — baseline finaleme_baseline_predict
  A1. + Beta-Binomial pseudo-likelihood — Beta-Binomial posterior MAP per (s, l)
  A2. + hierarchical SVI                — fit_svi_simplified on top of A1
  A3. + amortized flow posterior        — fit_svi_torch (Gaussian head, IMPL-04 follow-up)
  A4. + leakage-control terms           — A3 with VIB + counterfactual swap losses
  A5. + conformal wrapper (full)        — A4 wrapped with split-conformal interval

Every row's ``_status`` column flags this as a synthetic FinaleMe-proxy
run; replace the simulator with the EGA loader (Phase 5) for real numbers.
"""

from __future__ import annotations

import _common  # type: ignore
import numpy as np

import havi_methyl as hm


def _split_conformal_intervals(
    pred_calib: np.ndarray,
    std_calib: np.ndarray,
    truth_calib: np.ndarray,
    pred_test: np.ndarray,
    std_test: np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = hm.gaussian_conformal_intervals(
        mean_calib=pred_calib.flatten(),
        std_calib=std_calib.flatten(),
        y_calib=truth_calib.flatten(),
        mean_test=pred_test.flatten(),
        std_test=std_test.flatten(),
        alpha=alpha,
    )
    return lo.reshape(pred_test.shape), hi.reshape(pred_test.shape)


def _bb_posterior_map(n_meth: np.ndarray, n_total: np.ndarray, kappa: float) -> np.ndarray:
    """Beta-Binomial posterior mean with kappa concentration prior (alpha=gamma=kappa/2)."""
    alpha = kappa / 2.0
    return (n_meth + alpha) / (n_total + 2 * alpha)


def _metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    std: np.ndarray | None,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    interval: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    out = {
        "pearson_r": float(hm.pearson_r(truth, pred)),
        "spearman_r": float(hm.spearman_r(truth, pred)),
        "auc_meth_at_0p5": float(hm.auc_threshold(truth, pred, 0.5)),
        "icc_2_1": float(hm.icc_2_1(np.stack([truth.flatten(), pred.flatten()], axis=1))),
        "dmr_f1": float(hm.dmr_f1(truth[idx_a], truth[idx_b], pred[idx_a], pred[idx_b])),
        "ece_credible": float("nan"),
        "coverage_90": float("nan"),
        "mean_width_90": float("nan"),
    }
    if std is not None:
        z90 = 1.6448536269514722
        widths = 2 * z90 * std
        out["ece_credible"] = float(hm.interval_ece(truth, pred, widths))
    if interval is not None:
        lo, hi = interval
        out["coverage_90"] = float(hm.empirical_coverage(truth, lo, hi))
        out["mean_width_90"] = float(hm.mean_interval_width(lo, hi))
    return out


def main() -> None:
    parser = _common.base_parser("Sec. 12.3 nested ablation matrix.")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--loci", type=int, default=120)
    parser.add_argument("--coverage", type=float, default=2.0)
    parser.add_argument("--torch-iter", type=int, default=60)
    args = parser.parse_args()

    if hm.fit_svi_torch is None:  # pragma: no cover
        raise SystemExit("torch is required for the ablation matrix")

    rng = np.random.default_rng(args.seed)
    sim = hm.simulate_dataset(args.samples, args.loci, args.coverage, rng=rng)
    truth = sim.beta_sample
    S, L = truth.shape

    # Two-group split for DMR F1 (case = first half).
    idx_a = np.arange(S // 2)
    idx_b = np.arange(S // 2, S)
    # Inject DMR signal so the metric is exercised.
    n_dmr = max(1, L // 10)
    dmr_idx = rng.choice(L, size=n_dmr, replace=False)
    truth[idx_a[:, None], dmr_idx] = np.clip(truth[idx_a[:, None], dmr_idx] + 0.4, 0.02, 0.98)
    for s in idx_a:
        for ell in dmr_idx:
            n = int(sim.n[s, ell])
            feats, z = hm.sample_fragment_bag(float(truth[s, ell]), n, rng=rng)
            sim.bags[s][ell] = feats
            sim.n_meth[s, ell] = int(z.sum())

    # Calibration / test split: half of the loci for each (Phase 2 step 2).
    cal_loci = rng.choice(L, size=L // 2, replace=False)
    test_loci = np.array([i for i in range(L) if i not in set(cal_loci)])
    test_truth = truth[:, test_loci]
    test_idx_a = idx_a
    test_idx_b = idx_b

    rows = []
    status = (
        f"synthetic FinaleMe-proxy ablation (S={S}, L={L}, coverage={args.coverage}x); "
        "replace with EGA Liu 2024 loader for real-data values"
    )

    # ---------------- A0: Feature regression scaffold ----------------
    pred_baseline, fit = hm.finaleme_baseline_predict(sim.bags, sim.n)
    rows.append(
        {
            "configuration": "A0. Feature regression scaffold",
            **_metrics(
                test_truth,
                pred_baseline[:, test_loci],
                std=np.full_like(pred_baseline[:, test_loci], 0.1),
                idx_a=test_idx_a,
                idx_b=test_idx_b,
            ),
            "_status": status,
        }
    )

    # ---------------- A1: + Beta-Binomial pseudo-likelihood ----------------
    pred_a1 = _bb_posterior_map(sim.n_meth.astype(float), sim.n.astype(float), kappa=20.0)
    rows.append(
        {
            "configuration": "A1. + Beta-Binomial pseudo-likelihood",
            **_metrics(
                test_truth,
                pred_a1[:, test_loci],
                std=np.full_like(pred_a1[:, test_loci], 0.1),
                idx_a=test_idx_a,
                idx_b=test_idx_b,
            ),
            "_status": status,
        }
    )

    # ---------------- A2: + hierarchical SVI (simplified) ----------------
    state_a2 = hm.fit_svi_simplified(pred_a1, sim.n, n_iter=10)
    pred_a2, std_a2 = hm.predict_with_state(state_a2, pred_a1, sim.n)
    rows.append(
        {
            "configuration": "A2. + hierarchical SVI",
            **_metrics(
                test_truth,
                pred_a2[:, test_loci],
                std=std_a2[:, test_loci],
                idx_a=test_idx_a,
                idx_b=test_idx_b,
            ),
            "_status": status,
        }
    )

    # ---------------- A3: + amortized flow posterior (Gaussian head) ----------------
    cfg_a3 = hm.TorchSVIConfig(
        in_dim=5,
        hidden=16,
        num_inducing=8,
        num_layers=1,
        batch_samples=min(S, 4),
        batch_loci=min(L, 32),
    )
    state_a3 = hm.fit_svi_torch(
        sim.bags, sim.n, sim.n_meth, n_iter=args.torch_iter, config=cfg_a3, seed=args.seed
    )
    pred_a3, std_a3 = hm.predict_with_torch_state(state_a3, sim.bags, sim.n, n_samples=8)
    rows.append(
        {
            "configuration": "A3. + amortized flow posterior",
            **_metrics(
                test_truth,
                pred_a3[:, test_loci],
                std=std_a3[:, test_loci],
                idx_a=test_idx_a,
                idx_b=test_idx_b,
            ),
            "_status": status,
        }
    )

    # ---------------- A4: + leakage-control terms ----------------
    cfg_a4 = hm.TorchSVIConfig(
        in_dim=5,
        hidden=16,
        num_inducing=8,
        num_layers=1,
        batch_samples=min(S, 4),
        batch_loci=min(L, 32),
        vib_weight=0.05,
        counterfactual_weight=0.1,
    )
    state_a4 = hm.fit_svi_torch(
        sim.bags, sim.n, sim.n_meth, n_iter=args.torch_iter, config=cfg_a4, seed=args.seed
    )
    pred_a4, std_a4 = hm.predict_with_torch_state(state_a4, sim.bags, sim.n, n_samples=8)
    rows.append(
        {
            "configuration": "A4. + leakage-control terms",
            **_metrics(
                test_truth,
                pred_a4[:, test_loci],
                std=std_a4[:, test_loci],
                idx_a=test_idx_a,
                idx_b=test_idx_b,
            ),
            "_status": status,
        }
    )

    # ---------------- A5: + conformal wrapper (full) ----------------
    cal_pred = pred_a4[:, cal_loci]
    cal_std = std_a4[:, cal_loci]
    cal_truth = truth[:, cal_loci]
    test_pred = pred_a4[:, test_loci]
    test_std = std_a4[:, test_loci]
    lo, hi = _split_conformal_intervals(
        cal_pred, cal_std + 1e-3, cal_truth, test_pred, test_std + 1e-3, alpha=0.10
    )
    a5_metrics = _metrics(
        test_truth,
        test_pred,
        std=test_std,
        idx_a=test_idx_a,
        idx_b=test_idx_b,
        interval=(lo, hi),
    )
    rows.append(
        {"configuration": "A5. + conformal wrapper (full)", **a5_metrics, "_status": status}
    )

    out = _common.write_csv("outputs/tables/bench_ablation_matrix.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    for r in rows:
        cov_str = f"  cov90={r['coverage_90']:.3f}" if "coverage_90" in r else ""
        print(
            f"  {r['configuration']:<42s} r={r['pearson_r']:.3f}  "
            f"AUC={r['auc_meth_at_0p5']:.3f}  DMR F1={r['dmr_f1']:.3f}{cov_str}"
        )


if __name__ == "__main__":
    main()
