"""Sec. 9 / Sec. 11.5 tissue-of-origin benchmark with leave-one-tissue-out.

Compares four deconvolution methods on a synthetic T-tissue mixture:

  - FinaleMe-binarized + QP deconvolution (baseline; Sec. 11.5).
  - Continuous least-squares deconvolution (current simplified head).
  - HAVI-Methyl variance-weighted Dirichlet head (consumes posterior
    ``(mean, var)``; Sec. 9.1).
  - HDP-truncated stick-breaking deconvolution at T_max=64 (Sec. 9.2).

For each method we report:
  - Tissue-fraction RMSE on the full reference panel.
  - Worst-tissue RMSE on the same panel.
  - Leave-one-tissue-out (LOO) mean and worst RMSE per
    ``leave_one_tissue_out_stress``.

Status remains ``synthetic three-tissue proxy`` (per
docs/report/CODING_AGENT_HANDOFF.md IMPL-08) until a verified atlas
(Loyfer 2023) is wired in. The infrastructure is data-loader ready: swap
the reference matrix for the atlas reference and rerun.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm
import numpy as np


def main() -> None:
    parser = _common.base_parser("Tissue-of-origin LOO benchmark.")
    parser.add_argument("--n-tissues", type=int, default=4)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--loci", type=int, default=200)
    parser.add_argument(
        "--atlas-tsv",
        type=str,
        default=None,
        help=(
            "Path to a precomputed Loyfer-style atlas TSV "
            "(e.g. ecd/data_resources/human_methylome_atlas/human_methylome_atlas.tsv). "
            "When set, the synthetic reference panel is replaced by the real "
            "atlas matrix and pi_true is drawn from a Dirichlet prior."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    T = args.n_tissues
    S = args.samples
    L = args.loci

    if args.atlas_tsv:
        print(f"Loading Loyfer atlas TSV from {args.atlas_tsv} ...")
        atlas = hm.load_loyfer_atlas_matrix(args.atlas_tsv)
        T_full = atlas.n_tissues
        L_full = atlas.n_loci
        # Subsample loci/tissues to the requested shapes.
        loci_idx = rng.choice(L_full, size=min(L, L_full), replace=False)
        tissue_idx = rng.choice(T_full, size=min(T, T_full), replace=False)
        R = atlas.reference[np.ix_(tissue_idx, loci_idx)]
        T, L = R.shape
        tissue_label = f"Loyfer 2023 (T={T}/{T_full}, L={L}/{L_full}, atlas={args.atlas_tsv})"
    else:
        R = rng.uniform(0.0, 1.0, size=(T, L))
        tissue_label = f"synthetic {T}-tissue proxy (S={S}, L={L}, heteroskedastic posterior var)"
    pi_true = rng.dirichlet(np.ones(T), size=S)
    beta_var = np.where(rng.uniform(0, 1, size=L) < 0.3, 0.4, 0.005)
    obs = pi_true @ R + rng.normal(0, np.sqrt(beta_var), size=(S, L))
    obs = np.clip(obs, 0.0, 1.0)
    var_per_sample = np.tile(beta_var, (S, 1))

    # Each method needs both a (Y, R)->pi callable for LOO and an in-panel pi.
    def _hdp(Y, Rmat):
        return hm.hdp_truncated_deconvolve(Y, Rmat, alpha=1.0, T_max=64, rng=rng)[
            :, : Rmat.shape[0]
        ]

    methods: dict[str, tuple[np.ndarray, callable]] = {
        "FinaleMe-binarized + QP": (
            hm.binarize_and_deconvolve(obs, R),
            hm.binarize_and_deconvolve,
        ),
        "Continuous lstsq": (
            hm.deconvolve_least_squares(obs, R),
            hm.deconvolve_least_squares,
        ),
        "HAVI-Methyl Dirichlet head": (
            hm.dirichlet_head_predict(obs, var_per_sample, R),
            lambda Y, Rmat: hm.dirichlet_head_predict(Y, var_per_sample, Rmat),
        ),
        "HDP-truncated (T_max=64)": (_hdp(obs, R), _hdp),
    }

    rows = []
    status = tissue_label + ("; swap --atlas-tsv to use a real atlas" if not args.atlas_tsv else "")
    for name, (pi_pred, deconv_fn) in methods.items():
        per_tissue = np.sqrt(((pi_true - pi_pred) ** 2).mean(axis=0))
        rmse = float(np.sqrt(((pi_true - pi_pred) ** 2).mean()))
        worst = float(per_tissue.max())
        loo = hm.leave_one_tissue_out_stress(pi_true, R, obs, method=deconv_fn)
        rows.append(
            {
                "method": name,
                "tissue_fraction_rmse": rmse,
                "worst_tissue_rmse": worst,
                "loo_mean_rmse": float(loo["mean_rmse"]),
                "loo_worst_rmse": float(loo["worst_rmse"]),
                "_status": status,
            }
        )

    out = _common.write_csv("outputs/tables/bench_tissue_loo.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")
    for r in rows:
        print(
            f"  {r['method']:<35s}  RMSE={r['tissue_fraction_rmse']:.4f}  "
            f"worst={r['worst_tissue_rmse']:.4f}  "
            f"LOO mean={r['loo_mean_rmse']:.4f}  worst={r['loo_worst_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
