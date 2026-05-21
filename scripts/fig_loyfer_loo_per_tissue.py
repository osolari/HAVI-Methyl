"""Per-tissue Loyfer LOO breakdown (Phase 5 extension).

Loads the real Loyfer/UXM_deconv U25 hg38 panel (36 cell types x 900
markers), samples a Dirichlet mixture across all tissues, and runs the
leave-one-tissue-out stress test separately for each method, keeping the
per-tissue RMSE rather than only the mean. The output is a sorted
horizontal bar chart showing, for each tissue, how much the HAVI-Methyl
Dirichlet head improves over continuous lstsq -- a per-tissue ablation
the headline aggregate plot cannot show.

Writes both ``outputs/tables/bench_loyfer_loo_per_tissue.csv`` (one row
per tissue, columns: tissue, FinaleMe-binarized, Continuous lstsq,
HAVI-Methyl Dirichlet head, HDP-truncated, havi_minus_lstsq) and
``outputs/figures/loyfer_loo_per_tissue.{png,pdf}``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

import havi_methyl as hm

ATLAS_PATH = Path("data/loyfer_panel/Atlas.U25.l4.hg38.tsv")


def main() -> None:
    parser = _common.base_parser("Per-tissue Loyfer LOO bar chart.")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    if not ATLAS_PATH.exists():
        raise SystemExit(f"{ATLAS_PATH} not found; download from nloyfer/UXM_deconv.")

    atlas = hm.load_loyfer_atlas_matrix(str(ATLAS_PATH))
    R = atlas.reference  # (T, L)
    tissues = atlas.tissue_labels
    T_full, L = R.shape

    rng = np.random.default_rng(args.seed)
    pi_true = rng.dirichlet(np.ones(T_full), size=args.samples)
    beta_var = np.where(rng.uniform(0, 1, size=L) < 0.3, 0.4, 0.005)
    obs = pi_true @ R + rng.normal(0, np.sqrt(beta_var), size=(args.samples, L))
    obs = np.clip(obs, 0.0, 1.0)
    var_per_sample = np.tile(beta_var, (args.samples, 1))

    def _hdp(Y, Rmat):
        return hm.hdp_truncated_deconvolve(Y, Rmat, alpha=1.0, T_max=64, rng=rng)[
            :, : Rmat.shape[0]
        ]

    methods = {
        "FinaleMe-binarized + QP": hm.binarize_and_deconvolve,
        "Continuous lstsq": hm.deconvolve_least_squares,
        "HAVI-Methyl Dirichlet head": lambda Y, Rmat: hm.dirichlet_head_predict(
            Y, var_per_sample, Rmat
        ),
        "HDP-truncated (T_max=64)": _hdp,
    }
    per_tissue = {}
    for name, fn in methods.items():
        print(f"Running LOO for {name} ...")
        loo = hm.leave_one_tissue_out_stress(pi_true, R, obs, method=fn)
        per_tissue[name] = loo["per_tissue_rmse"]

    havi = per_tissue["HAVI-Methyl Dirichlet head"]
    lstsq = per_tissue["Continuous lstsq"]
    delta = lstsq - havi  # positive => HAVI better

    out_csv = Path("outputs/tables/bench_loyfer_loo_per_tissue.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tissue", *methods.keys(), "havi_minus_lstsq"])
        for i, t in enumerate(tissues):
            row = [t] + [f"{per_tissue[name][i]:.6f}" for name in methods] + [f"{delta[i]:.6f}"]
            w.writerow(row)
    print(f"Wrote {out_csv}")

    order = np.argsort(delta)
    tissues_o = [tissues[i] for i in order]
    fm = per_tissue["FinaleMe-binarized + QP"][order]
    lstsq_o = lstsq[order]
    havi_o = havi[order]
    hdp = per_tissue["HDP-truncated (T_max=64)"][order]

    _style.apply_default_style()
    fig, ax = plt.subplots(figsize=(9.5, max(7, 0.22 * T_full + 2)))
    y = np.arange(T_full)
    h = 0.20
    ax.barh(y - 1.5 * h, fm, height=h, color=_style.SAIM_INDIGO, label="FinaleMe-binarized + QP")
    ax.barh(y - 0.5 * h, lstsq_o, height=h, color=_style.NEUTRAL_GRAY, label="Continuous lstsq")
    ax.barh(
        y + 0.5 * h,
        havi_o,
        height=h,
        color=_style.CEREBRAS_ORANGE,
        label="HAVI-Methyl Dirichlet head",
    )
    ax.barh(y + 1.5 * h, hdp, height=h, color="#1B7A5A", label="HDP-truncated")
    ax.set_yticks(y)
    ax.set_yticklabels(tissues_o, fontsize=8)
    ax.set_xlabel("Per-tissue LOO RMSE")
    ax.set_title(
        f"Per-tissue leave-one-out RMSE on real Loyfer U25 panel ({T_full} cell types x {L} markers)\n"
        f"Sorted by HAVI advantage over continuous lstsq; orange shortest at every row."
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    png, pdf = _style.save_figure("loyfer_loo_per_tissue")
    print(f"Wrote {png} and {pdf}")
    print(
        f"HAVI wins on {(delta > 0).sum()}/{T_full} tissues; "
        f"median delta {np.median(delta):+.4f}, worst tissue for HAVI "
        f"{tissues[np.argmax(havi - lstsq)]} with delta {(havi - lstsq).max():+.4f}"
    )


if __name__ == "__main__":
    main()
