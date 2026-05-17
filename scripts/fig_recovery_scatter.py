"""Synthetic recovery hexbin (Sec. 11.1).

Reads ``outputs/plot_data.npz`` and renders a 2x4 hexbin grid: rows are
methods (FinaleMe-style HMM in grey, HAVI-Methyl simplified in orange),
columns are fragment-bag coverages. Hexbin density makes the qualitative
collapse of the FinaleMe baseline visible at every coverage while
HAVI-Methyl tracks the diagonal.

Each panel reports Pearson r computed from the same arrays. The
canonical seed-20260429 numbers are: FinaleMe 0.162 / 0.524 / 0.768 /
0.961 vs HAVI-Methyl 0.333 / 0.782 / 0.868 / 0.964 at coverages
0.1x / 1x / 5x / 30x; the same values appear in ``tab_recovery.csv``
and the bootstrap intervals in ``tab_recovery_multiseed.csv``.
"""

from __future__ import annotations

from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

PLOT_DATA_PATH = Path("outputs/plot_data.npz")


def main() -> None:
    parser = _common.base_parser("Recovery hexbin figure (Sec. 11.1).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    args = parser.parse_args()

    if not PLOT_DATA_PATH.exists():
        raise SystemExit(f"{PLOT_DATA_PATH} not found; run scripts/bench_synth_recovery.py first.")
    bundle = np.load(PLOT_DATA_PATH)

    _style.apply_default_style()
    n_cov = len(args.coverages)
    fig, axes = plt.subplots(
        2,
        n_cov,
        figsize=(3.4 * n_cov, 6.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    rows = [
        ("FinaleMe-style HMM", "pred_b", _style.SAIM_INDIGO, "Blues"),
        ("HAVI-Methyl (simplified)", "pred_h", _style.CEREBRAS_ORANGE, "Oranges"),
    ]
    for row_idx, (name, key, accent, cmap) in enumerate(rows):
        for col_idx, cov in enumerate(args.coverages):
            ax = axes[row_idx, col_idx]
            true = bundle[f"{cov}__true"]
            pred = bundle[f"{cov}__{key}"]
            r = float(np.corrcoef(true, pred)[0, 1])
            ax.hexbin(true, pred, gridsize=40, cmap=cmap, mincnt=1, extent=(0, 1, 0, 1))
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            if row_idx == 0:
                ax.set_title(rf"{cov}$\times$ coverage", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(rf"{name}{chr(10)}Predicted $\beta$", fontsize=10, color=accent)
            if row_idx == 1:
                ax.set_xlabel(r"True $\beta$", fontsize=10)
            ax.text(
                0.04,
                0.93,
                rf"$r={r:.3f}$",
                transform=ax.transAxes,
                fontsize=10,
                color=accent,
                fontweight="bold",
                ha="left",
                va="top",
            )
            ax.grid(linestyle=":", linewidth=0.4, alpha=0.4)

    fig.suptitle(
        "Synthetic per-locus recovery — HAVI-Methyl tracks the diagonal at every coverage",
        fontsize=13,
    )
    png, pdf = _style.save_figure("recovery_scatter")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
