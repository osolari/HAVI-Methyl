"""Regenerate ``docs/report/figures/recovery_scatter.png`` (Sec. 11.1, Fig.~\\ref{fig:recovery}).

Per-locus scatter of true vs. predicted beta at each coverage. The FinaleMe-style
HMM baseline is shown in gray, HAVI-Methyl in Cerebras orange. Both methods
converge to the diagonal as coverage increases.
"""

from __future__ import annotations

import _common  # type: ignore
import _style  # type: ignore
import havi_methyl as hm
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = _common.base_parser("Recovery scatter figure (Sec. 11.1).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    parser.add_argument("--samples-plot", type=int, default=2000)
    args = parser.parse_args()

    _style.apply_default_style()
    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    n_iter = _common.small_or_full(args.fast, full=(10,), fast_values=(3,))[0]

    rng = np.random.default_rng(args.seed)
    n_cov = len(args.coverages)
    fig, axes = plt.subplots(1, n_cov, figsize=(4 * n_cov, 4), sharey=True)
    if n_cov == 1:
        axes = [axes]
    for ax, cov in zip(axes, args.coverages, strict=False):
        result, _ = hm.run_one_coverage(S, L, cov, rng=rng, n_iter=n_iter)
        true = result.plot_data["true"]
        pred_b = result.plot_data["pred_b"]
        pred_h = result.plot_data["pred_h"]
        n = len(true)
        idx = rng.choice(n, size=min(args.samples_plot, n), replace=False)
        ax.scatter(
            true[idx], pred_b[idx], s=2, alpha=0.3, label="FinaleMe-HMM", color=_style.NEUTRAL_GRAY
        )
        ax.scatter(
            true[idx],
            pred_h[idx],
            s=2,
            alpha=0.4,
            label="HAVI-Methyl",
            color=_style.CEREBRAS_ORANGE,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(rf"{cov}$\times$ coverage")
        ax.set_xlabel(r"True $\beta$")
    axes[0].set_ylabel(r"Predicted $\beta$")
    axes[0].legend(loc="upper left", fontsize=8, markerscale=3)
    plt.tight_layout()
    png, pdf = _style.save_figure("recovery_scatter")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
