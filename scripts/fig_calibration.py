"""Calibration figure (Sec. 11.3): multi-coverage reliability diagram +
miscalibration inset + conformal wrapper target.

Reads ``outputs/plot_data.npz`` (synthetic recovery experiment) and the
A5 conformal-wrapper row of ``outputs/tables/bench_ablation_matrix.csv``.

Layout:
  Left panel (large): reliability diagram, one curve per coverage shaded
    from light to dark, ideal diagonal, conformal star at (0.90, 0.867),
    legend with per-coverage miscalibration error.
  Right panel (inset bar): mean absolute deviation from ideal vs
    coverage, with a horizontal line at the conformal post-wrapper
    deviation |0.90 - 0.867| ~ 0.033.

This replaces the previous primitive 4-panel layout. The story:
HAVI-Methyl's raw posterior sharpens monotonically as coverage rises
from 0.1x to 5x; at 30x the posterior actually over-shrinks (narrow
intervals miss faraway truth, so empirical coverage drops); the
conformal wrapper hits the target marginally at any depth.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PLOT_DATA_PATH = Path("outputs/plot_data.npz")
ABLATION_CSV_PATH = Path("outputs/tables/bench_ablation_matrix.csv")


def _load_conformal_target() -> tuple[float, float] | None:
    if not ABLATION_CSV_PATH.exists():
        return None
    with ABLATION_CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if "A5" in row["configuration"]:
                try:
                    return float(row["coverage_90"]), 0.90
                except (ValueError, KeyError):
                    return None
    return None


def main() -> None:
    parser = _common.base_parser("Calibration figure (Sec. 11.3).")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.1, 1.0, 5.0, 30.0])
    args = parser.parse_args()

    if not PLOT_DATA_PATH.exists():
        raise SystemExit(f"{PLOT_DATA_PATH} not found; run scripts/bench_synth_recovery.py first.")
    bundle = np.load(PLOT_DATA_PATH)

    from havi_methyl.calibration import coverage_curve

    nominal = np.linspace(0.05, 0.95, 19)
    conformal = _load_conformal_target()

    miscal_by_cov: dict[float, float] = {}
    empirical_by_cov: dict[float, np.ndarray] = {}
    for cov in args.coverages:
        true = bundle[f"{cov}__true"]
        lo_h = bundle[f"{cov}__lo_h"]
        hi_h = bundle[f"{cov}__hi_h"]
        centers = (lo_h + hi_h) / 2.0
        widths = hi_h - lo_h
        emp = coverage_curve(true, centers, widths, nominal)
        empirical_by_cov[cov] = emp
        miscal_by_cov[cov] = float(np.mean(np.abs(nominal - emp)))

    _style.apply_default_style()
    fig = plt.figure(figsize=(11, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 0.05])
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # --- Left: reliability diagram ---
    # Color gradient over coverages (light to dark orange).
    cmap = mpl.colormaps["YlOrRd"]
    n_cov = len(args.coverages)
    sorted_covs = sorted(args.coverages)
    colors = {cov: cmap(0.30 + 0.55 * (i / max(n_cov - 1, 1))) for i, cov in enumerate(sorted_covs)}

    ax.fill_between(
        [0, 1],
        [0, 1],
        [0, 1],
        color="gray",
        alpha=0.0,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="ideal y=x", zorder=4)

    for cov in sorted_covs:
        emp = empirical_by_cov[cov]
        color = colors[cov]
        ax.fill_between(nominal, nominal, emp, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            nominal,
            emp,
            "-o",
            color=color,
            markersize=5,
            linewidth=2.0,
            label=rf"raw posterior @ {cov}$\times$  (miscal={miscal_by_cov[cov]:.3f})",
        )

    if conformal is not None:
        emp_c, nom_c = conformal
        ax.scatter(
            [nom_c],
            [emp_c],
            marker="*",
            s=300,
            color="#1B7A5A",
            edgecolors="white",
            linewidth=1.2,
            zorder=10,
            label=f"+conformal wrapper (A5)\n  empirical={emp_c:.3f} @ nominal=0.90",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage level $\\alpha$", fontsize=11)
    ax.set_ylabel("Empirical coverage $\\widehat{P}(\\beta \\in C_\\alpha)$", fontsize=11)
    ax.set_title("Reliability diagram (raw posterior + conformal target)", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    # --- Right: miscalibration bar chart per coverage ---
    cov_labels = [f"{c}×" for c in sorted_covs]
    vals = [miscal_by_cov[c] for c in sorted_covs]
    bar_colors = [colors[c] for c in sorted_covs]
    ax_bar.bar(cov_labels, vals, color=bar_colors, edgecolor="white", linewidth=1.0)
    for i, v in enumerate(vals):
        ax_bar.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if conformal is not None:
        wrapper_dev = abs(0.90 - conformal[0])
        ax_bar.axhline(
            wrapper_dev,
            color="#1B7A5A",
            linestyle="--",
            linewidth=1.5,
            label=f"conformal: |0.90−{conformal[0]:.3f}| = {wrapper_dev:.3f}",
        )
        ax_bar.legend(loc="upper right", fontsize=8)
    ax_bar.set_ylabel("Mean |nominal − empirical|", fontsize=10)
    ax_bar.set_xlabel("Coverage", fontsize=10)
    ax_bar.set_title("Miscalibration vs coverage", fontsize=11)
    ax_bar.set_ylim(0, max(max(vals) * 1.25, 0.05))
    ax_bar.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "HAVI-Methyl posterior calibration on synthetic paired data — "
        "raw posterior sharpens with coverage; conformal wrapper holds the target at any depth",
        fontsize=12,
    )
    png, pdf = _style.save_figure("calibration")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
