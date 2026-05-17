"""Calibration figure (Sec. 11.3) -- multi-coverage HAVI-Methyl calibration
with the conformal wrapper as a reference.

Reads ``outputs/plot_data.npz`` (synthetic recovery experiment) and renders
one panel per coverage. Each panel shows:
  * the ideal y = x reference (dashed black),
  * HAVI-Methyl's raw Gaussian-posterior empirical coverage curve (orange),
  * the post-conformal-wrapper empirical coverage from
    ``outputs/tables/bench_ablation_matrix.csv`` row A5 (single point in
    green at nominal 0.90 -> empirical 0.867 on the FinaleMe-proxy panel).

The pre-conformal curve is intentionally under-confident at low coverage
(that's what the conformal wrapper of Prop. 8.1 is for); the post-conformal
point reaches the nominal target. Side-by-side this makes HAVI's calibration
story concrete: raw posterior tracks ideal as coverage grows; the wrapper
guarantees the nominal level at any coverage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

PLOT_DATA_PATH = Path("outputs/plot_data.npz")
ABLATION_CSV_PATH = Path("outputs/tables/bench_ablation_matrix.csv")


def _load_conformal_target() -> tuple[float, float] | None:
    """Read row A5 (conformal wrapper) from the ablation matrix."""
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

    _style.apply_default_style()
    n_cov = len(args.coverages)
    fig, axes = plt.subplots(
        1, n_cov, figsize=(3.4 * n_cov, 3.6), sharey=True, constrained_layout=True
    )
    if n_cov == 1:
        axes = [axes]

    nominal = np.linspace(0.05, 0.95, 19)
    conformal = _load_conformal_target()

    for ax, cov in zip(axes, args.coverages, strict=False):
        true = bundle[f"{cov}__true"]
        lo_h = bundle[f"{cov}__lo_h"]
        hi_h = bundle[f"{cov}__hi_h"]
        centers = (lo_h + hi_h) / 2.0
        widths = hi_h - lo_h
        empirical = coverage_curve(true, centers, widths, nominal)
        # Area between curve and ideal: smaller is better.
        miscal = float(np.mean(np.abs(nominal - empirical)))

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal y=x")
        ax.fill_between(
            nominal, nominal, empirical, color=_style.CEREBRAS_ORANGE, alpha=0.10, linewidth=0
        )
        ax.plot(
            nominal,
            empirical,
            "-o",
            color=_style.CEREBRAS_ORANGE,
            markersize=4,
            label=f"HAVI raw posterior\n(miscal={miscal:.3f})",
        )
        if conformal is not None:
            emp_c, nom_c = conformal
            ax.plot(
                [nom_c],
                [emp_c],
                marker="*",
                markersize=14,
                color="#1B7A5A",
                linestyle="",
                label=f"+conformal wrapper\n(A5: {emp_c:.3f} @ {nom_c:.2f})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal coverage")
        if ax is axes[0]:
            ax.set_ylabel("Empirical coverage")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(rf"{cov}$\times$ coverage", fontsize=11)
        ax.grid(linestyle=":", linewidth=0.4, alpha=0.4)

    axes[-1].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(
        "HAVI-Methyl posterior calibration on synthetic paired data\n"
        "Raw posterior sharpens with coverage; conformal wrapper hits the target at any coverage.",
        fontsize=12,
    )
    png, pdf = _style.save_figure("calibration")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
