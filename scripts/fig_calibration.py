"""Regenerate ``docs/report/figures/calibration.png`` (Sec. 11.3,
Fig.~\\ref{fig:calibration}) from the canonical ``outputs/plot_data.npz``.

Calibration curve at the chosen coverage: nominal vs. empirical coverage under
the simplified-Gaussian variant. Production deployment uses the conformal
wrapper of Prop.~\\ref{prop:conformal} which restores marginal coverage.
"""

from __future__ import annotations

from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

PLOT_DATA_PATH = Path("outputs/plot_data.npz")


def main() -> None:
    parser = _common.base_parser("Calibration figure (Sec. 11.3).")
    parser.add_argument("--coverage", type=float, default=5.0)
    args = parser.parse_args()

    if not PLOT_DATA_PATH.exists():
        raise SystemExit(f"{PLOT_DATA_PATH} not found; run scripts/bench_synth_recovery.py first.")
    bundle = np.load(PLOT_DATA_PATH)

    _style.apply_default_style()
    cov = args.coverage
    true = bundle[f"{cov}__true"]
    lo_h = bundle[f"{cov}__lo_h"]
    hi_h = bundle[f"{cov}__hi_h"]

    from havi_methyl.calibration import coverage_curve

    centers = (lo_h + hi_h) / 2.0
    widths = hi_h - lo_h
    nominal = np.linspace(0.05, 0.95, 19)
    empirical = coverage_curve(true, centers, widths, nominal)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(nominal, empirical, "-o", color=_style.CEREBRAS_ORANGE, label="HAVI-Methyl")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(rf"Calibration at {cov}$\times$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    plt.tight_layout()
    png, pdf = _style.save_figure("calibration")
    print(f"Wrote {png} and {pdf} from {PLOT_DATA_PATH}")


if __name__ == "__main__":
    main()
