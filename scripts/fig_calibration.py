"""Regenerate ``docs/report/figures/calibration.png`` (Sec. 11.3, Fig.~\\ref{fig:calibration}).

Calibration curve at 5x coverage: nominal vs. empirical coverage under the
simplified-Gaussian variant. Production deployment uses the conformal wrapper
of Prop.~\\ref{prop:conformal} which restores marginal coverage.
"""

from __future__ import annotations

import _common  # type: ignore
import _style  # type: ignore
import havi_methyl as hm
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = _common.base_parser("Calibration figure (Sec. 11.3).")
    parser.add_argument("--coverage", type=float, default=5.0)
    args = parser.parse_args()

    _style.apply_default_style()
    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    n_iter = _common.small_or_full(args.fast, full=(10,), fast_values=(3,))[0]

    rng = np.random.default_rng(args.seed)
    result, _ = hm.run_one_coverage(S, L, args.coverage, rng=rng, n_iter=n_iter)

    from havi_methyl.calibration import coverage_curve

    centers = (result.plot_data["lo_h"] + result.plot_data["hi_h"]) / 2.0
    widths = result.plot_data["hi_h"] - result.plot_data["lo_h"]
    nominal = np.linspace(0.05, 0.95, 19)
    empirical = coverage_curve(result.plot_data["true"], centers, widths, nominal)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(nominal, empirical, "-o", color=_style.CEREBRAS_ORANGE, label="HAVI-Methyl")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(rf"Calibration at {args.coverage}$\times$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    plt.tight_layout()
    png, pdf = _style.save_figure("calibration")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
