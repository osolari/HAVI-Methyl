"""Regenerate ``docs/report/figures/elbo_trajectory.png`` (Sec. 11.6,
Fig.~\\ref{fig:elbo}) from the canonical ``outputs/plot_data.npz``.

Surrogate-objective trajectory at the chosen coverage. The plotted quantity is
the negative-SSE proxy used by the simplified harness, not the full-model
ELBO; per Sec. 11.6 it should not be read as a verification of
Prop.~\\ref{prop:svi} for the full flow-based model.
"""

from __future__ import annotations

from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

PLOT_DATA_PATH = Path("outputs/plot_data.npz")


def main() -> None:
    parser = _common.base_parser("Surrogate objective trajectory figure (Sec. 11.6).")
    parser.add_argument("--coverage", type=float, default=30.0)
    args = parser.parse_args()

    if not PLOT_DATA_PATH.exists():
        raise SystemExit(f"{PLOT_DATA_PATH} not found; run scripts/bench_synth_recovery.py first.")
    bundle = np.load(PLOT_DATA_PATH)

    _style.apply_default_style()
    cov = args.coverage
    history = bundle[f"{cov}__elbo_history"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(len(history)), history, "-o", color=_style.CEREBRAS_ORANGE, lw=1.6)
    ax.set_xlabel("SVI iteration")
    ax.set_ylabel("Surrogate objective / pair")
    ax.set_title(rf"Surrogate trajectory at {cov}$\times$")
    plt.tight_layout()
    png, pdf = _style.save_figure("elbo_trajectory")
    print(f"Wrote {png} and {pdf} from {PLOT_DATA_PATH}")


if __name__ == "__main__":
    main()
