"""Regenerate ``docs/report/figures/elbo_trajectory.png`` (Sec. 11.6, Fig.~\\ref{fig:elbo}).

ELBO-per-pair trajectory at 30x coverage. The negative-SSE proxy used here as
the ELBO surrogate decreases monotonically with iteration, consistent with
Prop.~\\ref{prop:svi}.
"""

from __future__ import annotations

import _common  # type: ignore
import _style  # type: ignore
import havi_methyl as hm
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = _common.base_parser("ELBO trajectory figure (Sec. 11.6).")
    parser.add_argument("--coverage", type=float, default=30.0)
    args = parser.parse_args()

    _style.apply_default_style()
    S, L = _common.small_or_full(args.fast, full=(12, 300), fast_values=(4, 80))
    n_iter = _common.small_or_full(args.fast, full=(10,), fast_values=(3,))[0]

    rng = np.random.default_rng(args.seed)
    result, _ = hm.run_one_coverage(S, L, args.coverage, rng=rng, n_iter=n_iter)
    history = result.plot_data["elbo_history"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(len(history)), history, "-o", color=_style.CEREBRAS_ORANGE, lw=1.6)
    ax.set_xlabel("SVI iteration")
    ax.set_ylabel(r"ELBO / pair")
    ax.set_title(rf"ELBO trajectory at {args.coverage}$\times$")
    plt.tight_layout()
    png, pdf = _style.save_figure("elbo_trajectory")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
