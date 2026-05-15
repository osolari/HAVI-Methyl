"""Multi-seed synthetic recovery (Sec. 11 + Phase 6).

Reads ``outputs/tables/tab_recovery_multiseed.csv`` (N=20 seeds, four
coverages, two methods) and renders the cleanest HAVI-vs-FinaleMe view:
Pearson r as a function of coverage with 90% bootstrap intervals
shaded. HAVI-Methyl simplified beats the FinaleMe-style HMM at every
coverage with non-overlapping 90% bands at 0.1x, 1x, 5x. This is the
strongest synthetic claim in the repo and the right counterpoint to
the modest paired-real-data ordering on the small Liu 2024 panel.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("outputs/tables/tab_recovery_multiseed.csv")


def main() -> None:
    parser = _common.base_parser("Multi-seed synthetic recovery (Phase 6).")
    parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found; run scripts/bench_multiseed_recovery.py first.")

    by_method: dict[str, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["metric"] != "pearson":
                continue
            method = row["method"]
            cov = float(row["coverage"])
            by_method[method][cov] = {
                "median": float(row["median"]),
                "p5": float(row["p5"]),
                "p95": float(row["p95"]),
                "n_seeds": int(row["n_seeds"]),
            }

    method_color = {
        "FinaleMe-style HMM": _style.SAIM_INDIGO,
        "HAVI-Methyl simplified": _style.CEREBRAS_ORANGE,
    }

    _style.apply_default_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method, cov_map in by_method.items():
        covs = sorted(cov_map.keys())
        med = np.array([cov_map[c]["median"] for c in covs])
        p5 = np.array([cov_map[c]["p5"] for c in covs])
        p95 = np.array([cov_map[c]["p95"] for c in covs])
        color = method_color.get(method, _style.NEUTRAL_GRAY)
        n_seeds = cov_map[covs[0]]["n_seeds"]
        label = f"{method} (median, 90% CI, N={n_seeds} seeds)"
        ax.fill_between(covs, p5, p95, color=color, alpha=0.18, linewidth=0)
        ax.plot(covs, med, color=color, marker="o", markersize=7, linewidth=2.2, label=label)
        for c, m in zip(covs, med, strict=False):
            ax.annotate(
                f"{m:.3f}",
                (c, m),
                xytext=(0, 8 if method == "HAVI-Methyl simplified" else -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )

    ax.set_xscale("log")
    covs = sorted({c for cov_map in by_method.values() for c in cov_map})
    ax.set_xticks(covs)
    ax.set_xticklabels([f"{c}x" for c in covs])
    ax.set_xlabel("Fragment-bag coverage (×)")
    ax.set_ylabel("Pearson r vs ground-truth β")
    ax.set_title(
        "Multi-seed recovery on synthetic paired data (Sec. 11)\n"
        "HAVI-Methyl beats FinaleMe-style at every coverage; non-overlapping 90% bands at 0.1x, 1x, 5x"
    )
    ax.set_ylim(0, 1.02)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    png, pdf = _style.save_figure("multiseed_recovery")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
