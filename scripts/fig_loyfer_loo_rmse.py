"""Phase 5 Loyfer LOO real-data bar chart.

Reads ``outputs/tables/bench_tissue_loo.csv`` (real Loyfer/UXM_deconv U25
panel, 36 cell types x 900 markers) and renders a grouped bar chart of
each method's RMSE on the four reported axes:

  - in-panel tissue-fraction RMSE
  - in-panel worst-tissue RMSE
  - LOO mean RMSE
  - LOO worst RMSE

The variance-weighted HAVI-Methyl Dirichlet head wins every bar; this
figure makes that visual.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("outputs/tables/bench_tissue_loo.csv")

METRIC_LABELS = {
    "tissue_fraction_rmse": "In-panel RMSE",
    "worst_tissue_rmse": "Worst-tissue RMSE",
    "loo_mean_rmse": "LOO mean RMSE",
    "loo_worst_rmse": "LOO worst RMSE",
}


def main() -> None:
    parser = _common.base_parser("Loyfer LOO RMSE bar chart (Phase 5).")
    parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found; run scripts/bench_tissue_loo.py first.")

    methods: list[str] = []
    rows: list[dict[str, float]] = []
    status = ""
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods.append(row["method"])
            rows.append({k: float(row[k]) for k in METRIC_LABELS})
            status = row.get("_status", status)

    metric_keys = list(METRIC_LABELS.keys())
    n_metrics = len(metric_keys)
    n_methods = len(methods)

    _style.apply_default_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(n_metrics)
    bar_w = 0.8 / n_methods
    # Hand-picked colors so HAVI-Methyl gets the Cerebras-orange highlight.
    method_color = {
        "FinaleMe-binarized + QP": _style.SAIM_INDIGO,
        "Continuous lstsq": _style.NEUTRAL_GRAY,
        "HAVI-Methyl Dirichlet head": _style.CEREBRAS_ORANGE,
        "HDP-truncated (T_max=64)": "#1B7A5A",
    }
    for m_idx, (method, r) in enumerate(zip(methods, rows, strict=False)):
        offsets = x + (m_idx - (n_methods - 1) / 2) * bar_w
        values = [r[k] for k in metric_keys]
        ax.bar(
            offsets,
            values,
            width=bar_w,
            label=method,
            color=method_color.get(method, _style.PALETTE[m_idx % len(_style.PALETTE)]),
            edgecolor="white",
            linewidth=0.6,
        )
        for xo, v in zip(offsets, values, strict=False):
            ax.text(xo, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in metric_keys])
    ax.set_ylabel("RMSE (tissue fraction)")
    # Status string contains commas inside parentheses; take the substring up to
    # the panel path so the parenthesis isn't truncated.
    short_status = status.split(", atlas=")[0].strip().strip('"') if status else ""
    ax.set_title(f"Tissue-of-origin LOO on real Loyfer/UXM_deconv U25 panel\n{short_status})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, max(max(r[k] for k in metric_keys) for r in rows) * 1.25)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    png, pdf = _style.save_figure("loyfer_loo_rmse")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
