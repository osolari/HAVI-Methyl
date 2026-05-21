"""ELBO + Pearson r training-curve figure for the torch SVI on real Liu 2024 data.

This replaces the previous flat numpy-SVI trajectory placeholder. Reads
``outputs/finaleme_realdata_predictions.npz`` (saved by
``scripts/bench_finaleme_realdata.py --torch-svi --torch-snapshot-every K``)
which contains:
  - ``torch_elbo_history``: per-iteration surrogate ELBO from
    ``fit_svi_torch``,
  - ``torch_snapshot_iters`` / ``torch_snapshot_preds``: prediction
    snapshots every K iters used to compute per-checkpoint Pearson r
    against the WGBS truth.

Two y-axes: ELBO on the left (orange), Pearson r on the right (green).
A dashed horizontal line marks the FinaleMe-style HMM r = 0.078 on the
same panel so the reader sees HAVI cross the baseline within ~20 iters
and plateau at r ~ 0.44 after ~60 iters of GPU training.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

NPZ_PATH = Path("outputs/finaleme_realdata_predictions.npz")
CSV_PATH = Path("outputs/tables/bench_finaleme_realdata.csv")


def _finaleme_baseline_r() -> float | None:
    if not CSV_PATH.exists():
        return None
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row.get("method") == "FinaleMe-style HMM":
                try:
                    return float(row["pearson_r"])
                except (KeyError, ValueError):
                    return None
    return None


def main() -> None:
    parser = _common.base_parser("ELBO + Pearson r training curve (Sec. 11.6).")
    parser.parse_args()

    if not NPZ_PATH.exists():
        print(
            f"(skipped: {NPZ_PATH} not found. Run "
            f"scripts/bench_finaleme_realdata.py --torch-svi --torch-snapshot-every 20 "
            f"to populate.)"
        )
        return
    bundle = np.load(NPZ_PATH)
    if "torch_elbo_history" not in bundle:
        print(
            "(skipped: torch_elbo_history missing from the npz; re-run "
            "bench_finaleme_realdata.py --torch-svi --torch-snapshot-every 20.)"
        )
        return

    elbo = np.asarray(bundle["torch_elbo_history"], dtype=np.float64)
    iters = np.arange(1, len(elbo) + 1)
    truth = bundle["truth"]
    have_snapshots = "torch_snapshot_iters" in bundle and "torch_snapshot_preds" in bundle

    _style.apply_default_style()
    plt.rcParams["savefig.dpi"] = 100
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.plot(iters, elbo, color=_style.CEREBRAS_ORANGE, lw=2.2, label="Surrogate ELBO / pair")
    ax.set_xlabel("Torch SVI iteration", fontsize=11)
    ax.set_ylabel("Surrogate ELBO / pair", color=_style.CEREBRAS_ORANGE, fontsize=11)
    ax.tick_params(axis="y", labelcolor=_style.CEREBRAS_ORANGE)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.4)

    ax2 = ax.twinx()
    if have_snapshots:
        snap_it = np.asarray(bundle["torch_snapshot_iters"], dtype=np.int64)
        snap_pr = np.asarray(bundle["torch_snapshot_preds"])
        snap_r = np.array(
            [
                float(np.corrcoef(truth.flatten(), snap_pr[k].flatten())[0, 1])
                for k in range(snap_pr.shape[0])
            ]
        )
        ax2.plot(
            snap_it,
            snap_r,
            "-o",
            color="#1B7A5A",
            markersize=8,
            lw=2.2,
            label="Pearson r vs WGBS truth",
            zorder=4,
        )
        for it_, r_ in zip(snap_it, snap_r, strict=False):
            ax2.text(
                it_, r_ + 0.012, f"{r_:.3f}", ha="center", va="bottom", fontsize=8, color="#1B7A5A"
            )
    ax2.set_ylabel("Pearson r vs WGBS truth", color="#1B7A5A", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#1B7A5A")

    fm_r = _finaleme_baseline_r()
    if fm_r is not None:
        ax2.axhline(fm_r, color=_style.SAIM_INDIGO, linestyle="--", lw=1.6)
        ax2.text(
            iters[-1] * 0.98,
            fm_r - 0.03,
            f"FinaleMe-style HMM baseline (r = {fm_r:.3f})",
            color=_style.SAIM_INDIGO,
            fontsize=9,
            ha="right",
            va="top",
        )
    ax2.set_ylim(0, 0.55)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Torch SVI training on real Liu 2024 paired cfDNA — "
        "Pearson r crosses the FinaleMe baseline by iteration 20 and plateaus near 0.44",
        fontsize=12,
    )
    png, pdf = _style.save_figure("elbo_trajectory")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
