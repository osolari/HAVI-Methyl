"""Phase 5 FinaleMe paired per-locus prediction scatter.

Reads ``outputs/finaleme_realdata_predictions.npz`` (saved by
``scripts/bench_finaleme_realdata.py`` whenever it runs in real-data
mode) and renders a 2x2 panel of predicted vs WGBS-truth scatters,
one panel per method. The npz only exists after a real-data run --
on a fresh checkout, run::

    python scripts/bench_finaleme_realdata.py \\
      --data-dir "/Volumes/Omid Solari/finaleme" \\
      --manifest data/finaleme_manifest/sample_pairs.csv \\
      --locus-panel data/finaleme_manifest/high_variance_cpgs.hg19.bed \\
      --buffy-coat-bw "/Volumes/Omid Solari/finaleme/wgbs_buffyCoat_jensen2015GB.methy.hg19.bw"

first.
"""

from __future__ import annotations

from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

NPZ_PATH = Path("outputs/finaleme_realdata_predictions.npz")

PANELS = [
    ("pred_finaleme_hmm", "FinaleMe-style HMM", _style.SAIM_INDIGO),
    ("pred_havi_no_hier", "HAVI simplified\n(no hierarchy)", "#9CA3AF"),
    ("pred_havi_no_flow", "HAVI simplified\n(no flow)", "#9CA3AF"),
    ("pred_havi_full", "HAVI simplified\n(full)", "#9CA3AF"),
    ("pred_havi_torch", "HAVI-Methyl\n(full torch)", _style.CEREBRAS_ORANGE),
]


def main() -> None:
    parser = _common.base_parser("FinaleMe paired prediction scatter (Phase 5).")
    parser.add_argument(
        "--max-points",
        type=int,
        default=8000,
        help="Subsample to keep the scatter readable; per-panel cap.",
    )
    args = parser.parse_args()

    if not NPZ_PATH.exists():
        raise SystemExit(
            f"{NPZ_PATH} not found. Run scripts/bench_finaleme_realdata.py "
            f"in real-data mode (--data-dir ... --manifest ... --buffy-coat-bw ...) first."
        )
    bundle = np.load(NPZ_PATH)
    truth = bundle["truth"].flatten()
    rng = np.random.default_rng(args.seed)
    if truth.size > args.max_points:
        idx = rng.choice(truth.size, size=args.max_points, replace=False)
    else:
        idx = np.arange(truth.size)
    truth_s = truth[idx]

    # Filter PANELS to those actually present in the npz.
    panels_present = [(k, lab, col) for (k, lab, col) in PANELS if k in bundle]
    n = len(panels_present)
    # 2 rows x 3 cols layout (5 panels + 1 unused -> hide last) when 5 panels,
    # 1 row x N when N <= 4.
    if n >= 5:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 1, n
    _style.apply_default_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 5.0 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes).flatten()
    for ax_idx, (ax, (key, label, color)) in enumerate(zip(axes[:n], panels_present, strict=False)):
        pred = bundle[key].flatten()[idx]
        r = float(np.corrcoef(truth_s, pred)[0, 1])
        # Hexbin shows density better than scatter for tens of thousands of points.
        ax.hexbin(
            truth_s,
            pred,
            gridsize=40,
            cmap="Oranges" if color == _style.CEREBRAS_ORANGE else "Greys",
            mincnt=1,
            extent=(0, 1, 0, 1),
        )
        ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        title_color = _style.CEREBRAS_ORANGE if color == _style.CEREBRAS_ORANGE else "black"
        title_weight = "bold" if color == _style.CEREBRAS_ORANGE else "normal"
        ax.set_title(
            f"{label}\nr = {r:.3f}",
            fontsize=10,
            color=title_color,
            fontweight=title_weight,
        )
        in_last_row = ax_idx >= n - ncols
        if in_last_row:
            ax.set_xlabel("WGBS truth β")
        if ax_idx % ncols == 0:
            ax.set_ylabel("Predicted β")
        ax.grid(linestyle=":", linewidth=0.4, alpha=0.4)
        ax.set_aspect("equal", adjustable="box")
    # Hide any unused axes (e.g. the 6th cell of the 2x3 grid when n=5).
    for extra_ax in axes[n:]:
        extra_ax.axis("off")

    S, L = bundle["truth"].shape if bundle["truth"].ndim == 2 else (None, None)
    summary = ""
    if S is not None:
        summary = (
            f"  (n_samples={S}, n_loci={L}, {min(args.max_points, S * L)} of {S * L} points shown)"
        )
    fig.suptitle(
        f"Liu 2024 paired cfDNA WGS → WGBS prediction{summary}\n"
        f"Hexbin density; diagonal y=x dashed; HAVI full torch spans the full range",
        fontsize=13,
    )
    png, pdf = _style.save_figure("finaleme_paired_scatter")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
