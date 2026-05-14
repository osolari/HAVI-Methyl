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
    ("pred_havi_full", "HAVI-Methyl simplified (full)", _style.CEREBRAS_ORANGE),
    ("pred_havi_no_flow", "HAVI-Methyl simplified (no flow)", "#1B7A5A"),
    ("pred_havi_no_hier", "HAVI-Methyl simplified (no hierarchy)", _style.NEUTRAL_GRAY),
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

    _style.apply_default_style()
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    for ax, (key, label, color) in zip(axes.flat, PANELS, strict=False):
        pred = bundle[key].flatten()[idx]
        r = float(np.corrcoef(truth_s, pred)[0, 1])
        ax.scatter(truth_s, pred, s=4, alpha=0.25, color=color, edgecolors="none")
        ax.plot([0, 1], [0, 1], color="black", linewidth=0.7, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nPearson r = {r:.3f}", fontsize=10)
        ax.set_xlabel("WGBS truth β")
        ax.set_ylabel("Predicted β")
        ax.grid(linestyle=":", linewidth=0.4, alpha=0.4)

    S, L = bundle["truth"].shape if bundle["truth"].ndim == 2 else (None, None)
    summary = ""
    if S is not None:
        summary = f"  (n_samples={S}, n_loci={L}, {args.max_points} of {S * L} points shown)"
    fig.suptitle(f"Liu 2024 paired cfDNA WGS → WGBS prediction{summary}", fontsize=12)
    plt.tight_layout()
    png, pdf = _style.save_figure("finaleme_paired_scatter")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
