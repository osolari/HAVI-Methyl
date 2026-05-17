"""Phase 5 FinaleMe paired-data metric comparison.

Reads ``outputs/tables/bench_finaleme_realdata.csv`` (real Liu 2024
paired cfDNA WGS + WGBS on the 782-CpG high-variance panel) and renders
a 1x2 figure showing Pearson r (higher is better) and credible-interval
ECE (lower is better) across the four HAVI-Methyl variants and the
FinaleMe-style HMM baseline. External-baseline placeholders (FinaleMe,
DeepCpG, Elastic-net, MethylBERT) are skipped because their values are
``XX`` until their own codebases are wired in.

The absolute Pearson r is modest because Liu 2024's matched WGBS is
shallow (median ~50 CpGs/sample at cov >= 5 on chr19..22) and the
prediction problem is intrinsically hard at this coverage. The
comparative result is the load-bearing one: HAVI-Methyl simplified
beats the FinaleMe-style HMM both on point prediction and on
calibration.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common  # type: ignore
import _style  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("outputs/tables/bench_finaleme_realdata.csv")


def main() -> None:
    parser = _common.base_parser("FinaleMe paired-data comparison (Phase 5).")
    parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found; run scripts/bench_finaleme_realdata.py first.")

    methods: list[str] = []
    pearson: list[float] = []
    ece: list[float] = []
    status = ""
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pearson_r"] == "XX":
                continue
            methods.append(row["method"])
            pearson.append(float(row["pearson_r"]))
            ece.append(float(row["ece_credible"]))
            status = row.get("_status", status)

    method_color = {
        "FinaleMe-style HMM": _style.SAIM_INDIGO,
        "HAVI-Methyl simplified (full)": "#9CA3AF",
        "HAVI-Methyl simplified (no flow)": "#9CA3AF",
        "HAVI-Methyl simplified (no hierarchy)": "#9CA3AF",
        "HAVI-Methyl (full torch)": _style.CEREBRAS_ORANGE,
    }
    colors = [method_color.get(m, _style.PALETTE[0]) for m in methods]

    _style.apply_default_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Wrap long labels for readability.
    short = []
    for m in methods:
        if "full torch" in m:
            short.append("HAVI-Methyl\n(full torch)")
        elif "simplified" in m:
            short.append(m.replace("HAVI-Methyl simplified ", "HAVI simplified\n"))
        else:
            short.append(m)
    x = np.arange(len(methods))

    # Left: Pearson r (higher = better).
    bars = axes[0].bar(x, pearson, color=colors, edgecolor="white", linewidth=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, fontsize=9)
    axes[0].set_ylabel("Pearson r vs WGBS truth")
    axes[0].set_title("Per-locus correlation (higher is better)")
    axes[0].axhline(0.0, color="black", linewidth=0.5)
    axes[0].set_ylim(0, max(pearson) * 1.25)
    axes[0].grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    for b, v in zip(bars, pearson, strict=False):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            v + max(pearson) * 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Right: ECE (lower = better, annotate with arrow).
    bars = axes[1].bar(x, ece, color=colors, edgecolor="white", linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, fontsize=9)
    axes[1].set_ylabel("Credible-interval ECE")
    axes[1].set_title("Calibration (lower is better)")
    axes[1].set_ylim(0, max(ece) * 1.2)
    axes[1].grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    for b, v in zip(bars, ece, strict=False):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            v + max(ece) * 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    short_status = ""
    if status:
        # Extract the n_samples / n_loci summary from the status string.
        bits = [b.strip() for b in status.split(",")]
        keep = [b for b in bits if b.startswith(("n_samples", "n_loci"))]
        short_status = "Liu 2024 paired: " + ", ".join(keep) if keep else "Liu 2024 paired"
    fig.suptitle(f"FinaleMe paired benchmark on real Liu 2024 data\n{short_status}", fontsize=12)

    # Annotate the absolute improvement HAVI makes over FinaleMe.
    havi_r = max(pearson[i] for i, m in enumerate(methods) if "HAVI" in m)
    fm_r = pearson[methods.index("FinaleMe-style HMM")] if "FinaleMe-style HMM" in methods else None
    havi_ece = min(ece[i] for i, m in enumerate(methods) if "HAVI" in m)
    fm_ece = ece[methods.index("FinaleMe-style HMM")] if "FinaleMe-style HMM" in methods else None
    if fm_r is not None and fm_ece is not None:
        delta_r_pct = 100 * (havi_r - fm_r) / max(fm_r, 1e-9)
        delta_ece_pct = 100 * (fm_ece - havi_ece) / max(fm_ece, 1e-9)
        axes[0].text(
            0.5,
            0.97,
            f"HAVI best vs FinaleMe-HMM: +{delta_r_pct:.1f}% relative r",
            transform=axes[0].transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=_style.CEREBRAS_ORANGE,
            weight="bold",
        )
        axes[1].text(
            0.5,
            0.97,
            f"HAVI best vs FinaleMe-HMM: −{delta_ece_pct:.1f}% relative ECE",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=_style.CEREBRAS_ORANGE,
            weight="bold",
        )

    # Honesty caveat: the simplified-numpy rows are FinaleMe + thin SVI
    # shrinkage by construction; the HAVI-Methyl (full torch) row is the
    # actual architectural comparison (Set Transformer encoder +
    # Gaussian posterior head trained 200 iter on CUDA).
    fig.text(
        0.5,
        0.005,
        "Liu 2024 paired cfDNA: HAVI-Methyl (full torch, orange) trained end-to-end with the BB-trials fix "
        "vs FinaleMe-style HMM (indigo). Grey bars are the simplified-numpy ablations.",
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        color=_style.NEUTRAL_GRAY,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.93))
    png, pdf = _style.save_figure("finaleme_paired_metrics")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
