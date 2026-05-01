"""Shared matplotlib styling used by every figure script.

Publication-quality defaults: colorblind-safe palette, mathtext rendering,
tight layout, and Cerebras-orange/SAIM-indigo accents matching the
``main.pdf`` callouts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CEREBRAS_ORANGE = "#F26522"
SAIM_INDIGO = "#321278"
NEUTRAL_GRAY = "#7A7A7A"
PALETTE = [SAIM_INDIGO, CEREBRAS_ORANGE, NEUTRAL_GRAY, "#1B7A5A", "#B0006A"]


def apply_default_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler("color", PALETTE),
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "axes.grid": False,
        }
    )


def output_paths(stem: str) -> tuple[Path, Path]:
    """Return (PNG path, PDF path) under ``outputs/figures``.

    The orchestrator copies the PDF into ``docs/report/figures`` for the
    LaTeX build (Sec. 4 of the project plan in ``repo.md``).
    """
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"


def copy_pdf_to_report(pdf_path: Path) -> Path:
    """Mirror the PDF into ``docs/report/figures``."""
    target = Path("docs/report/figures") / pdf_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf_path.read_bytes())
    return target


def copy_png_to_report(png_path: Path) -> Path:
    """Mirror the PNG into ``docs/report/figures`` (the manuscript's
    ``\\includegraphics`` calls reference PNGs by name)."""
    target = Path("docs/report/figures") / png_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(png_path.read_bytes())
    return target


def save_figure(stem: str) -> tuple[Path, Path]:
    """Save the current figure to outputs/ and mirror to docs/report/figures."""
    png, pdf = output_paths(stem)
    plt.savefig(png)
    plt.savefig(pdf)
    copy_png_to_report(png)
    copy_pdf_to_report(pdf)
    return png, pdf
