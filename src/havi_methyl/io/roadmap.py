"""Roadmap Bisulfite-Seq atlas loader.

Each tissue subdirectory under ``Bisulfite-Seq/`` contains one or more
``GSM*_<UCSF>.<tissue>.Bisulfite-Seq.<sample>.wig.gz`` files in the
UCSC variableStep WIG format. Per-CpG methylation values are averaged
across samples within each tissue and aligned onto a user-supplied
locus panel.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RoadmapAtlas:
    reference: np.ndarray  # (T, L)
    tissue_labels: list[str]
    locus_chrom: list[str]
    locus_start: list[int]
    n_samples_per_tissue: dict[str, int]
    source: str


_VARIABLE_STEP_RE = re.compile(r"variableStep\s+chrom=(\S+)(?:\s+span=(\d+))?")


def _iter_wig_records(path: Path):
    """Yield (chrom, position, value) for variableStep WIG records."""
    opener = gzip.open if str(path).endswith(".gz") else open
    chrom = None
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("track"):
                continue
            m = _VARIABLE_STEP_RE.match(line)
            if m is not None:
                chrom = m.group(1)
                continue
            if line.startswith("fixedStep"):
                # Roadmap uses variableStep almost exclusively; skip
                # fixedStep blocks rather than mis-parse them.
                chrom = None
                continue
            if chrom is None:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                yield chrom, int(parts[0]), float(parts[1])
            except ValueError:
                continue


def load_roadmap_wgbs_atlas(
    path: str | Path,
    sites: list[tuple[str, int]],
    tissue_subset: list[str] | None = None,
    max_samples_per_tissue: int | None = None,
) -> RoadmapAtlas:
    """Aggregate Roadmap WGBS WIG files into a (T_tissues, L) reference matrix.

    ``sites`` is the list of CpG positions to evaluate. The reference
    value for tissue ``t`` at site ``(chrom, pos)`` is the mean WGBS
    methylation across samples in that tissue at that exact position.
    Sites with no coverage in a tissue receive that tissue's mean over
    covered sites.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    site_index = {(c, s): i for i, (c, s) in enumerate(sites)}

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    n_samples: dict[str, int] = {}

    for tissue_dir in sorted(root.iterdir()):
        if not tissue_dir.is_dir():
            continue
        tissue = tissue_dir.name
        if tissue_subset is not None and tissue not in tissue_subset:
            continue
        sums.setdefault(tissue, np.zeros(len(sites), dtype=np.float64))
        counts.setdefault(tissue, np.zeros(len(sites), dtype=np.float64))
        for sample_path in sorted(tissue_dir.glob("*.wig.gz")):
            if (
                max_samples_per_tissue is not None
                and n_samples.get(tissue, 0) >= max_samples_per_tissue
            ):
                break
            n_samples[tissue] = n_samples.get(tissue, 0) + 1
            for chrom, pos, value in _iter_wig_records(sample_path):
                idx = site_index.get((chrom, pos))
                if idx is None:
                    continue
                # WIG values are typically in [0, 1]; some Roadmap tracks
                # report percentages — divide if value > 1.
                v = value / 100.0 if value > 1.0 else value
                sums[tissue][idx] += v
                counts[tissue][idx] += 1.0

    tissues = sorted(sums.keys())
    reference = np.zeros((len(tissues), len(sites)), dtype=np.float64)
    for t_idx, tissue in enumerate(tissues):
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = sums[tissue] / np.maximum(counts[tissue], 1.0)
        beta[counts[tissue] == 0] = np.nan
        reference[t_idx] = beta

    for t in range(reference.shape[0]):
        m = np.isnan(reference[t])
        if m.any():
            mu = float(np.nanmean(reference[t])) if (~m).any() else 0.5
            reference[t, m] = mu
    return RoadmapAtlas(
        reference=reference,
        tissue_labels=tissues,
        locus_chrom=[c for c, _ in sites],
        locus_start=[s for _, s in sites],
        n_samples_per_tissue=n_samples,
        source=str(root),
    )
