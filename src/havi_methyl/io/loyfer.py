"""Loyfer 2023 human-methylome atlas loaders.

Two paths are supported:

  - ``load_loyfer_atlas_matrix(path)``: load the precomputed atlas TSV
    (``ecd/data_resources/human_methylome_atlas/human_methylome_atlas.tsv``).
    Returns a ``(T_tissues, L_loci)`` reference matrix plus tissue and
    locus labels. This is the path used by the Phase 3 LOO bench.

  - ``load_loyfer_pat_directory(path, sites)``: load per-sample PAT files
    from ``hg38_pat_downloads/`` and aggregate to per-(tissue, CpG) means
    on a user-supplied site list. Slower and more expressive; needed when
    the precomputed TSV is not available or has been re-thresholded.

Both loaders accept an optional ``loci_subset`` argument so the matrix
can be restricted to a manuscript-defined panel of CpGs (for example the
``bridge_panel.v2.1.extended.bed.gz`` ECD panel found alongside the
atlas data).
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LoyferAtlas:
    """Atlas reference + provenance."""

    reference: np.ndarray  # shape (T_tissues, L_loci)
    tissue_labels: list[str]
    locus_ids: list[str]
    source: str
    n_loci: int
    n_tissues: int


def _open_text(path: Path):
    """Open a text/gzipped file in text mode."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("r")


def load_loyfer_atlas_matrix(
    path: str | Path,
    locus_id_columns: tuple[str, ...] = ("chrom", "chr", "start", "end", "name"),
    locus_subset: set[str] | None = None,
    drop_columns: tuple[str, ...] = ("startCpG", "endCpG", "target", "direction"),
) -> LoyferAtlas:
    """Load the precomputed Loyfer human methylome atlas TSV.

    The expected file is a tab-separated table whose first few columns
    are locus identifiers (e.g. ``chrom start end name`` or ``chr start
    end startCpG endCpG target name direction``) and remaining columns
    are per-tissue or per-cell-type methylation fractions. This layout
    matches the published Loyfer/UXM_deconv panels
    ``Atlas.U25.l4.hg38.tsv`` / ``Atlas.U250.l4.hg38.tsv`` as well as any
    other Loyfer-derived atlas table with the same shape.

    ``locus_id_columns`` covers both the ``chrom`` and ``chr`` spellings
    by default. ``drop_columns`` defaults to the wgbstools panel
    metadata columns (``startCpG``, ``endCpG``, ``target``,
    ``direction``) so they are not mistaken for tissue methylation
    columns. If ``locus_subset`` is provided, only loci whose joined
    ID is in the set are kept.
    """
    p = Path(path)
    rows: list[list[float]] = []
    locus_ids: list[str] = []
    with _open_text(p) as f:
        header = f.readline().rstrip("\n").split("\t")
        # Identify locus-id and tissue columns.
        id_col_idx = []
        tissue_col_idx = []
        for i, col in enumerate(header):
            if col in locus_id_columns:
                id_col_idx.append((i, col))
            elif col in drop_columns:
                continue
            else:
                tissue_col_idx.append((i, col))
        tissue_labels = [c for _, c in tissue_col_idx]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            locus_id = "_".join(parts[i] for i, _ in id_col_idx) if id_col_idx else parts[0]
            if locus_subset is not None and locus_id not in locus_subset:
                continue
            try:
                row = [
                    float(parts[i]) if parts[i] not in ("", "NA", "NaN") else np.nan
                    for i, _ in tissue_col_idx
                ]
            except ValueError:
                continue
            rows.append(row)
            locus_ids.append(locus_id)
    if not rows:
        raise ValueError(f"No data rows parsed from {p}")
    matrix = np.asarray(rows, dtype=np.float64)  # shape (L_loci, T_tissues)
    # Reference matrix expected by the rest of the codebase is (T, L).
    reference = matrix.T
    # Drop tissue columns that are entirely NaN (atlas may include a few).
    keep = ~np.all(np.isnan(reference), axis=1)
    reference = reference[keep]
    tissue_labels = [t for t, k in zip(tissue_labels, keep, strict=False) if k]
    # Forward-fill NaN per tissue with the tissue mean to keep the matrix dense.
    for t in range(reference.shape[0]):
        m = np.isnan(reference[t])
        if m.any():
            reference[t, m] = float(np.nanmean(reference[t]))
    return LoyferAtlas(
        reference=reference,
        tissue_labels=tissue_labels,
        locus_ids=locus_ids,
        source=str(p),
        n_loci=len(locus_ids),
        n_tissues=len(tissue_labels),
    )


@dataclass
class _PatRecord:
    chrom: str
    start: int
    pattern: str  # e.g. "CCT" — C means methylated, T unmethylated
    count: int


def _iter_pat_records(path: Path):
    """Iterate (chrom, start, pattern, count) tuples from a Loyfer .pat.gz."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            try:
                yield _PatRecord(
                    chrom=parts[0],
                    start=int(parts[1]),
                    pattern=parts[2],
                    count=int(parts[3]),
                )
            except (ValueError, IndexError):
                continue


def _tissue_from_loyfer_filename(name: str) -> str | None:
    """Extract the tissue label from a Loyfer GSM file name.

    Files look like ``GSM5652176_Adipocytes-Z000000T7.hg38.pat.gz`` —
    we keep the cell-type tag (``Adipocytes``) as the tissue label.
    """
    m = re.match(r"GSM\d+_([^-]+(?:-[^-]+)*?)-Z[0-9A-Za-z]+\.hg38", name)
    return m.group(1) if m else None


def load_loyfer_pat_directory(
    path: str | Path,
    sites: list[tuple[str, int]],
    max_samples_per_tissue: int | None = None,
) -> LoyferAtlas:
    """Aggregate Loyfer per-sample PAT files into a per-tissue methylation matrix.

    ``sites`` is a list of ``(chrom, start)`` CpG positions; the loader
    returns a ``(T_tissues, len(sites))`` matrix of methylation fractions
    averaged across PAT records that overlap each site within each tissue.

    For each PAT record covering site ``s``, the methylation contribution
    is the fraction of ``C`` characters in the pattern at the offset
    matching that site, weighted by ``count``. Sites with no coverage in
    a given tissue receive the tissue mean over covered sites.

    ``max_samples_per_tissue`` caps the number of samples per tissue (use
    when the full atlas is too large for memory).
    """
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a directory")

    site_index = {(c, s): i for i, (c, s) in enumerate(sites)}
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    sample_counts: dict[str, int] = {}

    for child in sorted(p.iterdir()):
        name = child.name
        if not name.endswith(".pat.gz") or name.startswith("._"):
            continue
        tissue = _tissue_from_loyfer_filename(name)
        if tissue is None:
            continue
        if (
            max_samples_per_tissue is not None
            and sample_counts.get(tissue, 0) >= max_samples_per_tissue
        ):
            continue
        sample_counts[tissue] = sample_counts.get(tissue, 0) + 1

        if tissue not in sums:
            sums[tissue] = np.zeros(len(sites), dtype=np.float64)
            counts[tissue] = np.zeros(len(sites), dtype=np.float64)

        for rec in _iter_pat_records(child):
            for off, base in enumerate(rec.pattern):
                site_pos = rec.start + off
                idx = site_index.get((rec.chrom, site_pos))
                if idx is None:
                    continue
                if base == "C":
                    sums[tissue][idx] += rec.count
                    counts[tissue][idx] += rec.count
                elif base == "T":
                    counts[tissue][idx] += rec.count

    tissues = sorted(sums.keys())
    reference = np.zeros((len(tissues), len(sites)), dtype=np.float64)
    for t_idx, tissue in enumerate(tissues):
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = sums[tissue] / np.maximum(counts[tissue], 1.0)
        beta[counts[tissue] == 0] = np.nan
        reference[t_idx] = beta

    # Imputation: replace NaN with the tissue mean.
    for t in range(reference.shape[0]):
        m = np.isnan(reference[t])
        if m.any():
            reference[t, m] = float(np.nanmean(reference[t]))
    return LoyferAtlas(
        reference=reference,
        tissue_labels=tissues,
        locus_ids=[f"{c}:{s}" for c, s in sites],
        source=str(p),
        n_loci=len(sites),
        n_tissues=len(tissues),
    )
