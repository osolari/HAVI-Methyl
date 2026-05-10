"""Liu 2024 paired WGS + WGBS loaders for the HAVI-Methyl FinaleMe benchmark.

Three on-disk components are wired up:

  - ``frag_wgs/*.tsv.gz`` — per-fragment feature tables produced by the
    FinaleMe pre-processor. Standard columns include the chrom and
    (start, end) coordinates plus length, end-motif id, GC content, and
    any model-specific covariates. Used to build the encoder fragment
    bag per (sample, locus).
  - ``meth_wgbs/*.bed.gz`` — paired WGBS ground-truth methylation calls
    (``...CG.strand.6plus2.bed.gz``). BED columns: chrom, start, end,
    strand, methylation fraction, methylated count, total count, ...
  - ``meth_wgs/*.bw`` — FinaleMe's own predictions (``methy``,
    ``methy_count``, ``cov`` per sample). Used as the FinaleMe baseline
    in the ablation matrix.

The loader returns a ``FinaleMeDataset`` with the same fields as
``simulator.SimulatedDataset`` so existing benches drop-in.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FinaleMeDataset:
    sample_ids: list[str]
    locus_chrom: list[str]
    locus_start: list[int]
    locus_end: list[int]
    bags: list[list[np.ndarray]]
    n: np.ndarray  # (S, L) fragment counts per locus
    n_meth: np.ndarray  # (S, L) methylated-CpG counts (from WGBS truth)
    beta_sample: np.ndarray  # (S, L) WGBS-derived methylation fraction
    feature_names: list[str]
    source: dict[str, str]


def _open_text(path: Path):
    """Open a (possibly gzipped) text file."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("r")


def _sample_id_from_filename(name: str) -> str:
    """Stable sample identifier from a FinaleMe filename.

    Strips the common ``.b37.tsv.gz`` / ``.6plus2.bed.gz`` etc. tails so
    matched WGS and WGBS files share an id.
    """
    base = re.sub(r"\.(b37\.tsv|aligned\..*)\.gz$", "", name)
    base = re.sub(r"\.tsv\.gz$|\.bed\.gz$|\.bw$", "", base)
    base = re.sub(r"^WGBS_", "", base)
    return base


def _parse_locus_panel(
    panel_path: Path | None, default_n: int = 5000
) -> tuple[list[str], list[int], list[int]]:
    """Read a BED-like locus panel; if not provided, generate a default panel.

    The returned three lists give chrom/start/end for each locus.
    """
    if panel_path is None:
        # 5 kb-spaced default panel on chr1; only used if the caller
        # didn't supply a real panel and is happy with placeholder loci.
        chrom = ["chr1"] * default_n
        start = [i * 5000 + 1_000_000 for i in range(default_n)]
        end = [s + 1 for s in start]
        return chrom, start, end
    chrom: list[str] = []
    start: list[int] = []
    end: list[int] = []
    with _open_text(panel_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            try:
                chrom.append(parts[0])
                start.append(int(parts[1]))
                end.append(int(parts[2]))
            except ValueError:
                continue
    if not chrom:
        raise ValueError(f"No loci parsed from {panel_path}")
    return chrom, start, end


def _read_wgbs_bed(
    path: Path,
    locus_index: dict[tuple[str, int], int],
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a single WGBS BED into (n_meth, n_total) per locus.

    The 6plus2 format produced by FinaleMe lays out columns as:
        chrom  start  end  strand  beta  methylated  total ...
    We use ``methylated`` and ``total`` directly and ignore strand for
    the simplified per-locus aggregation.
    """
    n_meth = np.zeros(L, dtype=np.int64)
    n_total = np.zeros(L, dtype=np.int64)
    with _open_text(path) as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 7:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                meth = int(parts[5])
                total = int(parts[6])
            except (ValueError, IndexError):
                continue
            idx = locus_index.get((chrom, start))
            if idx is None:
                continue
            n_meth[idx] += meth
            n_total[idx] += total
    return n_meth, n_total


def _read_wgs_fragments(
    path: Path,
    locus_index: dict[tuple[str, int], int],
    L: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Read per-fragment features and group by locus.

    The Liu 2024 ``frag_wgs/*.tsv.gz`` files are TSVs whose first three
    columns are ``chrom start end``; subsequent columns are numeric
    feature values. We aggregate every fragment whose start position is
    within 1 bp of a panel locus into that locus's bag.
    """
    bags: list[list[np.ndarray]] = [[] for _ in range(L)]
    counts = np.zeros(L, dtype=np.int64)
    with _open_text(path) as f:
        header = f.readline().rstrip().split("\t")
        n_features = max(1, len(header) - 3)
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
            except ValueError:
                continue
            idx = locus_index.get((chrom, start))
            if idx is None:
                continue
            try:
                feats = [
                    float(x) if x not in ("", "NA") else 0.0 for x in parts[3 : 3 + n_features]
                ]
            except ValueError:
                continue
            bags[idx].append(np.asarray(feats, dtype=np.float64))
            counts[idx] += 1
    bag_arrays = [np.stack(items, axis=0) if items else np.zeros((0, n_features)) for items in bags]
    return bag_arrays, counts


def load_finaleme_dataset(
    path: str | Path,
    locus_panel: str | Path | None = None,
    max_samples: int | None = None,
    max_loci: int | None = None,
) -> FinaleMeDataset:
    """Load matched WGS + WGBS samples from the FinaleMe data directory.

    Expected directory layout (matches the on-disk shape under
    ``/Volumes/Omid Solari/finaleme/``):

        <path>/frag_wgs/*.tsv.gz   per-fragment WGS features
        <path>/meth_wgbs/*.bed.gz  WGBS truth (CpG strand 6plus2)

    Pairing is by stripped sample-id (see ``_sample_id_from_filename``).
    Only samples that appear in BOTH directories are kept.

    ``locus_panel`` is a BED file giving the (chrom, start, end) panel of
    CpG loci to evaluate. Without it the loader generates a placeholder
    5kb-spaced grid on chr1 (useful only for plumbing tests).
    """
    root = Path(path)
    frag_dir = root / "frag_wgs"
    meth_dir = root / "meth_wgbs"
    if not frag_dir.is_dir() or not meth_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {root}/frag_wgs and {root}/meth_wgbs; only got "
            f"{[d.name for d in root.iterdir() if d.is_dir()]}"
        )

    chrom_list, start_list, end_list = _parse_locus_panel(
        Path(locus_panel) if locus_panel else None
    )
    if max_loci is not None:
        chrom_list = chrom_list[:max_loci]
        start_list = start_list[:max_loci]
        end_list = end_list[:max_loci]
    L = len(chrom_list)
    locus_index = {(c, s): i for i, (c, s) in enumerate(zip(chrom_list, start_list, strict=False))}

    frag_files = {_sample_id_from_filename(p.name): p for p in sorted(frag_dir.glob("*.tsv.gz"))}
    meth_files = {_sample_id_from_filename(p.name): p for p in sorted(meth_dir.glob("*.bed.gz"))}
    paired = sorted(set(frag_files.keys()) & set(meth_files.keys()))
    if max_samples is not None:
        paired = paired[:max_samples]
    if not paired:
        raise ValueError(
            f"No paired WGS/WGBS samples in {root} after id-matching; "
            f"frag_wgs={len(frag_files)} meth_wgbs={len(meth_files)}"
        )

    S = len(paired)
    n_meth_mat = np.zeros((S, L), dtype=np.int64)
    n_total_mat = np.zeros((S, L), dtype=np.int64)
    bags_per_sample: list[list[np.ndarray]] = []
    feature_names: list[str] = []

    for s_idx, sample_id in enumerate(paired):
        # WGBS truth.
        n_m, n_t = _read_wgbs_bed(meth_files[sample_id], locus_index, L)
        n_meth_mat[s_idx] = n_m
        n_total_mat[s_idx] = n_t
        # WGS fragments.
        sample_bags, _counts = _read_wgs_fragments(frag_files[sample_id], locus_index, L)
        bags_per_sample.append(sample_bags)
        if not feature_names:
            with _open_text(frag_files[sample_id]) as f:
                header = f.readline().rstrip().split("\t")
            feature_names = header[3:]

    # n is the per-locus fragment count (from WGS frag_wgs), not the WGBS total.
    n = np.zeros((S, L), dtype=np.int64)
    for s_idx in range(S):
        for ell in range(L):
            n[s_idx, ell] = bags_per_sample[s_idx][ell].shape[0]

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(n_total_mat > 0, n_meth_mat / np.maximum(n_total_mat, 1), 0.5)
    return FinaleMeDataset(
        sample_ids=paired,
        locus_chrom=chrom_list,
        locus_start=start_list,
        locus_end=end_list,
        bags=bags_per_sample,
        n=n,
        n_meth=n_meth_mat,
        beta_sample=beta,
        feature_names=feature_names,
        source={"frag_wgs": str(frag_dir), "meth_wgbs": str(meth_dir)},
    )
