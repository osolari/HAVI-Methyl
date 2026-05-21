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
    n: np.ndarray  # (S, L) WGS fragment counts per locus (encoder feature)
    n_meth: np.ndarray  # (S, L) methylated read count (from WGBS reads, BB successes)
    n_total: np.ndarray  # (S, L) WGBS read coverage (BB trials -- distinct from n!)
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


def _flowcell_from_filename(name: str) -> str | None:
    """Extract a Loyfer-style ``FC########`` flowcell id from a filename.

    Liu 2024 paired-sample filenames embed the flowcell library id (e.g.
    ``1_FC16207043_H2MHNADXX.1.b37.tsv.gz`` -> ``FC16207043``); the
    manifest's ``WGS_library_id`` / ``WGBS_library_id`` columns use the
    same form, so flowcell id is the natural pairing key.
    """
    m = re.search(r"(FC\d+)", name)
    return m.group(1) if m else None


def _is_apple_shadow(name: str) -> bool:
    """macOS writes ``._<name>`` AppleDouble metadata sidecars onto
    non-APFS external volumes; they masquerade as data files under
    ``glob`` and must be filtered out.
    """
    return name.startswith("._")


def _load_finaleme_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load the FinaleMe Supplementary Table 1 manifest as a mapping
    ``WGS_library_id -> {WGBS_library_id, patient_id, tumor_type,
    sample_name, ...}``.

    Accepts a CSV with the published header row
    ``WGBS_library_id,WGS_library_id,patient_id,tumor_type,sample_name,
    WGS_bam_file,WGBS_bam_file``.
    """
    import csv

    out: dict[str, dict[str, str]] = {}
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wgs_id = row.get("WGS_library_id", "").strip()
            if not wgs_id:
                continue
            out[wgs_id] = {k: (v or "").strip() for k, v in row.items()}
    if not out:
        raise ValueError(f"No rows parsed from manifest {path}")
    return out


def _parse_locus_panel(
    panel_path: Path | None, default_n: int = 5000
) -> tuple[list[str], list[int], list[int]]:
    """Read a BED-like locus panel; if not provided, generate a default panel.

    The returned three lists give chrom/start/end for each locus.
    """
    if panel_path is None:
        # 5 kb-spaced default panel on chr1; only used if the caller
        # didn't supply a real panel and is happy with placeholder loci.
        chrom_default = ["chr1"] * default_n
        start_default = [i * 5000 + 1_000_000 for i in range(default_n)]
        end_default = [s + 1 for s in start_default]
        return chrom_default, start_default, end_default
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


def _norm_chrom(chrom: str) -> str:
    """Normalise hg19/b37 chromosome ids to ``chr``-prefixed form."""
    if chrom.startswith("chr"):
        return chrom
    if chrom == "MT":
        return "chrM"
    return f"chr{chrom}"


def _read_wgbs_bed(
    path: Path,
    panel_chrom: list[str],
    panel_start: list[int],
    panel_end: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Read a single WGBS BED into (n_meth, n_total) per locus.

    The published Liu 2024 ``meth_wgbs/`` files use the BED-track form
    produced by their bisulfite pipeline (an 8-column extension of
    BED6)::

        track name=... type=bedDetail ...
        <chrom>  <start>  <end>  <name>  <score>  <strand>  <meth_pct>  <coverage>

    where ``meth_pct`` is the percent-methylated value in [0,100] and
    ``coverage`` is the read count covering that CpG. ``meth_count`` is
    recovered as ``round(meth_pct/100 * coverage)``. ``chrom`` is the
    bare b37 form (``"1"``) and is normalised to ``"chr1"`` to match the
    panel. Each CpG record is aggregated into every locus whose interval
    contains it, so panels of block-style markers (e.g. Loyfer U25) can
    pool the multiple CpGs within each block.
    """
    L = len(panel_chrom)
    by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for i, (c, s, e) in enumerate(zip(panel_chrom, panel_start, panel_end, strict=False)):
        by_chrom.setdefault(c, []).append((s, e, i))
    for c in by_chrom:
        by_chrom[c].sort()

    n_meth = np.zeros(L, dtype=np.int64)
    n_total = np.zeros(L, dtype=np.int64)
    with _open_text(path) as f:
        for line in f:
            if line.startswith("track") or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 8:
                continue
            chrom = _norm_chrom(parts[0])
            try:
                start = int(parts[1])
                meth_pct = float(parts[6])
                total = int(parts[7])
            except (ValueError, IndexError):
                continue
            panel = by_chrom.get(chrom)
            if not panel:
                continue
            for l_start, l_end, idx in panel:
                if start < l_start:
                    break
                if start >= l_end:
                    continue
                n_total[idx] += total
                n_meth[idx] += int(round(meth_pct / 100.0 * total))
    return n_meth, n_total


def _read_wgs_fragments(
    path: Path,
    panel_chrom: list[str],
    panel_start: list[int],
    panel_end: list[int],
    feature_names: tuple[str, ...] = ("length", "strand_signed"),
) -> tuple[list[np.ndarray], np.ndarray]:
    """Read per-fragment features and group by locus by interval overlap.

    The Liu 2024 ``frag_wgs/*.tsv.gz`` files on disk are plain BED6
    fragment records (one per cfDNA fragment)::

        <chrom>  <start>  <end>  <name>  <score>  <strand>

    no header, ``chrom`` is the bare b37 form (``"1"``) and is
    normalised to ``"chr1"`` to match the panel. A fragment is binned
    into a locus if its [start, end) interval overlaps the locus
    (chrom, start, end) interval. Features extracted per fragment are
    determined by ``feature_names``; the default set is
    ``(length, strand_signed)`` where ``strand_signed`` is ``+1`` for
    ``"+"`` and ``-1`` for ``"-"``.
    """
    L = len(panel_chrom)
    # Group panel intervals by chrom for O(L_chrom) lookups.
    by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for i, (c, s, e) in enumerate(zip(panel_chrom, panel_start, panel_end, strict=False)):
        by_chrom.setdefault(c, []).append((s, e, i))
    for c in by_chrom:
        by_chrom[c].sort()

    n_features = len(feature_names)
    bags: list[list[np.ndarray]] = [[] for _ in range(L)]
    with _open_text(path) as f:
        for line in f:
            if line.startswith("track") or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            chrom = _norm_chrom(parts[0])
            try:
                f_start = int(parts[1])
                f_end = int(parts[2])
            except ValueError:
                continue
            strand = parts[5] if len(parts) >= 6 else "+"
            length = f_end - f_start
            feats_list: list[float] = []
            for name in feature_names:
                if name == "length":
                    feats_list.append(float(length))
                elif name == "strand_signed":
                    feats_list.append(1.0 if strand == "+" else -1.0)
                else:
                    feats_list.append(0.0)
            feats = np.asarray(feats_list, dtype=np.float64)
            panel = by_chrom.get(chrom)
            if not panel:
                continue
            for l_start, l_end, idx in panel:
                if f_end <= l_start:
                    break
                if f_start >= l_end:
                    continue
                bags[idx].append(feats)
    bag_arrays = [np.stack(items, axis=0) if items else np.zeros((0, n_features)) for items in bags]
    return bag_arrays, None


def _read_bigwig_per_locus_mean(
    bw_path: Path,
    panel_chrom: list[str],
    panel_start: list[int],
    panel_end: list[int],
) -> np.ndarray:
    """Per-locus mean of a BigWig signal track.

    Used to fold the Liu 2024 buffy-coat methylation prior
    (``wgbs_buffyCoat_jensen2015GB.methy.hg19.bw``, values in [0, 100])
    into each fragment's feature vector. ``NaN`` (no coverage) becomes
    the panel mean so the prior column never injects ``NaN`` downstream.
    """
    import pyBigWig

    L = len(panel_chrom)
    out = np.full(L, np.nan, dtype=np.float64)
    with pyBigWig.open(str(bw_path)) as bw:
        chroms = bw.chroms()
        for i, (c, s, e) in enumerate(zip(panel_chrom, panel_start, panel_end, strict=False)):
            if c not in chroms:
                continue
            try:
                vals = bw.stats(c, s, max(e, s + 1), type="mean")
            except RuntimeError:
                continue
            if vals and vals[0] is not None:
                out[i] = float(vals[0])
    # Scale [0,100] -> [0,1] to match downstream methylation conventions.
    out = out / 100.0
    if np.isnan(out).any():
        m = float(np.nanmean(out)) if np.isfinite(out).any() else 0.5
        out[np.isnan(out)] = m
    return out


def load_finaleme_dataset(
    path: str | Path,
    locus_panel: str | Path | None = None,
    max_samples: int | None = None,
    max_loci: int | None = None,
    manifest: str | Path | None = None,
    buffy_coat_bw: str | Path | None = None,
) -> FinaleMeDataset:
    """Load matched WGS + WGBS samples from the FinaleMe data directory.

    Expected directory layout (matches the on-disk shape under
    ``/Volumes/Omid Solari/finaleme/``):

        <path>/frag_wgs/*.tsv.gz   per-fragment WGS features
        <path>/meth_wgbs/*.bed.gz  WGBS truth (CpG strand 6plus2)

    Pairing strategies (in order of precedence):

    * ``manifest`` provided: parse Liu 2024 Supplementary Table 1 (CSV
      with ``WGS_library_id``/``WGBS_library_id``/``patient_id`` cols)
      and pair via the flowcell ids embedded in filenames. This is the
      published 80-pair mapping; the patient id becomes the canonical
      sample identifier.
    * ``manifest`` is ``None``: fall back to filename-id stripping; only
      works when WGS and WGBS share the same id stem (e.g. ``HD_45``).

    Apple-Double sidecars (``._<file>``) on macOS external volumes are
    filtered out before pairing.

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

    frag_paths = [p for p in sorted(frag_dir.glob("*.tsv.gz")) if not _is_apple_shadow(p.name)]
    meth_paths = [p for p in sorted(meth_dir.glob("*.bed.gz")) if not _is_apple_shadow(p.name)]

    if manifest is not None:
        manifest_map = _load_finaleme_manifest(Path(manifest))
        frag_by_fc = {fc: p for p in frag_paths if (fc := _flowcell_from_filename(p.name))}
        meth_by_fc = {fc: p for p in meth_paths if (fc := _flowcell_from_filename(p.name))}
        paired_pairs: list[tuple[str, Path, Path]] = []
        for wgs_id, row in sorted(manifest_map.items()):
            wgbs_id = row.get("WGBS_library_id", "")
            patient = row.get("patient_id", "") or row.get("sample_name", "") or wgs_id
            f = frag_by_fc.get(wgs_id)
            m = meth_by_fc.get(wgbs_id)
            if f is not None and m is not None:
                paired_pairs.append((patient, f, m))
        if max_samples is not None:
            paired_pairs = paired_pairs[:max_samples]
        if not paired_pairs:
            raise ValueError(
                f"Manifest {manifest} matched no on-disk pairs; "
                f"frag_flowcells={len(frag_by_fc)} meth_flowcells={len(meth_by_fc)} "
                f"manifest_rows={len(manifest_map)}"
            )
        paired = [pid for pid, _f, _m in paired_pairs]
        frag_files = {pid: f for pid, f, _m in paired_pairs}
        meth_files = {pid: m for pid, _f, m in paired_pairs}
    else:
        frag_files = {_sample_id_from_filename(p.name): p for p in frag_paths}
        meth_files = {_sample_id_from_filename(p.name): p for p in meth_paths}
        paired = sorted(set(frag_files.keys()) & set(meth_files.keys()))
        if max_samples is not None:
            paired = paired[:max_samples]
        if not paired:
            raise ValueError(
                f"No paired WGS/WGBS samples in {root} after id-matching; "
                f"frag_wgs={len(frag_files)} meth_wgbs={len(meth_files)}. "
                f"For the Liu 2024 lab drive layout, pass manifest=<sample_pairs.csv>."
            )

    if buffy_coat_bw is not None:
        prior_per_locus = _read_bigwig_per_locus_mean(
            Path(buffy_coat_bw), chrom_list, start_list, end_list
        )
        feature_names: list[str] = ["length", "strand_signed", "buffy_coat_prior"]
    else:
        prior_per_locus = None
        feature_names = ["length", "strand_signed"]

    S = len(paired)
    n_meth_mat = np.zeros((S, L), dtype=np.int64)
    n_total_mat = np.zeros((S, L), dtype=np.int64)
    bags_per_sample: list[list[np.ndarray]] = []

    for s_idx, sample_id in enumerate(paired):
        n_m, n_t = _read_wgbs_bed(meth_files[sample_id], chrom_list, start_list, end_list)
        n_meth_mat[s_idx] = n_m
        n_total_mat[s_idx] = n_t
        sample_bags, _counts = _read_wgs_fragments(
            frag_files[sample_id], chrom_list, start_list, end_list
        )
        if prior_per_locus is not None:
            for ell, fb in enumerate(sample_bags):
                if fb.shape[0] == 0:
                    sample_bags[ell] = np.zeros((0, len(feature_names)))
                else:
                    prior_col = np.full((fb.shape[0], 1), prior_per_locus[ell])
                    sample_bags[ell] = np.concatenate([fb, prior_col], axis=1)
        bags_per_sample.append(sample_bags)

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
        n_total=n_total_mat,
        beta_sample=beta,
        feature_names=feature_names,
        source={"frag_wgs": str(frag_dir), "meth_wgbs": str(meth_dir)},
    )
