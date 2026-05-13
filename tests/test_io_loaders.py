"""Phase 5 IO loaders: tests using synthetic fixtures.

We can't read ``/Volumes/Omid Solari`` from the test runner due to
macOS TCC, so each test fabricates a tiny on-disk fixture matching the
real format and verifies the loader parses it correctly.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def _write_text_gz(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(content)


# ---------------------------- Loyfer atlas TSV ----------------------------


def test_load_loyfer_atlas_matrix(tmp_path: Path) -> None:
    from havi_methyl.io import load_loyfer_atlas_matrix

    fixture = tmp_path / "atlas.tsv"
    fixture.write_text(
        "chrom\tstart\tend\tname\tLiver\tHeart\tLung\n"
        "chr1\t100\t101\tcpg1\t0.10\t0.50\t0.90\n"
        "chr1\t200\t201\tcpg2\t0.20\t0.40\t0.80\n"
        "chr1\t300\t301\tcpg3\t0.30\t0.30\t0.70\n"
    )
    atlas = load_loyfer_atlas_matrix(fixture)
    assert atlas.reference.shape == (3, 3)
    assert atlas.tissue_labels == ["Liver", "Heart", "Lung"]
    assert atlas.n_loci == 3 and atlas.n_tissues == 3
    np.testing.assert_allclose(atlas.reference[0], [0.1, 0.2, 0.3])  # Liver
    np.testing.assert_allclose(atlas.reference[2], [0.9, 0.8, 0.7])  # Lung


def test_load_loyfer_atlas_subset_filter(tmp_path: Path) -> None:
    from havi_methyl.io import load_loyfer_atlas_matrix

    fixture = tmp_path / "atlas.tsv"
    fixture.write_text(
        "chrom\tstart\tend\tname\tLiver\tLung\n"
        "chr1\t100\t101\tcpg1\t0.10\t0.90\n"
        "chr1\t200\t201\tcpg2\t0.20\t0.80\n"
    )
    atlas = load_loyfer_atlas_matrix(fixture, locus_subset={"chr1_100_101_cpg1"})
    assert atlas.n_loci == 1
    assert atlas.locus_ids == ["chr1_100_101_cpg1"]


# ---------------------------- Loyfer PAT directory ----------------------------


def test_load_loyfer_pat_directory_aggregates_per_tissue(tmp_path: Path) -> None:
    from havi_methyl.io import load_loyfer_pat_directory

    pat_dir = tmp_path / "pat"
    pat_dir.mkdir()
    # Two adipose samples, one liver sample. Pattern is 3 CpGs starting
    # at chr1:100; "C" = methylated, "T" = unmethylated.
    _write_text_gz(
        pat_dir / "GSM0000001_Adipocytes-Z00000001.hg38.pat.gz",
        "chr1\t100\tCCC\t10\nchr1\t100\tTTT\t5\n",
    )
    _write_text_gz(
        pat_dir / "GSM0000002_Adipocytes-Z00000002.hg38.pat.gz",
        "chr1\t100\tCCT\t8\n",
    )
    _write_text_gz(
        pat_dir / "GSM0000003_Liver-Hepatocytes-Z00000003.hg38.pat.gz",
        "chr1\t100\tCCC\t3\nchr1\t100\tTTC\t1\n",
    )

    sites = [("chr1", 100), ("chr1", 101), ("chr1", 102)]
    atlas = load_loyfer_pat_directory(pat_dir, sites)
    assert atlas.n_loci == 3
    # Liver-Hepatocytes is parsed as the leading "Liver" component because
    # the helper splits at the last "-Z..." sample tag.
    assert "Adipocytes" in atlas.tissue_labels
    assert all(0.0 <= v <= 1.0 for v in atlas.reference.flatten())


# ---------------------------- FinaleMe paired loader ----------------------------


def test_load_finaleme_dataset_parses_paired_samples(tmp_path: Path) -> None:
    from havi_methyl.io import load_finaleme_dataset

    root = tmp_path / "finaleme"
    frag_dir = root / "frag_wgs"
    meth_dir = root / "meth_wgbs"
    panel_path = tmp_path / "panel.bed"
    panel_path.write_text("chr1\t100\t101\nchr1\t200\t201\n")

    # cfDNA BED6 fragment records (no header; chrom, start, end, name, score, strand).
    # sampleA: 2 fragments spanning [100,267) and [100,300) cover both loci;
    # 1 fragment at [200,367) covers locus 1 only -> n = [2, 3].
    _write_text_gz(
        frag_dir / "sampleA.b37.tsv.gz",
        "1\t100\t267\t.\t0\t+\n" "1\t100\t300\t.\t0\t-\n" "1\t200\t367\t.\t0\t+\n",
    )
    _write_text_gz(
        frag_dir / "sampleB.b37.tsv.gz",
        "1\t100\t267\t.\t0\t-\n" "1\t200\t367\t.\t0\t+\n",
    )
    # WGBS truth (Liu 2024 8-col track form: chrom start end name score strand meth_pct coverage).
    _write_text_gz(
        meth_dir / "sampleA.aligned.duplicates_marked.cpg.filtered.sort.CG.strand.6plus2.bed.gz",
        "track name=sampleA type=bedDetail\n"
        "1\t100\t101\t.\t0\t+\t60.0\t10\n"
        "1\t200\t201\t.\t0\t+\t40.0\t10\n",
    )
    _write_text_gz(
        meth_dir / "sampleB.aligned.duplicates_marked.cpg.filtered.sort.CG.strand.6plus2.bed.gz",
        "track name=sampleB type=bedDetail\n"
        "1\t100\t101\t.\t0\t+\t80.0\t10\n"
        "1\t200\t201\t.\t0\t+\t20.0\t10\n",
    )

    ds = load_finaleme_dataset(root, locus_panel=panel_path)
    assert ds.beta_sample.shape == (2, 2)
    np.testing.assert_allclose(ds.beta_sample[0], [0.6, 0.4])
    np.testing.assert_allclose(ds.beta_sample[1], [0.8, 0.2])
    # Fragment counts via interval overlap (a fragment is binned into every
    # locus it spans, not just the one its start matches).
    np.testing.assert_array_equal(ds.n[0], [2, 3])
    np.testing.assert_array_equal(ds.n[1], [1, 2])


# ---------------------------- Roadmap WGBS WIG loader ----------------------------


def test_load_roadmap_wgbs_atlas(tmp_path: Path) -> None:
    from havi_methyl.io import load_roadmap_wgbs_atlas

    root = tmp_path / "roadmap"
    liver = root / "liver"
    aorta = root / "heart_aorta"
    liver.mkdir(parents=True)
    aorta.mkdir(parents=True)
    _write_text_gz(
        liver / "GSM1_liver_sample.wig.gz",
        "track type=wiggle_0 name=test\n" "variableStep chrom=chr1\n" "100\t0.10\n" "200\t0.20\n",
    )
    _write_text_gz(
        liver / "GSM2_liver_sample.wig.gz",
        "track type=wiggle_0 name=test\n" "variableStep chrom=chr1\n" "100\t0.30\n" "200\t0.40\n",
    )
    _write_text_gz(
        aorta / "GSM3_aorta_sample.wig.gz",
        "variableStep chrom=chr1\n"
        "100\t90.0\n"  # percentage form (>1) -> divided by 100
        "200\t80.0\n",
    )

    sites = [("chr1", 100), ("chr1", 200)]
    atlas = load_roadmap_wgbs_atlas(root, sites)
    assert atlas.tissue_labels == ["heart_aorta", "liver"]
    np.testing.assert_allclose(atlas.reference[0], [0.9, 0.8])  # heart_aorta
    np.testing.assert_allclose(atlas.reference[1], [0.2, 0.3])  # liver mean over 2 samples
    assert atlas.n_samples_per_tissue == {"heart_aorta": 1, "liver": 2}
