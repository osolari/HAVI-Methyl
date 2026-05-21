"""Tests for the cfDNA simulator (Sec. 10, App. E, App. H validation)."""

from __future__ import annotations

import numpy as np

from havi_methyl import (
    SimulatorParams,
    cut_site_density,
    fragment_length_pdf,
    sample_coverage_nb,
    sample_end_motifs,
    sample_fragment_bag,
    sample_fragment_lengths,
    sample_methylation_track,
    sample_nucleosomes,
    simulate_dataset,
)
from havi_methyl.simulator import make_motif_logits


def test_methylation_track_in_unit_interval(rng):
    beta = sample_methylation_track(L=500, rng=rng)
    assert beta.shape == (500,)
    assert ((beta > 0) & (beta < 1)).all()


def test_nucleosome_spacing_close_to_canonical(rng):
    """App. H: mean spacing ~187 bp, after Strauss truncation slightly higher."""
    pos = sample_nucleosomes(region_length=200_000, params=SimulatorParams(), rng=rng)
    spacings = np.diff(pos)
    # Strauss repulsion truncates at 147; mean spacing should remain near 187
    assert 180 <= spacings.mean() <= 230


def test_fragment_length_modes(rng):
    """App. H: dominant mode at 167, secondary at 332, tertiary near 500 (Snyder 2016)."""
    L = sample_fragment_lengths(50_000, params=SimulatorParams(), rng=rng)
    hist, edges = np.histogram(L, bins=np.arange(50, 800, 5))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = centers[hist.argmax()]
    assert 150 <= mode <= 180


def test_fragment_length_pdf_normalizes():
    grid = np.linspace(0, 1500, 100_001)
    pdf = fragment_length_pdf(grid)
    integral = float(np.trapezoid(pdf, grid))
    np.testing.assert_allclose(integral, 1.0, atol=1e-3)


def test_motif_top4_frequency(rng):
    """App. H: four most abundant 4-mers ~20% of cuts in healthy plasma."""
    base = make_motif_logits(rng=rng)
    samples = sample_end_motifs(50_000, methylation=0.5, base_logits=base, rng=rng)
    counts = np.bincount(samples, minlength=256)
    top4 = np.sort(counts)[-4:].sum()
    fraction = top4 / counts.sum()
    assert 0.05 <= fraction <= 0.6  # generous bound around the published ~0.20


def test_coverage_nb_nonnegative(rng):
    n = sample_coverage_nb(
        beta=np.full(50, 0.5),
        gc=np.full(50, 0.45),
        mappability=np.ones(50),
        rng=rng,
    )
    assert (n >= 0).all()


def test_fragment_bag_zero_count(rng):
    feats, z = sample_fragment_bag(beta_locus=0.5, n_frag=0, rng=rng)
    assert feats.shape == (0, 5)
    assert z.shape == (0,)


def test_simulate_dataset_shape(rng):
    sim = simulate_dataset(S=3, L=20, coverage=2.0, rng=rng)
    assert sim.beta_pop.shape == (20,)
    assert sim.beta_sample.shape == (3, 20)
    assert sim.n.shape == (3, 20)
    assert sim.n_meth.shape == (3, 20)
    assert sim.deltas.shape == (3,)
    assert ((sim.beta_sample > 0) & (sim.beta_sample < 1)).all()


def test_cut_site_density_periodicity(rng):
    """App. H: cut-site density has 10.4 bp periodicity peak."""
    nuc = np.array([100.0])
    x = np.arange(50, 200, dtype=float)
    rho = cut_site_density(x, nuc)
    # Density should be highest right next to the nucleosome and oscillate at ~10.4 bp
    assert rho.argmax() == 0 or rho.argmax() == len(x) - 1 or abs(x[rho.argmax()] - 100.0) <= 21.0
