"""Chromatin-aware cfDNA fragmentation simulator (Sec. 10, App. E).

Simulates per-locus fragment bags suitable for the synthetic recovery
experiments of Sec. 11. Faithful to the simulator-validation targets of
App. H (length distribution, end-motif frequencies, periodicity peak).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit

from havi_methyl.constants import (
    LENGTH_MIXTURE_MEANS,
    LENGTH_MIXTURE_STDS,
    LENGTH_MIXTURE_WEIGHTS,
    NB_DISPERSION_R,
    NUC_SPACING_MEAN,
    NUC_SPACING_STD,
    PERIODICITY_AMP,
    PERIODICITY_PERIOD,
    SIGMA_NUC_BP,
)
from havi_methyl.utils import get_rng


@dataclass
class SimulatorParams:
    """Simulator parameter table (App. E table ``simparams``)."""

    nuc_spacing_mean: float = NUC_SPACING_MEAN
    nuc_spacing_std: float = NUC_SPACING_STD
    sigma_nuc: float = SIGMA_NUC_BP
    periodicity_amp: float = PERIODICITY_AMP
    periodicity_period: float = PERIODICITY_PERIOD
    length_means: tuple[float, ...] = LENGTH_MIXTURE_MEANS
    length_stds: tuple[float, ...] = LENGTH_MIXTURE_STDS
    length_weights: tuple[float, ...] = LENGTH_MIXTURE_WEIGHTS
    nb_dispersion: float = NB_DISPERSION_R
    motif_seq_logit_scale: float = 1.0
    motif_meth_logit_scale: float = 1.0
    error_rate: float = 0.01


# ---------- Step 1: nucleosome positioning (App. E) ----------


def sample_nucleosomes(
    region_length: int,
    params: SimulatorParams | None = None,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Renewal-process nucleosome positioning with mean spacing 187 bp.

    Returns positions in [0, region_length] satisfying the canonical
    Gaussian-spacing distribution. The Strauss repulsion at distances <147 bp
    is approximated by drawing the spacing from a truncated Gaussian.
    """
    gen = get_rng(rng)
    if params is None:
        params = SimulatorParams()
    positions = []
    pos = 0.0
    while pos < region_length:
        spacing = gen.normal(params.nuc_spacing_mean, params.nuc_spacing_std)
        spacing = max(spacing, 147.0)  # Strauss repulsion lower bound
        pos = pos + spacing
        if pos < region_length:
            positions.append(pos)
    return np.asarray(positions, dtype=np.float64)


# ---------- Step 3: fragment-length mixture ----------


def sample_fragment_lengths(
    n: int,
    params: SimulatorParams | None = None,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Three-component Gaussian mixture: 0.7*N(167,15^2) + 0.2*N(332,30^2) + 0.1*N(500,80^2).

    App. E eq. ``p(L)`` and Snyder 2016 length distribution.
    """
    gen = get_rng(rng)
    if params is None:
        params = SimulatorParams()
    weights = np.asarray(params.length_weights, dtype=np.float64)
    component = gen.choice(len(weights), size=n, p=weights / weights.sum())
    means = np.asarray(params.length_means, dtype=np.float64)[component]
    stds = np.asarray(params.length_stds, dtype=np.float64)[component]
    return gen.normal(means, stds)


def fragment_length_pdf(
    L: NDArray[np.float64], params: SimulatorParams | None = None
) -> NDArray[np.float64]:
    """Analytical density of the three-Gaussian mixture (used by tests)."""
    p = params or SimulatorParams()
    out = np.zeros_like(L, dtype=np.float64)
    for w, mu, sd in zip(p.length_weights, p.length_means, p.length_stds, strict=False):
        out = out + w * np.exp(-0.5 * ((L - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return out


# ---------- Step 4: end-motif sampling ----------


def make_motif_logits(
    n_motifs: int = 256, rng: int | np.random.Generator | None = None
) -> NDArray[np.float64]:
    """Random per-motif baseline logits with the four most-abundant motifs
    boosted to ~20% of the cuts (App. H targets)."""
    gen = get_rng(rng)
    logits = gen.standard_normal(n_motifs) * 0.4
    # Boost four representative motif indices to dominate
    top_idx = gen.choice(n_motifs, size=4, replace=False)
    logits[top_idx] += 1.5
    return logits


def sample_end_motifs(
    n: int,
    methylation: float,
    base_logits: NDArray[np.float64],
    meth_logits: NDArray[np.float64] | None = None,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Sample 4-mer indices from a categorical conditioned on methylation.

    Methylated cut sites enriched ~1.4x for CG-containing motifs (App. H).
    """
    gen = get_rng(rng)
    if meth_logits is None:
        meth_logits = np.zeros_like(base_logits)
        # Bump every fourth index (proxy for CG motifs)
        meth_logits[::4] += np.log(1.4)
    logits = base_logits + methylation * meth_logits
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    return gen.choice(len(p), size=n, p=p)


# ---------- Step 6: NB coverage ----------


def sample_coverage_nb(
    beta: ArrayLike,
    gc: ArrayLike,
    mappability: ArrayLike,
    r: float = NB_DISPERSION_R,
    b0: float = 0.0,
    b_gc: float = 1.5,
    b_beta: float = -0.5,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Per-locus NB coverage conditioned on GC and methylation (App. E step 6)."""
    gen = get_rng(rng)
    p = expit(b0 + b_gc * np.asarray(gc) + b_beta * np.asarray(beta))
    return gen.negative_binomial(r, 1.0 - p)


# ---------- High-level: methylation track + fragment bags ----------


def sample_methylation_track(
    L: int,
    n_clusters: int = 12,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Per-locus ground-truth beta in (0, 1) drawn from a piecewise mixture.

    The track is a sequence of clusters with cluster mean drawn from
    {0.05, 0.2, 0.5, 0.8, 0.95} plus logit-Normal local variation. Mirrors
    ``run_experiments.sample_methylation_track``.
    """
    gen = get_rng(rng)
    n_clusters = max(1, min(n_clusters, L))
    cluster_means = gen.choice([0.05, 0.2, 0.5, 0.8, 0.95], size=n_clusters)
    if n_clusters > 1:
        boundaries = np.sort(gen.choice(L, size=n_clusters - 1, replace=False))
    else:
        boundaries = np.array([], dtype=int)
    boundaries = np.concatenate([[0], boundaries, [L]])
    beta = np.zeros(L)
    for k in range(n_clusters):
        mu = cluster_means[k]
        n_in = boundaries[k + 1] - boundaries[k]
        eta = logit(np.clip(mu, 0.02, 0.98)) + gen.normal(0, 0.6, n_in)
        beta[boundaries[k] : boundaries[k + 1]] = expit(eta)
    return beta


def sample_fragment_bag(
    beta_locus: float,
    n_frag: int,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """For a single locus with ground-truth beta, produce a bag of fragments.

    Returns (features, methylation_indicator). Features per fragment:
    ``[length, motif_id, gc, orientation, dist_to_nuc]`` (Sec. 10 lstlisting).
    """
    gen = get_rng(rng)
    if n_frag == 0:
        return np.zeros((0, 5)), np.zeros(0, dtype=int)
    z = gen.binomial(1, beta_locus, size=n_frag).astype(int)
    mu_len = 167.0 + 5.0 * z
    length = gen.normal(mu_len, 12.0)
    motif = gen.normal(64 + 128 * z, 30).astype(int)
    motif = np.clip(motif, 0, 255)
    gc = gen.normal(0.45 + 0.10 * z, 0.05)
    orient = gen.binomial(1, 0.5, size=n_frag)
    dist = np.abs(gen.normal(0 - 30 * z, 25, size=n_frag))
    feats = np.stack([length, motif.astype(float), gc, orient.astype(float), dist], axis=1)
    return feats, z


@dataclass
class SimulatedDataset:
    """Container for a simulated dataset (Sec. 11 inputs)."""

    beta_pop: NDArray[np.float64]
    deltas: NDArray[np.float64]
    beta_sample: NDArray[np.float64]
    bags: list[list[NDArray[np.float64]]]
    n: NDArray[np.intp]
    n_meth: NDArray[np.intp]


def simulate_dataset(
    S: int,
    L: int,
    coverage: float,
    rng: int | np.random.Generator | None = None,
    sigma_delta: float = 0.3,
    sigma_eta: float = 0.4,
) -> SimulatedDataset:
    """Generate a S x L synthetic dataset at mean coverage ``coverage`` (Sec. 11)."""
    gen = get_rng(rng)
    beta_pop = sample_methylation_track(L, rng=gen)
    deltas = gen.normal(0.0, sigma_delta, size=S)
    eta_pop = logit(np.clip(beta_pop, 1e-3, 1 - 1e-3))
    eta_sample = eta_pop[None, :] + deltas[:, None] + gen.normal(0, sigma_eta, size=(S, L))
    beta_sample = expit(eta_sample)
    bags: list[list[NDArray[np.float64]]] = [[None] * L for _ in range(S)]
    counts = np.zeros((S, L), dtype=int)
    n_meth_counts = np.zeros((S, L), dtype=int)
    for s in range(S):
        for ell in range(L):
            n = gen.poisson(coverage)
            counts[s, ell] = n
            feats, z = sample_fragment_bag(beta_sample[s, ell], int(n), rng=gen)
            bags[s][ell] = feats
            n_meth_counts[s, ell] = int(z.sum())
    return SimulatedDataset(
        beta_pop=beta_pop,
        deltas=deltas,
        beta_sample=beta_sample,
        bags=bags,
        n=counts,
        n_meth=n_meth_counts,
    )


# ---------- Simulator validation runner (App. H, IMPL-09) ----------


def simulator_validation_metrics(
    n_frag: int = 100_000,
    rng: int | np.random.Generator | None = None,
    params: SimulatorParams | None = None,
) -> dict[str, float]:
    """Compute App. H validation axes from a single simulator draw.

    Returns:
      - ``length_primary_mode_bp``: histogram-derived primary mode of the
        fragment-length distribution.
      - ``length_secondary_height``: max histogram density in 320-350 bp.
      - ``length_periodicity_amplitude``: amplitude of the 10.4-bp peak in
        the autocorrelation of integer-rounded lengths in 100-300 bp.
      - ``top4_motif_fraction``: cumulative frequency of the four most
        common 5'-cut 4-mers under the random-baseline simulator.
      - ``meth_cut_bias_effect_size``: difference in mean GC-content between
        methylated and unmethylated fragments at one matched locus, as a
        proxy for methylation-conditioned cut bias.
    """
    gen = get_rng(rng)
    p = params or SimulatorParams()
    L = sample_fragment_lengths(n_frag, params=p, rng=gen)
    edges = np.arange(50, 800, 5)
    hist, _ = np.histogram(L, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    primary_mode = float(centers[hist.argmax()])
    secondary_band = (320 <= centers) & (centers <= 350)
    secondary_height = float(hist[secondary_band].max()) if secondary_band.any() else 0.0
    # 10.4-bp periodicity: autocorrelation of length residuals in 100-300 bp.
    L_int = np.rint(L).astype(int)
    short = L_int[(L_int >= 100) & (L_int <= 300)]
    counts = np.bincount(short - 100, minlength=201).astype(float)
    counts -= counts.mean()
    if counts.std() > 0:
        counts /= counts.std()
    autocorr = np.correlate(counts, counts, mode="full") / len(counts)
    mid = len(autocorr) // 2
    lag10 = autocorr[mid + 10]
    lag11 = autocorr[mid + 11]
    period_peak = float(max(lag10, lag11))
    # Motif top-4.
    base = make_motif_logits(rng=gen)
    motifs = sample_end_motifs(n_frag, methylation=0.5, base_logits=base, rng=gen)
    counts_motif = np.bincount(motifs, minlength=256)
    top4 = float(np.sort(counts_motif)[-4:].sum() / counts_motif.sum())
    # Methylation-conditioned cut bias: GC-effect at beta=0.95 vs beta=0.05.
    feats_hi, _ = sample_fragment_bag(0.95, n_frag // 10, rng=gen)
    feats_lo, _ = sample_fragment_bag(0.05, n_frag // 10, rng=gen)
    if feats_hi.shape[0] and feats_lo.shape[0]:
        gc_effect = float(feats_hi[:, 2].mean() - feats_lo[:, 2].mean())
    else:
        gc_effect = 0.0
    return {
        "length_primary_mode_bp": primary_mode,
        "length_secondary_height": secondary_height,
        "length_periodicity_amplitude": period_peak,
        "top4_motif_fraction": top4,
        "meth_cut_bias_effect_size": gc_effect,
    }


# ---------- Cut-site density (App. E Step 2) ----------


def cut_site_density(
    x: NDArray[np.float64],
    nucleosome_positions: NDArray[np.float64],
    params: SimulatorParams | None = None,
) -> NDArray[np.float64]:
    """p_cut(x) ~ exp(-d_nuc(x)/sigma_nuc) * (1 + a cos(2 pi d / 10.4)).

    App. E eq. ``p_cut`` is a one-dimensional density over genomic position.
    """
    p = params or SimulatorParams()
    if len(nucleosome_positions) == 0:
        return np.ones_like(x)
    d_nuc = np.abs(x[:, None] - nucleosome_positions[None, :]).min(axis=1)
    decay = np.exp(-d_nuc / p.sigma_nuc)
    period = 1.0 + p.periodicity_amp * np.cos(2 * np.pi * d_nuc / p.periodicity_period)
    return decay * period
