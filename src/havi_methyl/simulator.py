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
    chromatin_aware: bool = False,
    motif_top4_target: float = 0.20,
) -> dict[str, float]:
    """Compute App. H validation axes from a single simulator draw.

    With ``chromatin_aware=True`` the metrics are computed against the
    full chromatin-aware simulator (Phase 4 / IMPL-09):
      - the cut-site autocorrelation peak at 10.4 bp uses cut positions
        from ``sample_cut_positions`` (linker-biased density), not just
        the length distribution mode;
      - the motif logits are boosted so top-4 fraction reaches the
        published ~0.20 target.

    Returns the same keys regardless of ``chromatin_aware`` so downstream
    consumers do not branch.
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

    # 10.4 bp periodicity.
    if chromatin_aware:
        # Compute autocorrelation of cut-position counts at 1-bp resolution
        # over a single 5 kb region with several nucleosomes — that's where
        # the linker-biased density's 10.4-bp modulation should appear.
        region = 5000
        nucs = sample_nucleosomes(region, params=p, rng=gen)
        cuts = sample_cut_positions(region, nucs, n_cuts=min(n_frag, 50_000), params=p, rng=gen)
        cut_counts = np.bincount(np.rint(cuts).astype(int), minlength=region).astype(float)
        cut_counts -= cut_counts.mean()
        if cut_counts.std() > 0:
            cut_counts /= cut_counts.std()
        autocorr = np.correlate(cut_counts, cut_counts, mode="full") / len(cut_counts)
        mid = len(autocorr) // 2
        # Look in a window 8-13 bp for the helical-pitch peak.
        period_peak = float(autocorr[mid + 8 : mid + 14].max())
    else:
        L_int = np.rint(L).astype(int)
        short = L_int[(L_int >= 100) & (L_int <= 300)]
        counts = np.bincount(short - 100, minlength=201).astype(float)
        counts -= counts.mean()
        if counts.std() > 0:
            counts /= counts.std()
        autocorr = np.correlate(counts, counts, mode="full") / len(counts)
        mid = len(autocorr) // 2
        period_peak = float(max(autocorr[mid + 10], autocorr[mid + 11]))

    # Motif top-4: with the chromatin-aware path, use a boosted baseline.
    if chromatin_aware:
        # See ``simulate_dataset_chromatin_aware`` for boost rationale.
        boost = float(np.log(63 * motif_top4_target / max(1 - motif_top4_target, 1e-6))) + 0.15
        base = gen.standard_normal(256) * 0.4
        top_idx = gen.choice(256, size=4, replace=False)
        base[top_idx] += boost
    else:
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

    Legacy center-biased form retained for backward compatibility with the
    Sec. 11 simplified harness. The App. E correction (Phase 4 / IMPL-09)
    moves to a *linker*-biased density implemented in
    ``cut_site_density_linker``; that is the formula the chromatin-aware
    simulator uses.
    """
    p = params or SimulatorParams()
    if len(nucleosome_positions) == 0:
        return np.ones_like(x)
    d_nuc = np.abs(x[:, None] - nucleosome_positions[None, :]).min(axis=1)
    decay = np.exp(-d_nuc / p.sigma_nuc)
    period = 1.0 + p.periodicity_amp * np.cos(2 * np.pi * d_nuc / p.periodicity_period)
    return decay * period


def cut_site_density_linker(
    x: NDArray[np.float64],
    nucleosome_positions: NDArray[np.float64],
    params: SimulatorParams | None = None,
) -> NDArray[np.float64]:
    """App. E corrected linker-biased cut density (Phase 4 / IMPL-09).

    ``p_cut(x) ∝ exp(-d_linker(x) / sigma_link) * [1 + a cos(2 pi d_center / 10.4)]_+``

    where ``d_linker`` is the distance to the nearest midpoint between
    consecutive nucleosomes and ``d_center`` is the distance to the
    nearest nucleosome center. The bracket is truncated at zero so the
    density stays nonnegative even when the periodicity term subtracts.
    """
    p = params or SimulatorParams()
    pos = np.sort(np.asarray(nucleosome_positions, dtype=np.float64))
    if pos.size == 0:
        return np.ones_like(x)
    d_center = np.abs(x[:, None] - pos[None, :]).min(axis=1)
    if pos.size >= 2:
        midpoints = 0.5 * (pos[:-1] + pos[1:])
        d_linker = np.abs(x[:, None] - midpoints[None, :]).min(axis=1)
    else:
        # Single nucleosome: fall back to center distance scaled by sigma.
        d_linker = d_center
    sigma_link = p.sigma_nuc  # planning value 20 bp per simparams table
    decay = np.exp(-d_linker / sigma_link)
    period = np.maximum(
        0.0, 1.0 + p.periodicity_amp * np.cos(2 * np.pi * d_center / p.periodicity_period)
    )
    return decay * period


def sample_cut_positions(
    region_length: int,
    nucleosome_positions: NDArray[np.float64],
    n_cuts: int,
    params: SimulatorParams | None = None,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample ``n_cuts`` positions in ``[0, region_length]`` from the
    linker-biased cut density.

    Discretises the region at 1-bp resolution, normalises the density to a
    pmf, then draws cuts with replacement. Used by the chromatin-aware
    simulator's Step 2.
    """
    gen = get_rng(rng)
    if n_cuts == 0:
        return np.zeros(0, dtype=np.float64)
    grid = np.arange(region_length, dtype=np.float64) + 0.5
    density = cut_site_density_linker(grid, nucleosome_positions, params=params)
    if density.sum() <= 0:
        return gen.uniform(0, region_length, size=n_cuts)
    pmf = density / density.sum()
    return gen.choice(grid, size=n_cuts, p=pmf, replace=True)


# ---------- Chromatin-aware simulator (Phase 4 / IMPL-09) ----------


def simulate_dataset_chromatin_aware(
    S: int,
    L: int,
    coverage: float,
    rng: int | np.random.Generator | None = None,
    sigma_delta: float = 0.3,
    sigma_eta: float = 0.4,
    region_length_per_locus: int = 1000,
    motif_top4_target: float = 0.20,
) -> SimulatedDataset:
    """Sec. 10 / App. E full chromatin-aware simulator (Phase 4 / IMPL-09).

    Differs from the Sec. 11 ``simulate_dataset`` by:
      1. Sampling nucleosome positions for each region via
         ``sample_nucleosomes`` (renewal + Strauss repulsion).
      2. Drawing cut positions from the *linker-biased* density
         ``cut_site_density_linker`` (App. E corrected formula).
      3. Sampling fragment lengths from the three-mode mixture and
         derived feature vectors per fragment.
      4. Sampling end motifs from a categorical with logits boosted so
         the top-4 fraction matches the published ``~0.20`` target.

    Returns a ``SimulatedDataset`` compatible with the downstream
    pipeline so existing benches / fits work unchanged.
    """
    gen = get_rng(rng)
    params = SimulatorParams()
    beta_pop = sample_methylation_track(L, rng=gen)
    deltas = gen.normal(0.0, sigma_delta, size=S)
    eta_pop = logit(np.clip(beta_pop, 1e-3, 1 - 1e-3))
    eta_sample = eta_pop[None, :] + deltas[:, None] + gen.normal(0, sigma_eta, size=(S, L))
    beta_sample = expit(eta_sample)

    # Boosted motif baseline so top-4 fraction hits the published ~20% target.
    # 4 exp(b) / (4 exp(b) + 252) = p → exp(b) = 63 p / (1 - p)
    # 4 exp(b) / (4 exp(b) + 252 * E[exp(noise)]) = target
    # noise ~ N(0, 0.4^2) -> E[exp(noise)] = exp(0.08); empirical tuning over
    # 10 seeds shows +0.15 to the analytical boost recovers a top-4 mean
    # around the target with std ~0.03.
    boost = float(np.log(63 * motif_top4_target / max(1 - motif_top4_target, 1e-6))) + 0.15
    base_logits = gen.standard_normal(256) * 0.4
    top_idx = gen.choice(256, size=4, replace=False)
    base_logits[top_idx] += boost

    bags: list[list[NDArray[np.float64]]] = [[None] * L for _ in range(S)]
    counts = np.zeros((S, L), dtype=int)
    n_meth_counts = np.zeros((S, L), dtype=int)
    for s in range(S):
        for ell in range(L):
            n = int(gen.poisson(coverage))
            counts[s, ell] = n
            if n == 0:
                bags[s][ell] = np.zeros((0, 5))
                continue
            # Step 1: nucleosome positions for this region.
            nucs = sample_nucleosomes(region_length_per_locus, params=params, rng=gen)
            # Step 2: linker-biased cut positions (used as the dist_to_nuc feature).
            cuts = sample_cut_positions(
                region_length_per_locus, nucs, n_cuts=n, params=params, rng=gen
            )
            # Step 3: lengths from the 3-component mixture.
            lengths = sample_fragment_lengths(n, params=params, rng=gen)
            # Step 4: end motifs conditional on per-fragment methylation.
            z = gen.binomial(1, beta_sample[s, ell], size=n).astype(int)
            motifs = sample_end_motifs(
                n,
                methylation=float(beta_sample[s, ell]),
                base_logits=base_logits,
                rng=gen,
            )
            # GC, orientation, distance-to-nearest-nucleosome features.
            gc = gen.normal(0.45 + 0.10 * z, 0.05)
            orient = gen.binomial(1, 0.5, size=n)
            if nucs.size > 0:
                dist_to_nuc = np.abs(cuts[:, None] - nucs[None, :]).min(axis=1)
            else:
                dist_to_nuc = np.zeros(n)
            feats = np.stack(
                [
                    lengths,
                    motifs.astype(float),
                    gc,
                    orient.astype(float),
                    dist_to_nuc,
                ],
                axis=1,
            )
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
