# Quickstart

A 30-second loop from a fresh checkout. For the conceptual map first,
see [Concepts](concepts.md); for the full surface, see the
[API reference](api.md).

## Install

```bash
make install        # numpy + scipy core
make install-dev    # + matplotlib, pandas, pytest, ruff, mypy
make install-torch  # + torch (Set Transformer + neural-spline flow)
```

The core library only requires `numpy` and `scipy`. The full HAVI-Methyl
encoder (Set Transformer + NSF flow) requires `torch`; without it, the
same math is exercised through the simplified-Gaussian variant of
Sec. 11. See [Installation](installation.md) for the full table.

## Simplified-numpy loop

```python
import havi_methyl as hm

# 1. Simulate a small dataset (Sec. 10, App. E).
sim = hm.simulate_dataset(S=12, L=300, coverage=5.0, rng=20260429)

# 2. FinaleMe-style baseline (2-cluster Gaussian EM on per-fragment features).
pred_baseline, fit = hm.finaleme_baseline_predict(sim.bags, sim.n)

# 3. Simplified HAVI-Methyl SVI (Sec. 6, Algorithm 1).
state = hm.fit_svi_simplified(pred_baseline, sim.n, n_iter=10)
pred_havi, std_havi = hm.predict_with_state(state, pred_baseline, sim.n)

# 4. Compare.
print("FinaleMe Pearson r   :", round(hm.pearson_r(sim.beta_sample, pred_baseline), 3))
print("HAVI-Methyl Pearson r:", round(hm.pearson_r(sim.beta_sample, pred_havi), 3))
```

`fit_svi_simplified` is the empirical-Bayes Gaussian-posterior variant
used for the synthetic recovery benchmark of §11. It runs in seconds on
CPU and is the right tool for plumbing checks, the conformal A5
ablation, and the $N=20$-seed multi-seed sweep.

## Full torch SVI loop

```python
import havi_methyl as hm
from havi_methyl import TorchSVIConfig, fit_svi_torch, predict_with_torch_state

sim = hm.simulate_dataset_chromatin_aware(S=16, L=400, coverage=2.0, rng=20260429)

cfg = TorchSVIConfig(
    in_dim=5,           # fragment-feature dimension
    hidden=32,
    num_inducing=16,    # ISAB inducing points
    num_layers=2,       # ISAB stack depth
    kappa=20.0,         # Beta-Binomial concentration
    posterior="gaussian",  # or "flow" for the Conditional NSF head
    k_iwae=4,           # IWAE samples (DReG-tightened below)
    iwae_dreg=True,     # Tucker-2019 doubly-reparameterised gradient
)

state = fit_svi_torch(
    bags=sim.bags,
    n_frag=sim.n,          # WGS fragment counts (encoder feature)
    n_meth=sim.n_meth,     # Beta-Binomial successes
    n_obs=sim.n_total,     # Beta-Binomial trials (= WGBS coverage in real data)
    n_iter=200,
    config=cfg,
    seed=20260429,
)

pred, var = predict_with_torch_state(state, sim.bags, sim.n)
print("Pearson r:", round(hm.pearson_r(sim.beta_sample, pred), 3))
print("ELBO trajectory length:", len(state.elbo_history))
```

This is the loop that lands the headline real-data row (Pearson
$r = 0.467$ on Liu 2024, $500$-iteration A10G run). The same code path
with `tissue_reference`/`tissue_target`/`tissue_weight` kwargs runs the
variance-weighted Dirichlet head jointly (see
[Tissue-of-origin](tissue.md)).

!!! note "Trials vs fragments"
    For the synthetic simulator, every fragment is one Beta-Binomial
    trial so `n_obs = n_frag`. For *real* cfDNA paired data this
    identity does not hold: WGBS read coverage at the CpG is the
    Beta-Binomial trial count, distinct from the WGS fragment count
    that feeds the encoder. Always pass `ds.n_total` from
    `load_finaleme_dataset` as `n_obs=`; the BB-trials bug fix is what
    lifted HAVI-Methyl on Liu 2024 from $r = -0.07$ to $r = 0.467$
    (see the [Changelog](changelog.md)).

## Reproducing the manuscript

```bash
bash scripts/run_all.sh                 # canonical pipeline + every CSV + figure
bash scripts/run_all.sh --fast          # smoke run (S=4, L=80, n_iter=3)
bash scripts/run_all.sh --figures       # only figures
bash scripts/run_all.sh --tables        # only tables
bash scripts/run_all.sh --benchmarks    # only benchmarks
```

`scripts/bench_synth_recovery.py` is the single source of truth for
`outputs/results.json` + `outputs/plot_data.npz`; every Sec. 11 table
and figure derives from those two artifacts and stays byte-identical
between `outputs/` and `docs/report/`. See
[Reproducibility](reproducibility.md) for the full pipeline graph.

## Real-data benches

The Phase 5 benches consume real datasets via the loaders in
`havi_methyl.io`:

=== "Liu 2024 paired"

    ```bash
    python3 scripts/bench_finaleme_realdata.py \
        --data-dir /path/to/finaleme \
        --manifest data/finaleme_manifest/sample_pairs.csv \
        --locus-panel data/finaleme_manifest/high_variance_cpgs.hg19.bed \
        --buffy-coat-bw /path/to/wgbs_buffyCoat_jensen2015GB.methy.hg19.bw \
        --torch-svi --torch-iter 500 --torch-device cuda --torch-iwae-k 4 \
        --torch-snapshot-every 20
    ```

=== "Loyfer LOO"

    ```bash
    python3 scripts/bench_tissue_loo.py \
        --atlas-tsv data/loyfer_panel/Atlas.U25.l4.hg38.tsv \
        --n-tissues 36 --loci 900 --samples 50
    python3 scripts/fig_loyfer_loo_per_tissue.py
    ```

The high-variance CpG panel is committed to the repo; re-running
`scripts/build_high_variance_panel.py` is rarely needed. See
[Datasets](datasets.md) for the full loader contract.

## Next steps

- [Concepts](concepts.md) — three observation regimes, the three-level
  hierarchy, fragmentomic rationale.
- [Architecture](architecture.md) — `TorchSVIConfig` fields,
  Set Transformer + NSF + Beta-Binomial, IWAE-DReG, gradient reversal.
- [Results](results.md) — the §12 figures and numbers embedded inline.
