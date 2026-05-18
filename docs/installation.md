# Installation

HAVI-Methyl runs on Python $\ge$ 3.10.

## Three install routes

| Route | Adds | Use case |
|---|---|---|
| `make install` | `numpy`, `scipy` | Core library: closed-form distributions, ELBO, simplified-numpy SVI, baseline classifiers, calibration / identifiability / tissue helpers, theoretical-bound utilities. Everything under `havi_methyl.{distributions, likelihoods, model, varfamily, elbo, svi, identifiability, calibration, tissue, simulator, baseline, bounds, pipeline}` works under this minimal install. |
| `make install-dev` | + `matplotlib`, `pandas`, `pytest`, `ruff`, `mypy` | Developer workflow: run the 156-test pytest suite, the figure / table scripts under `scripts/`, and the linters. |
| `make install-torch` | + `torch` | Full production stack: Set Transformer encoder (`encoders.py`), Conditional NSF flow head (`flow.py`), and the end-to-end `fit_svi_torch` / `predict_with_torch_state` loop. Without it, the same math is exercised through the simplified-Gaussian variant of Sec. 11. |

The torch install is what lands the headline real-data Pearson $r = 0.455$
on Liu 2024; the numpy install reproduces the §11 synthetic recovery
benchmark and the A0..A5 ablation matrix end-to-end.

## Data setup

The Phase 5 real-data benches consume three external sources. None of
them are bundled in the package — the loaders just provide a consistent
schema once they are present on disk.

### Liu 2024 paired cfDNA WGS/WGBS

The published manifest (Liu 2024 Nature Communications Supplementary
Table 1) is committed at
`data/finaleme_manifest/sample_pairs.csv`. The expected on-disk layout
is

```
<root>/frag_wgs/*.tsv.gz       per-fragment WGS features (BED6+motif)
<root>/meth_wgbs/*.bed.gz      WGBS CpG calls (strand 6plus2)
```

Pairing is via the embedded flowcell id (`FC########`) in the
filenames, joined against the manifest's `WGS_library_id` and
`WGBS_library_id` columns. macOS AppleDouble sidecars (`._<file>`) are
filtered out before pairing.

### Loyfer U25 panel

The hg19 / hg38 panel tables ship in `nloyfer/UXM_deconv` under
`Atlas.U25.l4.hg{19,38}.tsv`. The loader matches the upstream schema
(`chr`/`chrom` alias, auto-drops `startCpG`/`endCpG`/`target`/`direction`)
and returns a $T \times L$ reference matrix of 36 cell types $\times$
900 hypomethylated marker blocks.

### Jensen 2015 buffy-coat WGBS

The cfDNA WGBS buffy-coat reference (`wgbs_buffyCoat_jensen2015GB.methy.hg19.bw`,
Zenodo 7647046) provides the per-locus methylation prior consumed by
the encoder when `--buffy-coat-bw` is passed to
`scripts/bench_finaleme_realdata.py`.

## High-variance CpG panel

The default 782-CpG panel committed at
`data/finaleme_manifest/high_variance_cpgs.hg19.bed` is built by

```bash
python3 scripts/build_high_variance_panel.py \
    --meth-dir /path/to/finaleme/meth_wgbs \
    --out data/finaleme_manifest/high_variance_cpgs.hg19.bed \
    --top-n 1000 --min-cov 1 --min-presence-frac 0.40 \
    --chroms chr1,chr19,chr20,chr21,chr22
```

Re-running is rarely needed — the committed panel is what the headline
real-data row of `bench_finaleme_realdata.csv` was trained on.

## Verifying the install

```python
import havi_methyl as hm

print(hm.__version__ if hasattr(hm, "__version__") else "(local)")
print(sorted(name for name in dir(hm) if not name.startswith("_"))[:20])
```

You should see a list of public surfaces including
`fit_svi_simplified`, `simulate_dataset`, `simulate_dataset_chromatin_aware`,
`finaleme_baseline_predict`, `load_finaleme_dataset`,
`load_loyfer_atlas_matrix`, `dirichlet_head_predict`,
`evaluate_real_data_benchmark`, and `pearson_r`.

If the optional torch install is present you will additionally see
`TorchSVIConfig`, `fit_svi_torch`, and `predict_with_torch_state`.

## Test suite

```bash
make test
```

156 pytest tests pin every theoretical claim in the manuscript against
the implementation (closed-form KL formulas vs Monte Carlo, Fano bound
monotonicity in MI, hierarchical pooling shrinkage, conformal coverage
at the nominal level, simulator validation against published cfDNA
distributions, etc.). 9 are torch-conditional and skip on a numpy-only
install.

!!! warning "macOS Removable-Volumes TCC"
    The Phase 5 real-data benches read from `/Volumes/Omid Solari/finaleme/`.
    On macOS the running VS Code process must have explicit
    Files-and-Folders access to *Removable Volumes* before the lab
    drive becomes visible — re-launch VS Code from the Dock if a fresh
    install of `havi_methyl` cannot see the volume.
