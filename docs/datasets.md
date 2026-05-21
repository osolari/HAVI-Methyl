# Datasets and loaders

The Phase 5 real-data benches consume three external sources through
the loaders in `havi_methyl.io`. Every loader returns a typed dataclass
with the on-disk provenance recorded in its `source` field, so figure
captions can show what was loaded.

| Loader | Upstream | Returned object | §12 case study |
|---|---|---|---|
| `load_finaleme_dataset` | Liu 2024 paired cfDNA WGS/WGBS (dbGaP phs003287.v1.p1 / Zenodo 7779198), paired via Supplementary Table 1 | `FinaleMeDataset` | Liu 2024 paired methylation (§12.4) |
| `load_loyfer_atlas_matrix` | `nloyfer/UXM_deconv` `Atlas.U25.l4.hg{19,38}.tsv` (GSE186458) | $(T\times L)$ reference matrix + locus metadata | Loyfer LOO tissue-of-origin (§12.4) |
| `load_loyfer_pat_directory` | UXM_deconv `.pat.gz` per-sample CpG methylation files | per-sample CpG calls | optional locus-level comparisons |
| `load_roadmap_wgbs_atlas` | Roadmap Epigenomics WGBS BED tracks | tissue-keyed BED arrays | atlas-prior initialization candidates |

## Liu 2024 paired panel

```python
from havi_methyl import load_finaleme_dataset

ds = load_finaleme_dataset(
    path="/Volumes/Omid Solari/finaleme",
    manifest="data/finaleme_manifest/sample_pairs.csv",
    locus_panel="data/finaleme_manifest/high_variance_cpgs.hg19.bed",
    buffy_coat_bw="/path/to/wgbs_buffyCoat_jensen2015GB.methy.hg19.bw",
)

ds.sample_ids        # list[str]      length S
ds.locus_chrom       # list[str]      length L
ds.bags              # list[list[np.ndarray]]   S x L, each (n, in_dim)
ds.n                 # np.ndarray     (S, L) WGS fragment counts  (encoder feature)
ds.n_meth            # np.ndarray     (S, L) WGBS methylated reads (BB successes)
ds.n_total           # np.ndarray     (S, L) WGBS read coverage    (BB trials -- NOT ds.n)
ds.beta_sample       # np.ndarray     (S, L) WGBS-derived methylation fraction
```

The expected directory layout is

```
<path>/frag_wgs/*.tsv.gz       per-fragment WGS features
<path>/meth_wgbs/*.bed.gz      WGBS CpG calls (strand 6plus2)
```

### Manifest-based pairing

The `manifest=` kwarg points at the Liu 2024 Supplementary Table 1
manifest (committed at `data/finaleme_manifest/sample_pairs.csv`).
Pairing is via the flowcell id (`FC########`) embedded in filenames,
joined against the manifest's `WGS_library_id` and `WGBS_library_id`
columns. Without a manifest, the loader falls back to filename-id
stripping, which only works when WGS and WGBS share the same id stem
(e.g. `HD_45`). macOS AppleDouble sidecars (`._<file>`) are filtered
out before pairing. After variance filtering and manifest pairing the
realised panel is $S = 77$ patients $\times$ $L = 782$ CpGs.

!!! warning "BB trials = ds.n_total, not ds.n"
    The most expensive lesson of Phase 5 is that the Beta-Binomial
    trials parameter must be `ds.n_total` (WGBS coverage), not
    `ds.n` (WGS fragment count). They are *different observation
    streams*: WGBS coverage measures direct methylation reads, WGS
    fragment count measures cfDNA fragments visible to the encoder.
    Using `ds.n_total` (correctly) lifts HAVI-Methyl from $r = -0.07$
    to $r = 0.467$ on the Liu 2024 panel ($500$-iteration A10G run);
    the same loop with the wrong stream is negatively correlated with
    truth. See [Changelog](changelog.md) for the full incident.

### Jensen 2015 buffy-coat prior

The `buffy_coat_bw=` kwarg points at the Jensen 2015 cfDNA WGBS prior
(`wgbs_buffyCoat_jensen2015GB.methy.hg19.bw`, Zenodo 7647046). When
present, the loader extracts the per-locus buffy-coat methylation
level at each panel CpG and the encoder concatenates it into the
context vector. This is the prior input that the VIB and mQTL
identifiability terms guard against leaking.

## Loyfer U25 panel

```python
from havi_methyl import load_loyfer_atlas_matrix

R, locus_meta = load_loyfer_atlas_matrix(
    tsv_path="data/loyfer_panel/Atlas.U25.l4.hg38.tsv",
    max_tissues=36,
    max_loci=900,
)

R.shape       # (T=36, L=900)
locus_meta    # list of dicts with chrom / start / end / direction
```

The Phase 5 loader update aligned the defaults with the real UXM_deconv
panel schema: `chr`/`chrom` alias, auto-drop of
`startCpG`/`endCpG`/`target`/`direction` columns, and the same macOS
`._` AppleDouble filtering used by the Liu 2024 loader. On the full
panel (36 cell types $\times$ 900 marker blocks), the
variance-weighted Dirichlet head wins LOO RMSE at every one of the 36
cell types — see [Tissue-of-origin](tissue.md).

## High-variance CpG panel

The default 782-CpG panel committed at
`data/finaleme_manifest/high_variance_cpgs.hg19.bed` is built by

```bash
python3 scripts/build_high_variance_panel.py \
    --meth-dir /path/to/finaleme/meth_wgbs \
    --out data/finaleme_manifest/high_variance_cpgs.hg19.bed \
    --top-n 1000 \
    --min-cov 1 \
    --min-presence-frac 0.40 \
    --chroms chr1,chr19,chr20,chr21,chr22
```

The selection criterion is per-CpG **cross-patient methylation
variance**, restricted to CpGs with WGBS coverage $\ge 1$ in $\ge 40\%$
of samples. The realised top-782 panel passes both filters; the
committed BED is the panel the headline real-data row of
`bench_finaleme_realdata.csv` was trained on. Regenerating it is
rarely needed.

## Dataset summary (manuscript Table 13)

| Dataset | $S$ / scope | Modality | Role | Access |
|---|---|---|---|---|
| Liu 2024 | $77$ patients, $L=782$ CpGs | paired cfDNA WGS + WGBS (shallow) | methylation prediction target | dbGaP phs003287.v1.p1 / Zenodo 7779198 |
| Loyfer 2023 (U25) | $36$ cell types, $L=900$ marker blocks | WGBS atlas (`Atlas.U25.l4.hg38`) | ToO reference | GSE186458 / `nloyfer/UXM_deconv` |
| Jensen 2015 | buffy-coat reference | WGBS bigwig (single sample) | per-locus methylation prior | Zenodo 7647046 |
