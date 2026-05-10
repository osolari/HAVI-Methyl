# CODING_AGENT_HANDOFF.md

## 1. Project Overview

HAVI-Methyl is a LaTeX manuscript and prototype repository for hierarchical amortized variational inference of methylation from cfDNA fragmentomics. The proposed full method uses hierarchical logit-scale methylation latents, a fragment-bag encoder, a conditional local posterior, de-confounding losses, conformal calibration, and a tissue-of-origin head.

Current manuscript status: publication-development-ready draft. Completed empirical material is limited to a fixed-seed simplified synthetic harness. The full Set Transformer + normalizing-flow model, real-data benchmark, conformal evaluation, full tissue head, simulator validation artifacts, and compute profiling remain implementation and evaluation tasks.

## 2. Build Instructions

Root LaTeX file: `main.tex`.

Document class: `saim.cls`.

Bibliography file: `refs.bib`.

Bibliography style/tool: `plainnat` with BibTeX.

Recommended build command:

```bash
latexmk -pdf main.tex
```

Equivalent explicit build:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Required project files: `main.tex`, `math_commands.tex`, `saim.cls`, `refs.bib`, all `sections/*.tex`, all `sections/appendix/*.tex`, `figures/*`, `assets/saim_logo.png`, and `latexmkrc`.

Known caveat from Phase One: the container's default `bibtex` executable path was broken, but `/usr/bin/bibtex.original` worked. Overleaf should use normal BibTeX.

## 3. Implementation Tasks

### IMPL-01 Full repository path handling and experiment artifact policy
Description: keep all experiment outputs rooted at the repository root and write both `results.json` and `code/results.json`.
Related manuscript section: Section 11.
Expected input: command-line invocation from root or `code/`.
Expected output: synchronized JSON, figures, and sidecar tables.
Dependencies: Python, NumPy, SciPy, Matplotlib.
Priority: high.
Completion criteria: running `python3 code/run_experiments.py` from root and from `code/` produces identical outputs.

### IMPL-02 Full fragment-bag encoder
Description: implement ISAB/PMA Set Transformer for fragment bags.
Related manuscript section: Sections 4 and 6.
Expected input: per-locus fragment feature tensors with masks.
Expected output: context vector `c_frag`.
Dependencies: PyTorch or JAX, batching/masking utilities.
Priority: high.
Completion criteria: unit tests verify permutation invariance and mask handling.

### IMPL-03 Sequence-context encoder module
Description: implement dilated CNN and frozen long-range DNA encoder projection options.
Related manuscript section: Section 4.
Expected input: local sequence windows or precomputed embeddings.
Expected output: `c_seq` context vector.
Dependencies: genome FASTA loader, optional pretrained encoder.
Priority: medium.
Completion criteria: reproducible embeddings with documented coordinates and strand conventions.

### IMPL-04 Conditional normalizing-flow local posterior
Description: implement scalar conditional NSF posterior for `eta_{s,l}`.
Related manuscript section: Sections 4, 5, Appendix C.
Expected input: context vector and base noise.
Expected output: samples, log probabilities, posterior summaries.
Dependencies: tested flow library or custom spline implementation.
Priority: high.
Completion criteria: density integrates numerically, gradients pass finite-difference checks, boundaries are stable.

### IMPL-05 Approximate SVI population/sample updates
Description: implement moment-matched updates for population and sample-shift variational parameters.
Related manuscript section: Sections 5 and 6.
Expected input: local posterior moments and mini-batch weights.
Expected output: updated `lambda_l` and `nu_s`.
Dependencies: optimizer state, sampling weights.
Priority: high.
Completion criteria: global recentering residual near zero; synthetic Gaussian tests match conjugate posterior.

### IMPL-06 De-confounding losses
Description: implement VIB, matched counterfactual prior swaps, mQTL anchor loss, and domain-adversarial classifier.
Related manuscript section: Section 7.
Expected input: prior inputs, matched pairs, genotype anchors, cohort/batch labels.
Expected output: loss components and diagnostics.
Dependencies: genotype processing, anchor map, balance diagnostics.
Priority: high for real-data benchmark.
Completion criteria: losses are dimensionally consistent and logged separately; balance diagnostics pass pre-specified thresholds.

### IMPL-07 Conformal calibration wrapper
Description: implement split-conformal density-set and CQR-style interval calibration.
Related manuscript section: Section 8.
Expected input: held-out calibration predictions and ground truth.
Expected output: conformal thresholds, coverage metrics, worst-stratum coverage.
Dependencies: disjoint calibration split.
Priority: high.
Completion criteria: finite-sample rank formula is used; synthetic exchangeability test achieves nominal marginal coverage.

### IMPL-08 Tissue-of-origin head
Description: implement finite Dirichlet head first, then optional HDP truncation.
Related manuscript section: Section 9.
Expected input: posterior methylation summaries and reference matrix.
Expected output: tissue-fraction posterior summaries and calibration diagnostics.
Dependencies: reference atlas, tissue labels, optional ground truth.
Priority: medium-high.
Completion criteria: synthetic mixtures recover fractions; uncertainty is reported; leave-one-tissue-out stress test runs.

### IMPL-09 Full simulator
Description: replace compact synthetic feature generator with full chromatin-aware simulator.
Related manuscript section: Section 10 and Appendices E/H.
Expected input: genome, chromatin tracks, methylation tracks, parameters.
Expected output: fragment bags, ground truth methylation, validation artifacts.
Dependencies: genome and annotation loaders.
Priority: medium.
Completion criteria: validation plots/tables generated for length, motifs, periodicity, and methylation-conditioned bias.

### IMPL-10 Table and figure regeneration pipeline
Description: generate all manuscript CSVs and figures from JSON/experiment artifacts.
Related manuscript section: Sections 11-12 and Appendices H-I.
Expected input: experiment run directory.
Expected output: `tables/*.csv`, `figures/*.png`, `results.json`.
Dependencies: plotting scripts and table writer.
Priority: high.
Completion criteria: no table values are manually stale relative to JSON.

## 4. Experiment Plan

Completed experiments: fixed-seed simplified synthetic harness with `S=12`, `L=300`, coverages `0.1x`, `1x`, `5x`, `30x`, synthetic prior-leakage stress test, and continuous tissue-deconvolution proxy.

Planned real-data experiments:

- Paired WGS/WGBS methylation prediction benchmark against FinaleMe-style and additional baselines.
- Nested HAVI-Methyl ablations: no hierarchy, Gaussian posterior, no de-confounding losses, no conformal wrapper, no joint ToO head.
- Calibration diagnostics: raw posterior coverage, conformal density sets, CQR intervals, coverage by stratum, worst-stratum coverage.
- Robustness strata: coverage, CpG density or low-information proxy, chromatin state, GC, mappability, tumor fraction/CNV, cohort, ancestry, batch.
- Tissue-of-origin evaluation: continuous deconvolution baselines, finite Dirichlet head, posterior calibration, leave-one-tissue-out stress test.
- Simulator validation: length modes, motif frequencies, periodicity, methylation-conditioned cut bias, multi-seed uncertainty.
- Multi-seed synthetic sensitivity: `kappa`, `sigma_eta`, `sigma_delta`, number of samples, number of loci, coverage, flow depth.

Success criteria: improved or competitive methylation prediction relative to direct baselines; calibrated conformal coverage; clear failure-case reporting; consistent table/figure artifacts; reproducible splits; no leakage from evaluation targets into atlas initialization.

## 5. Figures and Tables to Generate

- `fig:recovery` (`figures/recovery_scatter.png`): currently generated by simplified harness; regenerate after full model experiments.
- `fig:calibration` (`figures/calibration.png`): currently simplified Gaussian calibration; replace with raw plus conformal calibration curves.
- `fig:elbo` (`figures/elbo_trajectory.png`): currently surrogate objective; replace or relabel after full ELBO logging exists.
- `tab:recovery`: currently synchronized to fixed-seed JSON; regenerate from final experiment artifacts.
- `tab:ext-recovery`: currently fixed-seed simplified harness; add confidence intervals after multi-seed runs.
- `tab:ext-ident`: currently synthetic prior-leakage stress test; add real-data diagnostics once available.
- `tab:ext-tissue`: currently continuous deconvolution proxy; replace with full Dirichlet-head evaluation when implemented.
- `tab:datasets`: planning table requiring external verification.
- `tab:ablations`: planned ablation matrix; populate with observed metrics only after experiments run.
- Simulator validation plots/tables: currently targets/placeholders; generate from full simulator.

## 6. Projected or Expected Results

Do not treat planned benchmark tables, simulator validation targets, compute budgets, or full architecture ablation expectations as observed results. After experiments are completed, replace planned/projected language with observed language only where the data artifact, script, seed, split, and commit are available. Any numerical claim must be traceable to a JSON/CSV artifact generated by code.

## 7. Theory-to-Code Connections

- Hierarchical pooling proposition -> compare no-hierarchy ablation and low-coverage strata.
- Fano recoverability bound -> stratify by coverage, low-information proxy, mappability, and fragment count.
- VIB leakage proposition -> log prior-input information proxy and partial-attribution metrics.
- mQTL identifiability proof sketch -> implement anchor mapping, ancestry/batch adjustment, and exclusion-risk diagnostics.
- Conformal coverage proposition -> use disjoint calibration splits and exact rank quantiles.
- SVI surrogate convergence -> log natural-parameter drift, recentering residuals, and held-out objective.
- ToO expected log-likelihood -> implement expected-log-likelihood objective consistently with maximization/minimization signs.

## 8. Open Technical Questions

- Which real-data accessions and exact sample counts are available and approved for use?
- Which methylation ground truth is available for each dataset and split?
- What is the final definition of CpG-poor versus low-information regions?
- Which flow library and spline boundary convention will be used?
- How will atlas initialization avoid leakage into benchmark targets?
- Which mQTL anchors remain valid after ancestry and batch adjustment?
- How will high-density conformal sets be represented as intervals, if required?
- Which tissue references and ground-truth fractions are available for ToO evaluation?
- What hardware and data-loader stack will be used for measured compute profiling?
- Which external citation claims require verification before final submission?

## 9. Files Changed or Added

Changed: `README.md`, `CHANGELOG.md`, `main.tex`, `code/run_experiments.py`, `sections/0-abstract.tex`, `sections/1.intro.tex`, `sections/2.background.tex`, `sections/3.model.tex`, `sections/4.varfamily.tex`, `sections/5.elbo.tex`, `sections/6.algorithm.tex`, `sections/7.identifiability.tex`, `sections/8.calibration.tex`, `sections/9.tissue.tex`, `sections/10.simulator.tex`, `sections/11.synth.tex`, `sections/12.benchmark.tex`, `sections/13.theory.tex`, `sections/14.discussion.tex`, `sections/15.related.tex`, `sections/16.conclusion.tex`, all appendices A-I, and multiple `tables/*.csv` sidecars.

Added: `CODING_AGENT_HANDOFF.md`.

Preserved: `saim.cls`, `math_commands.tex`, `refs.bib`, `latexmkrc`, figures, assets, and historical `CHANGELOG-saim.md`.

## 10. Do-Not-Change Constraints

Preserve unless explicitly approved by the author:

- The HAVI-Methyl theoretical direction and hierarchical amortized VI framing.
- The planned full Set Transformer + normalizing-flow architecture.
- Planned experiments, benchmark protocols, ablations, expected outputs, and placeholder/status tables.
- The distinction between completed simplified synthetic results and planned full-model/real-data results.
- Generated synthetic figures unless replacing them with regenerated artifacts from code.
- Template, document class, bibliography style, and macro conventions.
- Existing citation keys unless citation verification justifies additions/removals.
- No numerical result may be inserted unless it is generated by a reproducible artifact.
