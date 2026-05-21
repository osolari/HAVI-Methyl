# HAVI-Methyl: Hierarchical Amortized Variational Inference for cfDNA Methylation Prediction

This is a publication-development LaTeX project for the HAVI-Methyl manuscript. The project contains the proposed full probabilistic framework, completed simplified synthetic experiments, planned real-data benchmark material, generated figures, sidecar tables, and implementation handoff notes.

## Current status

- The manuscript specifies the full HAVI-Methyl architecture: hierarchical logit-scale methylation model, amortized fragment-bag encoder, normalizing-flow posterior, leakage-control losses, conformal calibration, and tissue-of-origin head.
- The released code now ships the **full torch SVI loop** (Set Transformer encoder + Gaussian or Conditional NSF posterior head + Beta-Binomial reconstruction + Robbins-Monro recentering + true gradient-reversal sample-id discriminator + joint variance-weighted Dirichlet ToO head + Tucker-2019 DReG-IWAE estimator) end-to-end on synthetic and real Liu 2024 paired data, alongside the simplified-numpy harness used for the multi-seed synthetic recovery benchmark.
- Real-data results are measured: Liu 2024 paired ($r=0.467$ vs FinaleMe $r=0.078$, 500-iter A10G run), Loyfer U25 LOO (HAVI Dirichlet head wins 36/36 cell types), App. H simulator validation (5/5 axes verified). The only remaining research direction is prospective clinical validation under regulatory protocol (cf. §14).

## Key files

- `main.tex` - root LaTeX file.
- `main.pdf` - compiled PDF from the current source build.
- `saim.cls` - SAIM document class supplied with this project.
- `math_commands.tex` - math macros and theorem environments.
- `latexmkrc` - latexmk configuration.
- `refs.bib` - BibTeX bibliography.
- `sections/` - manuscript sections.
- `sections/appendix/` - appendices A-I.
- `figures/` - generated figures embedded in the manuscript.
- `tables/` - sidecar CSV tables synchronized to the current manuscript/result status where applicable.
- `code/run_experiments.py` - simplified synthetic simulator and experiment harness.
- `results.json` and `code/results.json` - current fixed-seed synthetic results used by the manuscript.
- `CHANGELOG.md` - changelog for the Phase Three revision pass.
- `CHANGELOG-saim.md` - historical SAIM-port changelog retained for provenance.
- `CODING_AGENT_HANDOFF.md` - implementation/evaluation handoff for the next coding-agent iteration.

## Compilation

Use pdfLaTeX plus BibTeX through latexmk:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Equivalent explicit sequence:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

On Overleaf, set compiler to **pdfLaTeX** and bibliography tool to **BibTeX**.

## Reproducing the simplified synthetic experiments

Run from either the repository root or the `code/` directory:

```bash
python3 code/run_experiments.py
```

or:

```bash
cd code
python3 run_experiments.py
```

The script resolves paths relative to the repository root and writes `results.json`, `code/results.json`, and the three manuscript figures under `figures/`. Random seed is fixed at `20260429`.

## Important interpretation constraints

The synthetic results in §11 are completed measured results from the simplified-numpy harness; the §12 real-data results come from the full torch SVI loop on the published Liu 2024 paired panel and the published Loyfer/UXM_deconv U25 atlas. External-baseline rows in `tables/bench_finaleme_realdata.csv` (FinaleMe upstream, DeepCpG, Elastic-net, MethylBERT) remain `XX` placeholders pending each project's own codebase being aligned to the 782-CpG panel; the HAVI-Methyl-and-FinaleMe-style-HMM-reimplementation rows are real Liu 2024 numbers.
