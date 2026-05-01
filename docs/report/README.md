# HAVI-Methyl: Hierarchical Amortized Variational Inference for cfDNA Methylation Prediction (SAIM template)

This is the SAIM-styled version of the HAVI-Methyl manuscript, ported from the earlier Cerebras-styled deliverable. It also incorporates research-mode factual corrections verified against primary literature and 27 newly added primary references that deepen the related-work and methods sections.

## Files

- `main.tex` — top-level document; orchestrates the modular section files.
- `main.pdf` — pre-compiled PDF, 52 pages, clean build.
- `saim.cls` — SAIM document class (with two minor host-environment adaptations; see `CHANGELOG-saim.md` §A.11).
- `math_commands.tex` — math macros, including HAVI-Methyl extensions appended at the bottom.
- `latexmkrc` — build configuration.
- `refs.bib` — bibliography with the original entries plus 27 new primary references.
- `CHANGELOG.md` — changelog from the original Cerebras-styled revision pass.
- `CHANGELOG-saim.md` — changelog for this SAIM port plus research-mode corrections plus new literature.
- `sections/` — body sections, one file per `\section`.
- `sections/appendix/` — appendices A through I.
- `tables/` — reserved for extracted table fragments; currently empty.
- `figures/` — generated figures embedded in the manuscript.
- `code/run_experiments.py` — simulator and experiment harness.
- `code/results.json` — numerical results from the most recent harness run.
- `results.json` — copy of the same file at the project root (the manuscript references either path).
- `assets/saim_logo.png` — SAIM logo for the title block.

## Compilation

```bash
latexmk -pdf main.tex
```

or explicitly:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Reproducing the experiments

```bash
cd code
python3 run_experiments.py
```

Runtime is approximately two minutes on a single CPU thread. The harness regenerates `results.json` and the three figures. Random seed is fixed, so numerical claims in the manuscript reproduce exactly.

## Required LaTeX packages

Standard TeX Live (`texlive-full` on Ubuntu, MacTeX on macOS), with the SAIM class supplied in this project. The class loads `geometry, microtype, placeins, hyphenat, setspace, parskip, babel, etoolbox, graphicx, subcaption, booktabs, nicematrix, multirow, bm, tcolorbox, xcolor, hyperref, cleveref, natbib, titlesec, caption, fontenc`. The top-level `main.tex` additionally loads `url, enumitem, float, algorithm2e, longtable, tabularx, listings, tikz` and the `tikz-bayesnet` library. If `tikz-bayesnet` is missing, install via `tlmgr install tikz-bayesnet` or download `bayesnet.sty` from CTAN.

## Overleaf

Upload the contents of this directory to a new Overleaf project. Set the compiler to **pdfLaTeX** and the bibliography tool to **BibTeX**. No further configuration required.

## Manuscript structure

The 52-page manuscript covers an introduction motivated against seven specific FinaleMe limitations; background sections on cfDNA fragmentomics, variational Bayes, and probabilistic methylation prediction; a three-level hierarchical generative model; an amortized normalizing-flow variational family with a Set Transformer encoder; a full ELBO derivation; an SVI training algorithm; identifiability analysis via VIB, mQTL anchors, and counterfactual augmentation; calibration via conformal prediction with conformal risk control; a joint tissue-of-origin head with HDP nonparametric extension; a chromatin-aware cfDNA simulator; synthetic experiments with reproducible measurements; a benchmarking plan; seven formal propositions with full proofs (including the new variance-reduction and Fano-style recoverability bounds); a discussion section; an extended related-work survey covering the new fragmentomics, probabilistic-methylation-prediction, foundation-model, advanced-VI, and counterfactual-causal-ML literature; a conclusion; and nine appendices.
