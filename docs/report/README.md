# HAVI-Methyl: Hierarchical Amortized Variational Inference for cfDNA Methylation Prediction

This is a publication-development LaTeX project for the HAVI-Methyl manuscript. The project contains the proposed full probabilistic framework, completed simplified synthetic experiments, planned real-data benchmark material, generated figures, sidecar tables, and implementation handoff notes.

## Current status

- The manuscript specifies the full HAVI-Methyl architecture: hierarchical logit-scale methylation model, amortized fragment-bag encoder, normalizing-flow posterior, leakage-control losses, conformal calibration, and tissue-of-origin head.
- The included code implements a simplified synthetic harness used for the completed fixed-seed synthetic results. It is not the full Set Transformer + normalizing-flow implementation.
- Real-data benchmarking, full ablations, conformal calibration evaluation, full tissue-head training, and simulator validation plots remain planned implementation tasks.

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

The synthetic results are completed results from the simplified harness. They must not be described as completed real-data results or as completed evaluations of the full Set Transformer + normalizing-flow architecture. Real-data benchmark tables with placeholder values are intentional planning artifacts and should remain clearly labeled until experiments are run.
