# CHANGELOG - Phase Three Technical Development Pass

## Summary

Applied a publication-development pass across the LaTeX project, source code, sidecar CSVs, README, build documentation, and handoff documentation.

Approximate counts by category: A technical corrections 36; B theoretical-development changes 9; C rigor/completeness improvements 29; D clarity/organization improvements 10; E planned/projected/status clarifications 16; F new theory/method/experiment/handoff additions 14; G citation/positioning safeguards 9; H LaTeX/build/documentation fixes 15.

Technical-error fixes applied: notation separation between recognition inputs and likelihood observations; global recentering correction; mQTL genotype dimensionality; Negative-Binomial parameterization; ToO expected-log-likelihood formula; objective signs; proof-status refinements; Fano/MSE bound correction; simulator cut-density correction; code path handling; sidecar CSV synchronization.

Theoretical-development changes applied: explicit observation/sampling assumption; instrumental-anchor assumption; conditional identifiability proof status; feasible-reference VIB finite-penalty bound; surrogate convergence statement; logit-scale hierarchical pooling variance result; corrected Fano recoverability statement; theory-to-experiment remark.

Methodology-strengthening changes applied: clarified full model versus simplified harness; improved ELBO notation; separated posterior family from objective; clarified ISAB/PMA architecture; added calibration set representation details; clarified tissue-head status; added split/leakage-control guidance.

Experiment-plan changes applied: strengthened real-data benchmark protocol, metrics, baselines, ablations, robustness strata, calibration diagnostics, simulator validation tasks, and figure/table generation requirements.

Planned experiments preserved or clarified: real-data FinaleMe benchmark, full architecture ablations, conformal calibration, tissue-head evaluation, simulator validation, compute profiling, robustness checks, and multi-seed synthetic sensitivity analyses.

Expected/projected results preserved or clarified: no planned or projected result was converted into an observed empirical finding. Sidecar real-data benchmark placeholders remain placeholders.

Placeholder figures/tables preserved or clarified: generated synthetic figures retained; sidecar placeholder benchmark and validation tables clarified; table CSVs synchronized to current status.

Implementation-plan material preserved or clarified: full Set Transformer, normalizing-flow posterior, conformal wrapper, mQTL anchors, ToO head, data loaders, and benchmark automation preserved as implementation tasks.

Coding-agent-relevant tasks added or clarified: `CODING_AGENT_HANDOFF.md` added; code path handling fixed; CSV regeneration policy documented; full implementation and experiment roadmap added.

Proposed changes not applied: Bib-H1 removal of unused bibliography entries was not applied; unused bibliography entries were left in place to avoid accidental citation loss. No new bibliography entries were added.

New package/macro/auxiliary/bibliography additions: none. New markdown file added: `CODING_AGENT_HANDOFF.md`. No new LaTeX package, macro, figure file, bibliography entry, or structural project change was introduced.

## Applied changes in document order

### Project-level and build
- Proj-A1 - README/build documentation: documented page-count provenance and source/PDF distinction.
- Proj-A2 / Code-A1 - README and `code/run_experiments.py`: corrected experiment execution path handling by resolving outputs relative to repository root.
- Proj-A3 / Code-C2 - `tables/*.csv`, `results.json`, `code/results.json`: synchronized stale sidecar CSV tables to the current JSON result values or status labels.
- Proj-C1 - README: clarified full proposed architecture versus released simplified harness.
- Proj-F1 / Code-F1 - `CODING_AGENT_HANDOFF.md`: added full coding-agent handoff with implementation, experiment, figure/table, and do-not-change tasks.
- Proj-H1 - `CHANGELOG.md`: replaced prior root changelog with this Phase Three changelog while preserving `CHANGELOG-saim.md`.
- Proj-H2 / Build-H3 - packaging: prepared clean Overleaf-ready deliverables.
- Proj-H3 / Build-H1 / Build-H2 - build: performed final pdfLaTeX + BibTeX compile and render verification; adjusted listing keyword style to avoid typewriter-bold font substitution warnings.

### Abstract and Introduction
- 0-E1 / 0-A1 / 1-C1 / 1-F1 - abstract and introduction: distinguished proposed full model, released simplified synthetic harness, completed synthetic results, and planned real-data/full-architecture validation.
- 0-B1 / 13-B1 - contribution language: aligned proof-status language with revised theory section.
- 0-G1 / 1-G1 - benchmark and foundation-model language: preserved positioning while avoiding new unverified external claims.
- 1-A1 / 1-A2 - limitations framing: qualified FinaleMe and prior-leakage claims.
- 1-D1 - introduction organization: improved flow from motivation to contributions and status.

### Background and model
- 2-D1 - Section 2 title: reframed as background/preliminaries to reduce overlap with extended related work.
- 2-C1 / 2-F1 - Section 2: clarified observed methylation, pseudo-labels, fragmentomic features, and signal-to-component mapping.
- 2-G1 - citation handling: preserved citations and deferred verification tasks to handoff.
- 3-A1 / 3-D1 / App-G-D1 - model notation and glossary: separated `F_{s,l}` recognition inputs from `O_{s,l}` likelihood observations.
- 3-A2 - likelihood regimes: clarified direct Bernoulli evidence versus aggregate/pseudo Beta-Binomial evidence.
- 3-A3 - joint distribution: reconciled generative likelihood terms with recognition-model feature conditioning.
- 3-A4 / 6-A2 - recentering: corrected mini-batch recentering to global or running global recentering.
- 3-C1 / 5-A2 - Negative-Binomial convention: defined mean-dispersion parameterization and delegated derivative consistency to autodiff.
- 3-B1 - assumptions: added observation and sampling assumptions.
- 3-H1 - plate diagram: revised nodes and caption to match corrected notation and added label.

### Variational family, ELBO, algorithm
- 4-C1 - structured VI: clarified deterministic context approximation.
- 4-A1 - flow expressiveness: narrowed scalar-flow claims and avoided unsupported multimodality claims.
- 4-D1 - Set Transformer: standardized ISAB/PMA terminology.
- 4-C2 - variational-family table: relabeled IWAE as objective tightening.
- 4-F1 / 6-F1 - implementation status: linked full architecture to handoff tasks.
- 5-A1 / App-A-A1 - ELBO notation: updated reconstruction and KL notation to use likelihood observations `O`.
- 5-C1 - local KL: clarified mean plug-in versus posterior-predictive averaging.
- 5-A3 / 13-A3 - population updates: stated moment-matched approximate update and surrogate convergence conditions.
- 5-C2 - rescaling: added uniform-sampling assumption and nonuniform weighting caveat.
- 5-B1 / 9-A2 - total objective: added consistent maximization convention.
- 6-A1 - algorithm notation: separated AdamW learning rate from Robbins-Monro natural step size.
- 6-C1 - algorithm inputs/outputs: expanded to optional anchors, counterfactuals, tissue references, and calibration outputs.
- 6-E1 / 6-C2 / App-F-A1 - warm-start/hyperparameters: distinguished recommendations and planned defaults from completed harness settings.

### De-confounding, calibration, tissue
- 7-A1 - mQTL anchors: corrected genotype notation to sample-specific anchor genotypes and revised loss target.
- 7-B1 / 7-B2 / 13-A1 - identifiability: added IV assumptions and downgraded overstrong conclusions to conditional proof-sketch status.
- 7-A2 - VIB: clarified that VIB limits specified prior-input leakage rather than all confounding.
- 7-C1 / 7-E1 - counterfactuals and stress test: clarified matching diagnostics and synthetic-test interpretation.
- 8-A1 - conformal quantile: stated exact rank/order-statistic form.
- 8-C1 - nonconformity sets: distinguished high-density conformal sets from intervals.
- 8-E1 / 8-F1 - calibration plan: labeled planned reporting and added stratified diagnostics.
- 9-A1 / 9-C1 - ToO formula: corrected expected log-likelihood form and reference-matrix dimensions.
- 9-E1 / 9-A3 - ToO implementation status: labeled HDP and full Dirichlet-head evaluation as planned; clarified synthetic proxy.
- 9-F1 - ToO evaluation tasks: added to handoff.

### Simulator, synthetic experiments, benchmark
- 10-A1 / App-E-A2 - simulator status: separated full simulator spec from simplified released harness.
- 10-E1 / App-H-E1 - validation: labeled validation numbers/artifacts as targets or pending regeneration.
- 10-C1 / 10-F1 / App-H-F1 - simulator reproducibility: added validation-generation tasks.
- 10-G1 - parameter-source claims: preserved but deferred verification.
- 11-E1 / 11-E2 / App-I-E1 - synthetic status: labeled all current results as simplified fixed-seed synthetic results.
- 11-A1 / App-I-A1 - sidecar results: synchronized CSV values to `results.json`; marked sensitivity sweeps as planned because not in JSON.
- 11-A2 - CpG-poor wording: renamed to low-information proxy.
- 11-A3 - calibration: emphasized under-coverage and planned conformal wrapper.
- 11-A4 - tissue result: clarified continuous deconvolution proxy.
- 11-A5 - ELBO figure: relabeled as surrogate objective trajectory.
- 11-H1 - figure caption: removed template-specific color branding.
- 11-C1 - caveats: added multi-seed and bootstrap tasks.
- 12-C1 / 12-C2 / 12-C3 / 12-C4 - benchmark protocol: clarified dataset verification, metrics, split design, and atlas leakage controls.
- 12-A1 - ablations: separated direct baselines from nested HAVI-Methyl ablations.
- 12-A2 - MethylGPT: reclassified as downstream consumer.
- 12-F1 / 12-F2 - robustness outputs: added planned stratified analyses and output tasks.
- 12-A3 / App-D-A1 / App-D-C1 / App-E-C1 - compute: aligned compute language as planning estimates pending profiling.

### Theory, discussion, related work, conclusion, appendices
- 13-A2 - finite VIB bound: replaced invalid finite-penalty bound with feasible-reference inequality.
- 13-B2 - pooling: refined assumptions and delta-method interpretation.
- 13-A4 - Fano: corrected classification/MSE lower-bound direction and packing statement.
- 13-F1 / 13-C1 - theory organization: added theory-to-experiment remark and proof-status preamble.
- 14-A1 / 14-C1 / 14-G1 / 14-F1 - discussion: clarified compute estimates, diagnostics, regulatory caution, and current-status limitations.
- 15-D1 / 15-G1 / 15-C1 / 15-H1 - related work: reduced duplication, separated baselines/downstream tools, and added citation-status caution.
- 16-E1 / 16-A1 / 16-D1 - conclusion: separated demonstrated synthetic behavior from planned validation and polished final scope.
- App-A-C1 - Appendix A: distinguished variational, amortization, and implementation gaps.
- App-B-C1 / App-B-H1 - Appendix B: clarified VB-HMM as baseline, not nested full-model ablation.
- App-C-A1 / App-C-C1 - Appendix C: clarified STL/DReG implementation notes.
- App-D-A2 - Appendix D/table CSV: synchronized architecture status as planning estimates.
- App-E-A1 - Appendix E: corrected linker-bias cut-density definition.
- App-F-C1 - Appendix F: clarified data split status.
- App-G-H1 / App-I-H1 - appendices: simplified table layouts to reduce layout risk.
- Bib-H1 - bibliography: unused entries retained; no bibliography deletion applied.
- Bib-G1 / Bib-F1 - citations: no new citations added; verification tasks moved to handoff.
