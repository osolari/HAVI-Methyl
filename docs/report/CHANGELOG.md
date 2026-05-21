# CHANGELOG

## 2026-05-17 — Phase Five Real-Data Push

Closed four of five open follow-ups from §14 Discussion:

1. **Length-mixture re-fit on real Liu 2024 fragments.** EM-fitted a 3-mode Gaussian mixture to 5M cfDNA fragments from 8 Liu 2024 patients (chr 1 + 19–22). New `LENGTH_MIXTURE_*` constants in `src/havi_methyl/constants.py` are π=[0.874, 0.117, 0.009], μ=[161, 313, 455] bp, σ=[21, 38, 27] bp. The previously-cited 0.005 target for the 320–350 bp peak was incorrect; real Liu 2024 cfDNA shows 0.001 per bp. All five App H validation axes flip to *verified*.
2. **True gradient-reversal adversarial head.** Custom `torch.autograd.Function` with identity forward / sign-flipped scaled-gradient backward, plus a 2-layer MLP discriminator that classifies encoder context into sample id. The discriminator parameters train normally via AdamW; the encoder receives the negated gradient and is pushed toward a sample-invariant context. Replaces the previous context-variance proxy.
3. **Joint tissue-head training inside `fit_svi_torch`.** Optional `tissue_reference`/`tissue_target`/`tissue_weight` kwargs. Each mini-batch solves the variance-weighted deconvolution on the locus subset via differentiable `torch.linalg.lstsq` and adds `(pi_pred − pi_true).pow(2).mean()` to the loss. Post-hoc Sec. 9 head recovered at `tissue_weight=0`.
4. **Proper Tucker-2019 DReG estimator.** When `iwae_dreg=True` and K>1, the log-density `log q_phi(eta_k | x)` is computed with `(mu_q, log_sigma)` detached, leaving only the pathwise term in the encoder gradient. Smoke test at K=8 shows ~10% ELBO-trajectory variance reduction.

**Headline real-data result.** The BB-trials bug fix (using WGBS read coverage `ds.n_total` rather than WGS fragment count `ds.n` as the Beta-Binomial trials) lifts HAVI-Methyl (full torch) on the Liu 2024 paired panel from r = −0.07 to **r = 0.467 vs FinaleMe-style HMM r = 0.078** (∼6.0× lift, 500-iter A10G run; the 200-iter intermediate landed r = 0.455), AUC 0.750 vs 0.564, ECE 0.311 vs 0.474, ICC(2,1) 0.436 vs 0.052. Per-stratum: HAVI is the only method with positive Pearson r in every WGBS-depth stratum (FinaleMe is anti-correlated at r ≈ −0.05 in the multi-read interior; HAVI reaches r ≈ +0.28 there and r ≈ +0.64 in the β-extreme stratum). On the Loyfer/UXM\_deconv U25 panel the variance-weighted Dirichlet head wins LOO RMSE at every one of the 36 cell types (mean RMSE 0.017 vs lstsq 0.028 vs FinaleMe-binarized 0.038 vs HDP-truncated 0.035; median advantage over lstsq +0.011). Multi-seed (N=20) synthetic recovery shows non-overlapping 90% bootstrap intervals at 0.1×, 1×, 5× (medians 0.357 vs 0.191, 0.790 vs 0.520, 0.901 vs 0.826). All five App. H simulator-validation axes are verified after the 3-mode length-mixture re-fit on real Liu 2024 fragments. VIB + mQTL leakage stress drops prior attribution from 5.93% to 0.62% to 0.037%.

**Figure overhaul.** Nine real-data + synthetic figures: `finaleme_paired_metrics`, `finaleme_paired_scatter` (5-panel hexbin), `finaleme_coverage_strat`, `loyfer_loo_rmse`, `loyfer_loo_per_tissue`, `multiseed_recovery`, `recovery_scatter` (2×4 hexbin), `calibration` (reliability + miscal inset), `elbo_trajectory` (real torch training curve). All wired into §11/§12/§13 of the manuscript.

**Manuscript prose.** All "planned" hedges replaced with measured numbers across §1, §2, §6, §7, §8, §9, §11, §12, §13, §14, §16, App E, App F, App H, App I. Static cross-ref scan: 0 broken `\ref`/`\cite`/`\includegraphics`. The only open research direction in §14 is prospective clinical validation.

---

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
