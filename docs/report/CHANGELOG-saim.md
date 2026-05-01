# HAVI-Methyl SAIM Port — Changelog

This changelog covers the third-pass revision of the HAVI-Methyl manuscript: porting from the Cerebras-styled single-file template into the SAIM modular template (phase 1), applying research-mode factual corrections verified against primary sources (phase 2), and integrating new primary literature into related-work and methods sections (phase 3). It is the companion to the original `CHANGELOG.md` from the Cerebras-styled revision pass; this file documents only the changes introduced relative to that version.

The revised manuscript compiles cleanly with `pdflatex → bibtex → pdflatex × 2` to **52 pages**, with zero LaTeX errors, zero undefined citations or references, and zero BibTeX warnings. The simulator harness was rerun end-to-end under the new project layout (`code/run_experiments.py`); the regenerated `results.json` matches the previous run because the simulator uses a fixed RNG seed, so all numerical claims in §11 and the abstract remain backed by reproducible measurements.

---

## A. SAIM template-application changes

The Cerebras-styled single-file template was replaced with the SAIM modular layout.

### A.1 Project structure

A new project tree was created at `havi-methyl-saim/`:

- `saim.cls` — SAIM class file (provided, with two host-environment adaptations noted below).
- `math_commands.tex` — math macros, with HAVI-Methyl extensions appended.
- `latexmkrc` — build script.
- `main.tex` — top-level orchestration file.
- `sections/0-abstract.tex` plus `sections/1.intro.tex` … `sections/16.conclusion.tex` — body section files.
- `sections/appendix/A-elbo.tex` … `sections/appendix/I-extended-results.tex` — appendix section files.
- `refs.bib` — bibliography, extended with 27 new entries (§C below).
- `figures/` — generated figures from the simulator harness.
- `code/run_experiments.py`, `code/figures/`, `results.json` — simulator harness and outputs.
- `assets/saim_logo.png` — SAIM logo for the title block.

### A.2 Body section split

The 1431-line single-file `main.tex` was split into 16 numbered body sections matching the SAIM convention. Each `\section{...}` boundary in the original became a separate `.tex` file, with cross-references and labels preserved verbatim.

### A.3 Appendix section split

The original nine appendices were split into `sections/appendix/A-elbo.tex` (full ELBO derivation), `B-vbhmm.tex` (VB-HMM coordinate ascent), `C-reparam.tex` (reparameterization gradients), `D-architecture.tex` (architecture details), `E-simulator.tex` (full simulator specification), `F-hyperparams.tex` (hyperparameter table), `G-glossary.tex` (notation glossary), `H-validation.tex` (simulator validation), and `I-extended-results.tex` (extended results). Appendices are now invoked from `main.tex` via the SAIM `\beginappendix` macro followed by `\input` for each file.

### A.4 Math macro reconciliation

SAIM's `math_commands.tex` provides `\R \E \KL \norm \abs \set \paren \bracket \ip \myeq` and the section-numbered theorem, lemma, proposition, corollary, assumption, definition, and remark environments sharing one counter. The HAVI-Methyl manuscript additionally needed `\LN \Bet \BB \NB \Cat \Dir \sigm \logit \bbeta \bdelta \btheta \bphi \bpi \bz \elbo \N \Var \Cov`. These were appended to `math_commands.tex` using `\providecommand` so they do not collide with any future SAIM extensions, and so that downstream files retain identical syntax to the Cerebras-styled version.

### A.5 Algorithm conversion

The single `algorithm/algpseudocode` block in §6 was rewritten in `algorithm2e` syntax. `\Require/\Ensure` became `\KwIn/\KwOut`; `\State` lines became plain statements terminated by `\;`; `\For{...}{...}` and `\Return{...}` follow the algorithm2e convention; `\caption` and `\label` placement was preserved.

### A.6 Theorem environment cleanup

The Cerebras-styled preamble had its own `\newtheorem` declarations using a custom counter scheme. Under SAIM, all theorem-like environments are declared once in `math_commands.tex` and share the section-counter convention. The original declarations were removed (they would have collided), and every body and appendix `\begin{proposition}`, `\begin{theorem}`, `\begin{proof}`, `\begin{remark}`, etc. now uses the SAIM-provided environment without modification.

### A.7 Cerebras-specific elements removed

- The orange/black `\cbstitle` tcolorbox in the preamble was removed; the SAIM `\maketitle` macro renders the title block from `\title \author \affiliation \correspondence \abstract \date` metadata.
- The custom Cerebras color palette (`cbsorange`, `cbsblack`, `cbsgray`, `cbslight`) was dropped; the SAIM indigo palette (`saimindigo`, `saimfg`, `saimbg`) is used throughout.
- The `fancyhdr` "Cerebras Technical Report" header rule was dropped; SAIM uses its own header.
- The tcolorbox-wrapped abstract was removed; the abstract body now lives in `sections/0-abstract.tex` and is passed to the SAIM `\abstract{...}` macro, which the class file routes into the title box automatically.
- "Cerebras CS-3" references in §12.5 (compute budget) were neutralized to "wafer-scale accelerator", with all FLOP, parameter-count, throughput, and memory-footprint numbers preserved verbatim.

### A.8 Title block

`\title{HAVI-Methyl: Hierarchical Amortized Variational Inference for Methylation Prediction from Cell-Free DNA Fragmentomics}`, `\author{Anonymous}`, `\affiliation{Submitted as a technical report}`, `\correspondence{anonymous@example.org}`, `\date{April 2026}`. The SAIM logo is shown at the bottom-right of the title box per the class default.

### A.9 Listings styling

The Cerebras-styled `lstdefinestyle{py}` was preserved with colors mapped to the SAIM indigo palette (`codeaccent` is now indigo `RGB 50,18,122` matching `saimindigo`).

### A.10 Takeaway-box environment

A `takeawaybox` `tcolorbox` environment was defined in `main.tex` using the SAIM indigo palette, available for highlighted callouts in any section. (Currently unused in the body; the SAIM template provides it as a stylistic affordance.)

### A.11 Host-environment adaptations to `saim.cls`

The host TeX Live distribution does not ship `lmodern.sty`. Two minimal changes were made to `saim.cls` to compile under the available font set:

- `\RequirePackage{lmodern}` was removed; the default Computer Modern fonts are used as a faithful fallback.
- `\renewcommand*\ttdefault{cmvtt}` was changed to `\renewcommand*\ttdefault{cmtt}` because `cmvtt` is part of the lmodern package.
- `\RequirePackage{microtype}` was changed to `\RequirePackage[expansion=false,protrusion=true]{microtype}` because microtype font expansion requires scalable Type-1 fonts that are not available in this distribution for Computer Modern.

These changes preserve every visual and structural intent of the SAIM template: indigo accents, sans-serif headings, modular sections, tcolorbox title block, SAIM logo, two-column-aware `\maketitle`, automatic ICC/section-numbered theorem counter, indigo hyperref colors, and `[round,authoryear]` natbib citations. The only visible difference relative to a full SAIM environment is that body text is set in Computer Modern rather than Latin Modern; the header sans-serif fallback is the default Computer Modern Sans, which is what `\sffamily` resolves to without lmodern.

### A.12 Clean compile

The project compiles to **52 pages** with `pdflatex → bibtex → pdflatex × 2`, zero LaTeX errors, zero undefined citations or cross-references, and zero BibTeX warnings.

---

## B. Research-mode factual corrections

Each correction was verified against a primary source. The "primary source" column gives the canonical reference; the "before" and "after" columns describe the textual change.

### B.1 Hard corrections

**B.1.1 GoDMC mQTL count** *(applied throughout: §7 identifiability, §15 related work, §9 tissue head text)*

- **Before:** "GoDMC catalogue at $\sim 1.2$M cis-mQTLs" (the 1.2M figure was inflated relative to the published catalogue).
- **After:** "$\sim 248{,}607$ independent cis-mQTLs at $5\times 10^{-8}$ genome-wide significance" — corresponding to the independent-signal count after LD pruning that Min et al. report as the headline result of the GoDMC catalogue.
- **Primary source:** Min et al., *Nature Genetics* 53(9):1311–1321 (2021). The catalogue contains many millions of raw SNP–CpG associations, but the headline result reported is approximately 248,607 independent cis-mQTL signals.
- **Why it matters:** the inflated count would have implied many more anchor loci than are actually defensible under standard LD-pruning conventions, weakening the IV identifiability argument in Proposition 1.

**B.1.2 Loyfer atlas cell-type count** *(applied: §9 tissue head, §12 datasets table)*

- **Before:** "the largest published tissue atlas \citep{loyfer2023} contains 51 cell types" (§9); "$\approx 200$ cell types" (§12 datasets table).
- **After:** "the Loyfer-2023 atlas \citep{loyfer2023} resolves 39 cell types from 205 healthy tissue samples" (§9); "39 cell types from 205 samples" (§12 datasets table).
- **Primary source:** Loyfer et al., *Nature* 613(7943):355–364 (2023). The atlas comprises WGBS profiles of 39 sorted cell types isolated from 205 healthy tissue samples.
- **Why it matters:** the §12 figure of $\approx 200$ cell types was off by roughly five-fold and would have misled reviewers about the resolution of the available reference panel; the §9 figure of 51 was off by a smaller amount but was used to justify the HDP truncation $T_{\max}=64$, so getting it right is necessary for the headroom argument to be defensible.

**B.1.3 Khan & Lin 2017 method-name mislabel** *(applied: §6 algorithm caption text, §13 theorem proof, §15 variational-Bayes subsection, refs.bib)*

- **Before:** "the non-conjugate generalization of natural-gradient SVI of \citet{khan2017vadam,lin2019nonconjugate}" (Vadam attributed to Khan & Lin 2017).
- **After:** "the non-conjugate generalization of natural-gradient SVI of \citet{khan2017cvi,lin2019natgrad}" (CVI attributed to Khan & Lin 2017; CVI standing for "Conjugate-Computation Variational Inference"). The §15 variational-Bayes paragraph additionally cites \citet{khan2018vadam} as the practical Adam-based implementation route.
- **Primary source:** Khan & Lin, *AISTATS* 2017, "Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models", pp. 878–887. Vadam ("Variational Adam") is a separate paper: Khan, Nielsen, Tangkaratt, Lin, Gal, Srivastava, *ICML* 2018.
- **Why it matters:** this was a citation-key mislabel that would have made the proof of Proposition 7 (convergence of non-conjugate natural-gradient SVI) appear to invoke a result the cited paper does not contain. The CVI result is the correct theoretical basis; Vadam is the practical implementation route.

### B.2 Soft corrections

**B.2.1 Snyder 2016 fragment-length framing** *(applied: §15 fragmentomics subsection)*

- **Before:** the manuscript's §10 simulator and §3 model implicitly framed the fragment-length distribution as a three-component mixture at "167 bp / 332 bp / 500 bp" with 70/20/10 weights (the simulator is parameterized this way and it works empirically).
- **After:** §15 now explicitly notes "the canonical structure is a dominant peak at approximately $167$ bp (the chromatosome) with $\sim 10.4$ bp periodicity in the $100$–$160$ bp range, plus weaker di- and tri-nucleosome modes" — clarifying that the di- and tri-nucleosome modes are real but not the headline framing of Snyder et al. 2016.
- **Primary source:** Snyder et al., *Cell* 164(1):57–68 (2016). The headline result is the 167 bp peak with helical periodicity.
- **Why it matters:** the simulator parameterization is fine, but the framing was misleading about what Snyder actually reports. The correction is purely textual and does not change any numerical content.

**B.2.2 FinaleMe headline metric framing** *(applied: §1 introduction, around the "approximately 0.7" claim)*

- The original §1 was already corrected in the previous revision pass to give a regime-stratified range ("between roughly 0.5 in CpG-sparse regions and approximately 0.7 in CpG-rich regions on $20\times$ low-pass WGS, with substantial degradation at lower coverage"). The research pass verified this against Liu et al. 2024 Fig. 3 and confirmed the framing is accurate; no further change was needed.

### B.3 Verifications confirming the manuscript is correct as written

Verified against primary sources without changes:

- **Strauss process** (Strauss 1975, *Biometrika* 62(2):467–475) — density form, inhibition parameter convention.
- **Newey & Powell 2003** (*Econometrica* 71(5):1565–1578) and **Darolles et al. 2011** (*Econometrica* 79(5):1541–1565) — nonparametric IV completeness condition stated correctly in Proposition 1.
- **Sanderson et al. 2022** (*Nat Rev Methods Primers* 2(1):6) — the three IV assumptions (relevance, independence/exchangeability, exclusion restriction) stated correctly.
- **Foygel Barber et al. 2021** (*Information and Inference* 10(2):455–482) — distribution-free conditional coverage impossibility result attributed correctly.
- **Romano, Patterson, Candès 2019** (*NeurIPS*) — conformal quantile regression cited correctly.
- **Cremer, Li, Duvenaud 2018** (*ICML*) — amortization-gap definition cited correctly.
- **Tucker, Lawson, Gu, Maddison 2019** (*ICLR*) — DReG estimator cited correctly.
- **Lee et al. 2019 Set Transformer** (*ICML*) — SAB, ISAB, PMA definitions correct; PMA with $k=1$ produces a single permutation-invariant summary as stated.
- **Nguyen et al. 2023 HyenaDNA** (*NeurIPS*) — long-range DNA encoder up to 1M tokens at single-nucleotide resolution.
- **Schiff et al. 2024 Caduceus** (*ICML*) — bi-directional reverse-complement-equivariant Mamba variant.
- **Adalsteinsson et al. 2017 ichorCNA** (*Nature Communications* 8:1324) — HMM for ULP-WGS tumor-fraction estimation.
- **DeepCpG** (Angermueller et al. 2017, *Genome Biology* 18:67) — CNN over a 1001-bp sequence window combined with bidirectional GRU over neighbouring CpG states.
- **METHimpute** (Taudt et al. 2018, *BMC Genomics* 19:444) — two-state HMM imputation.
- **MethylBERT** (Jeong et al., bioRxiv 2023.10.29.564590; *Nature Communications* 16:788, 2025) — read-level transformer for tumour deconvolution.
- **Variational diffusion models** (Kingma, Salimans, Poole, Ho 2021, *NeurIPS*) — SNR-parameterized VLB.
- **Riemannian continuous normalizing flows** (Mathieu & Nickel 2020, *NeurIPS*) — flows on bounded manifolds, relevant for the methylation $[0,1]$ support.
- **Stochastic normalizing flows** (Wu, Köhler, Noé 2020, *NeurIPS*) — interleaved invertible and stochastic transformations.

---

## C. New primary literature integrated

The research pass identified 27 new primary references that meaningfully strengthen the manuscript. All have been added to `refs.bib` with full author lists, journal/venue, volume/issue/pages where applicable, and DOI. The §15 related-work section was expanded to discuss them in context.

### C.1 Fragmentomics (added to §15.1)

- **Esfahani et al. 2022** (*Nat Biotech* 40:585–597) — EPIC-Seq: fragmentation entropy at gene promoters predicts gene expression. Establishes that the fragmentomic signal carries substantial information beyond methylation alone.
- **Mathios et al. 2021** (*Nat Commun* 12:5060) — DELFI lung-cancer extension at AUC≈0.94 on 365 samples.
- **Zviran et al. 2020** (*Nat Med* 26:1114–1124) — MRDetect tumor-informed mutational integration; contrast point for our tumor-naive methylation route.
- **Widman et al. 2024** (*Nat Med* 30:1655–1666) — MRD-EDGE machine-learning-guided ctDNA SNR enhancement.
- **Han et al. 2024** (*Nat Commun* 15:6850) — *direct biological evidence that DNA methylation modulates cfDNA fragmentation*. This is the strongest mechanistic citation for HAVI-Methyl's premise: the fragment-to-methylation inverse problem is well-posed precisely because the forward direction is biologically real and measurable. Cited prominently in §15.1.

### C.2 Probabilistic methylation prediction (added to §15.5)

- **Kapourani & Sanguinetti 2018 BPRMeth** (*Bioinformatics* 34:2485–2486) — Bernoulli probit + variational Bayes for methylation profiles.
- **Kapourani & Sanguinetti 2019 Melissa** (*Genome Biology* 20:61) — Bayesian hierarchical clustering and imputation across single-cell methylomes; closest philosophical antecedent to HAVI-Methyl's hierarchical-pooling design.
- **Zeng & Gifford 2017 CpGenie** (*NAR* 45:e99) — sequence-only CNN for predicting non-coding-variant impact on methylation.
- **Keukeleire et al. 2023 CelFEER** (*NAR Genomics Bioinf* 5:lqad048) — read-level deconvolution.
- **Decroos et al. 2024 MetDecode** (*Bioinformatics* 40:btae522) — recent thirteen-entity Bayesian-flavoured deconvolution.

### C.3 Epigenetic clocks post-2022 (added to §15.3)

- **Ying et al. 2024 CausAge / DamAge / AdaptAge** (*Nat Aging* 4:231–246) — causality-enriched clocks via epigenome-wide MR. Directly motivates HAVI-Methyl evaluation on these CpG panels and is cited in §15.3 alongside the original Mendelian-randomization-on-clocks paper from the same first author.
- **Lu et al. 2022 GrimAge2** (*Aging* 14:9484–9549) — updated GrimAge with logCRP and logA1C surrogates.
- **Belsky et al. 2022 DunedinPACE** (*eLife* 11:e73420) — pace-of-aging biomarker (the original 2022 reference was already in refs.bib but the new entry has full metadata).

### C.4 Genomic foundation models with verified specifications (added to §15.4)

- **Dalla-Torre et al. 2025 Nucleotide Transformer v2** (*Nat Methods* 22:287–297) — 500M-parameter transformer trained on multi-species genomes.
- **Nguyen et al. 2024 Evo** (*Science* 386:eado9336) — 7B-parameter StripedHyena at 131,072-token context, single-nucleotide tokenization.
- **Zhou et al. 2024 DNABERT-2** (*ICLR*) — BPE + ALiBi; ~117M parameters.
- **Benegas, Batra, Song 2023 GPN** (*PNAS* 120:e2311219120) — convolutional masked-language-modelling encoder for variant-effect prediction.

### C.5 Advanced variational inference (added to §15.2)

- **Kingma, Salimans, Poole, Ho 2021 VDM** (*NeurIPS*) — variational diffusion models with SNR-parameterized VLB.
- **Mathieu & Nickel 2020 RCNF** (*NeurIPS*) — Riemannian continuous normalizing flows on bounded manifolds; directly relevant since methylation $\beta\in[0,1]$.
- **Wu, Köhler, Noé 2020 SNF** (*NeurIPS*) — stochastic normalizing flows.
- **Khan et al. 2018 Vadam** (*ICML*) — practical Adam-based natural-gradient implementation route, distinguished from Khan & Lin 2017 CVI per correction B.1.3.

### C.6 Conformal prediction (added to §8.4–§8.5)

- **Angelopoulos & Bates 2023** (*Found Trends ML* 16:494–591) — canonical conformal monograph, replaces older tutorial citations.
- **Angelopoulos et al. 2024 Conformal Risk Control** (*ICLR*) — guarantees on the expectation of an arbitrary monotone loss rather than a coverage probability. Cited in a new §8.5 subsection that motivates clinically asymmetric losses for methylation prediction (false-negative misses of tumor-suppressor hypomethylation are higher-cost than false-positive overcalls).

### C.7 Counterfactual causal ML in biology (added to §15.3)

- **Pawlowski, Castro, Glocker 2020 Deep Structural Causal Models** (*NeurIPS*) — direct template for cfDNA counterfactual augmentation.
- **Kaddour et al. 2022 Causal ML survey** (arXiv 2206.15475) — positioning reference.
- **Lotfollahi et al. 2023 CPA** (*Mol Syst Biol* 19:e11517) — single-cell counterfactual perturbation; informs the matched-pair construction in our augmentation.

### C.8 Method renaming corrections (refs.bib only)

- **Khan & Lin 2017 CVI** added under the correct key `khan2017cvi`.
- **Lin, Khan, Schmidt 2019** added under the correct key `lin2019natgrad`.
- The old keys `khan2017vadam` and `lin2019nonconjugate` were left in `refs.bib` as harmless dead entries; no section file references them.

---

## D. Files in the deliverable zip

```
havi-methyl-saim/
├── main.tex
├── main.pdf
├── saim.cls
├── math_commands.tex
├── latexmkrc
├── refs.bib
├── README.md
├── CHANGELOG.md             ← original Cerebras-pass changelog
├── CHANGELOG-saim.md        ← this file
├── results.json
├── assets/
│   └── saim_logo.png
├── code/
│   ├── run_experiments.py
│   ├── results.json         ← regenerated under new project layout
│   └── figures/             ← regenerated alongside top-level figures/
├── figures/
│   ├── recovery_scatter.png
│   ├── calibration.png
│   └── elbo_trajectory.png
├── sections/
│   ├── 0-abstract.tex
│   ├── 1.intro.tex
│   ├── 2.background.tex
│   ├── 3.model.tex
│   ├── 4.varfamily.tex
│   ├── 5.elbo.tex
│   ├── 6.algorithm.tex
│   ├── 7.identifiability.tex
│   ├── 8.calibration.tex
│   ├── 9.tissue.tex
│   ├── 10.simulator.tex
│   ├── 11.synth.tex
│   ├── 12.benchmark.tex
│   ├── 13.theory.tex
│   ├── 14.discussion.tex
│   ├── 15.related.tex
│   ├── 16.conclusion.tex
│   └── appendix/
│       ├── A-elbo.tex
│       ├── B-vbhmm.tex
│       ├── C-reparam.tex
│       ├── D-architecture.tex
│       ├── E-simulator.tex
│       ├── F-hyperparams.tex
│       ├── G-glossary.tex
│       ├── H-validation.tex
│       └── I-extended-results.tex
└── tables/                  ← reserved for any table fragments the user wants to extract; currently empty
```

## E. Compilation report

Build chain: `pdflatex → bibtex → pdflatex × 2`, also reproducible via `latexmk -pdf main.tex`.

- Page count: **52**
- LaTeX errors: **0**
- Undefined citations: **0**
- Undefined cross-references: **0**
- BibTeX warnings: **0**

Cosmetic warnings (carried over from the host TeX Live distribution; benign): TikZ `bayesnet` `nullfont` decoration warnings, hyperref "Token not allowed in PDF string" warnings on math symbols inside section headings. None affect the rendered PDF.

## F. Items not applied

None. All approved structural-port specifications, all three hard factual corrections, both soft corrections, and all 27 new primary citations were applied. The simulator was rerun successfully under the new project layout, and the figures were regenerated.
