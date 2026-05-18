# Manuscript

<div class="saim-cite" markdown>
> **Solari, O. S.** (2026). *HAVI-Methyl: Hierarchical Amortized
> Variational Inference for Methylation Prediction from cfDNA
> Fragmentomics.* sAIm Labs.
> [PDF](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/main.pdf){target=_blank} ·
> [GitHub](https://github.com/osolari/HAVI-Methyl){target=_blank}
</div>

The technical report (PDF) and source LaTeX are bundled in
`docs/report/`.

- [`docs/report/main.pdf`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/main.pdf) — the rendered manuscript.
- [`docs/report/sections/`](https://github.com/osolari/HAVI-Methyl/tree/main/docs/report/sections) — individual `.tex` section sources.
- [`docs/report/CHANGELOG.md`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/CHANGELOG.md) — manuscript change log (see also the repo-side [Changelog](changelog.md)).
- [`docs/report/CODING_AGENT_HANDOFF.md`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/CODING_AGENT_HANDOFF.md) — author-verification checklist mirrored in the repo.

## Section sources

| Section | Topic | Source |
|---|---|---|
| 0 | Abstract | [`0-abstract.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/0-abstract.tex) |
| 1 | Introduction | [`1.intro.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/1.intro.tex) |
| 2 | Background and preliminaries | [`2.background.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/2.background.tex) |
| 3 | Probabilistic model | [`3.model.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/3.model.tex) |
| 4 | Variational family | [`4.varfamily.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/4.varfamily.tex) |
| 5 | ELBO derivation | [`5.elbo.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/5.elbo.tex) |
| 6 | Inference algorithm | [`6.algorithm.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/6.algorithm.tex) |
| 7 | Identifiability | [`7.identifiability.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/7.identifiability.tex) |
| 8 | Calibration | [`8.calibration.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/8.calibration.tex) |
| 9 | Tissue-of-origin head | [`9.tissue.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/9.tissue.tex) |
| 10 | Simulator | [`10.simulator.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/10.simulator.tex) |
| 11 | Synthetic experiments | [`11.synth.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/11.synth.tex) |
| 12 | Real-data benchmark | [`12.benchmark.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/12.benchmark.tex) |
| 13 | Theory | [`13.theory.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/13.theory.tex) |
| 14 | Discussion | [`14.discussion.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/14.discussion.tex) |
| 15 | Related work | [`15.related.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/15.related.tex) |
| 16 | Conclusion | [`16.conclusion.tex`](https://github.com/osolari/HAVI-Methyl/blob/main/docs/report/sections/16.conclusion.tex) |
| Appendix | Proofs, simulator validation, hyperparameter tables, notation glossary | [`appendix/`](https://github.com/osolari/HAVI-Methyl/tree/main/docs/report/sections/appendix) |

## Code-to-manuscript cross-reference

Every public surface in `havi_methyl` cites the manuscript label it
implements. The [API reference](api.md) gives the canonical map.
Highlights:

- `havi_methyl.fit_svi_torch` matches the §6 Algorithm 1 SVI loop with
  Set Transformer + posterior head + Beta-Binomial reconstruction +
  Robbins-Monro recentering.
- `havi_methyl.TorchSVIConfig.iwae_dreg` matches §5.3 (Tucker-2019
  doubly-reparameterised gradient estimator).
- `havi_methyl.TorchSVIConfig.adversarial_weight` matches §7
  (gradient-reversal head with sample-id discriminator).
- `havi_methyl.calibration.gaussian_conformal_intervals` matches §8.2
  Proposition 8.1 (split-conformal marginal coverage).
- `havi_methyl.tissue.dirichlet_head_predict` matches §9
  (variance-weighted Dirichlet head).
- `havi_methyl.bounds.hierarchical_pooling_variance` matches §13.1
  (hierarchical pooling variance reduction).
- `havi_methyl.bounds.fano_error_lower_bound` matches §13.3
  (information-theoretic recoverability limit).

## Reproducing manuscript figures and tables

```bash
bash scripts/run_all.sh
make report
```

The first command regenerates every CSV under `outputs/tables/` and
every PNG/PDF under `outputs/figures/`, mirroring into
`docs/report/tables/` and `docs/report/figures/` byte-for-byte. The
second recompiles `docs/report/main.pdf` (requires pdfLaTeX). See
[Reproducibility](reproducibility.md) for the full pipeline.
