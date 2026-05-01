"""Regenerate Appendix~F hyperparameter table from ``havi_methyl.constants``.

This is the canonical source: the LaTeX table in App. F documents these
defaults; the CSV here re-extracts them so the LaTeX and library cannot
disagree.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm


def main() -> None:
    parser = _common.base_parser("Hyperparameter table (App. F).")
    parser.parse_args()
    h = hm.DEFAULT_HPARAMS

    rows = [
        # Model priors
        {
            "section": "Model priors",
            "symbol": "mu_0",
            "value": h.mu_0,
            "description": "Population logit prior mean",
        },
        {
            "section": "Model priors",
            "symbol": "tau_0",
            "value": h.tau_0,
            "description": "Population logit prior std",
        },
        {
            "section": "Model priors",
            "symbol": "sigma_delta",
            "value": h.sigma_delta,
            "description": "Sample-shift std",
        },
        {
            "section": "Model priors",
            "symbol": "sigma_eta",
            "value": h.sigma_eta,
            "description": "Per-(s,l) latent std",
        },
        {
            "section": "Model priors",
            "symbol": "kappa",
            "value": h.kappa,
            "description": "Beta-Binomial concentration",
        },
        # Regularization
        {
            "section": "Regularization",
            "symbol": "beta_VIB",
            "value": h.beta_vib,
            "description": "VIB weight",
        },
        {
            "section": "Regularization",
            "symbol": "lambda_cf",
            "value": h.lambda_cf,
            "description": "Counterfactual invariance",
        },
        {
            "section": "Regularization",
            "symbol": "lambda_mQTL",
            "value": h.lambda_mqtl,
            "description": "mQTL anchor weight",
        },
        {
            "section": "Regularization",
            "symbol": "lambda_ToO",
            "value": h.lambda_too,
            "description": "Tissue-of-origin weight",
        },
        # Architecture
        {
            "section": "Architecture",
            "symbol": "d_c",
            "value": h.hidden_dim,
            "description": "Encoder hidden dim",
        },
        {
            "section": "Architecture",
            "symbol": "L_e",
            "value": h.isab_layers,
            "description": "ISAB layers",
        },
        {
            "section": "Architecture",
            "symbol": "m",
            "value": h.inducing_points,
            "description": "ISAB inducing points",
        },
        {
            "section": "Architecture",
            "symbol": "K (flow)",
            "value": h.flow_blocks,
            "description": "NSF stack depth",
        },
        {
            "section": "Architecture",
            "symbol": "NSF bins",
            "value": h.nsf_bins,
            "description": "Spline knots per block",
        },
        {
            "section": "Architecture",
            "symbol": "T_max",
            "value": h.t_max_hdp,
            "description": "HDP truncation level",
        },
        # Optimization
        {
            "section": "Optimization",
            "symbol": "LR encoder",
            "value": h.lr_encoder,
            "description": "AdamW",
        },
        {
            "section": "Optimization",
            "symbol": "weight decay",
            "value": h.weight_decay,
            "description": "AdamW",
        },
        {
            "section": "Optimization",
            "symbol": "rho_t exponent",
            "value": h.rho_exponent,
            "description": "Robbins-Monro",
        },
        {
            "section": "Optimization",
            "symbol": "|B_s|",
            "value": h.batch_samples,
            "description": "Sample mini-batch size",
        },
        {
            "section": "Optimization",
            "symbol": "|B_l|",
            "value": h.batch_loci,
            "description": "Locus mini-batch size",
        },
        {
            "section": "Optimization",
            "symbol": "g_clip",
            "value": h.grad_clip,
            "description": "Gradient clip norm",
        },
        # IWAE
        {
            "section": "IWAE",
            "symbol": "K_IWAE_train",
            "value": h.k_iwae_train,
            "description": "Single-sample reparam",
        },
        {
            "section": "IWAE",
            "symbol": "K_IWAE_finetune",
            "value": h.k_iwae_finetune,
            "description": "DReG-IWAE",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_hparams.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
