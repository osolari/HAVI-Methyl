"""Regenerate Appendix~F hyperparameter table.

Schema matches ``docs/report/tables/tab_hparams.csv``: each row separates the
full-model planning value from the simplified-harness value used in Sec. 11
so readers can tell which numbers are demonstrated and which are planning
defaults. The ``simplified_harness_value`` column is the value actually used
by the released code in ``src/havi_methyl/`` (or ``not implemented`` /
``synthetic default`` where applicable).
"""

from __future__ import annotations

import _common  # type: ignore

import havi_methyl as hm


def main() -> None:
    parser = _common.base_parser("Hyperparameter table (App. F).")
    parser.parse_args()
    h = hm.DEFAULT_HPARAMS

    rows = [
        {
            "section": "Model priors",
            "symbol": "mu_0",
            "full_model_planning_value": h.mu_0,
            "simplified_harness_value": "synthetic default",
            "description": "Population logit prior mean",
        },
        {
            "section": "Model priors",
            "symbol": "tau_0",
            "full_model_planning_value": h.tau_0,
            "simplified_harness_value": "synthetic default",
            "description": "Population logit prior std",
        },
        {
            "section": "Model priors",
            "symbol": "sigma_delta",
            "full_model_planning_value": h.sigma_delta,
            "simplified_harness_value": "code-specific synthetic shift",
            "description": "Sample shift std",
        },
        {
            "section": "Model priors",
            "symbol": "sigma_eta",
            "full_model_planning_value": h.sigma_eta,
            "simplified_harness_value": "empirical-Bayes proxy",
            "description": "Local latent std",
        },
        {
            "section": "Model priors",
            "symbol": "kappa",
            "full_model_planning_value": h.kappa,
            "simplified_harness_value": "not full Beta-Binomial",
            "description": "Concentration",
        },
        {
            "section": "Regularization",
            "symbol": "beta_VIB",
            "full_model_planning_value": "tune on validation",
            "simplified_harness_value": "0.3 in stress test",
            "description": "VIB weight",
        },
        {
            "section": "Regularization",
            "symbol": "lambda_cf",
            "full_model_planning_value": "tune on validation",
            "simplified_harness_value": "not implemented",
            "description": "Counterfactual penalty",
        },
        {
            "section": "Regularization",
            "symbol": "lambda_mQTL",
            "full_model_planning_value": "tune on validation",
            "simplified_harness_value": "synthetic proxy",
            "description": "mQTL anchor weight",
        },
        {
            "section": "Regularization",
            "symbol": "lambda_ToO",
            "full_model_planning_value": "tune on validation",
            "simplified_harness_value": "proxy deconvolution",
            "description": "Tissue loss weight",
        },
        {
            "section": "Architecture",
            "symbol": "d_c",
            "full_model_planning_value": h.hidden_dim,
            "simplified_harness_value": "not used directly",
            "description": "Encoder hidden dim",
        },
        {
            "section": "Architecture",
            "symbol": "L_e",
            "full_model_planning_value": h.isab_layers,
            "simplified_harness_value": "not implemented",
            "description": "ISAB layers",
        },
        {
            "section": "Architecture",
            "symbol": "m",
            "full_model_planning_value": h.inducing_points,
            "simplified_harness_value": "not implemented",
            "description": "ISAB inducing points",
        },
        {
            "section": "Architecture",
            "symbol": "K flow",
            "full_model_planning_value": h.flow_blocks,
            "simplified_harness_value": "Gaussian posterior",
            "description": "NSF depth",
        },
        {
            "section": "Optimization",
            "symbol": "LR encoder",
            "full_model_planning_value": h.lr_encoder,
            "simplified_harness_value": "not applicable",
            "description": "AdamW",
        },
        {
            "section": "Optimization",
            "symbol": "|B_s|",
            "full_model_planning_value": h.batch_samples,
            "simplified_harness_value": "full synthetic set",
            "description": "Sample mini-batch",
        },
        {
            "section": "Optimization",
            "symbol": "|B_l|",
            "full_model_planning_value": h.batch_loci,
            "simplified_harness_value": "L=300",
            "description": "Locus mini-batch",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_hparams.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
