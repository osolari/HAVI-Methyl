"""Regenerate Appendix~D Table~\\ref{tab:arch} — HAVI-Methyl architecture
parameter accounting at d_c=128, L_e=2 ISAB layers, K=6 NSF blocks, T=64.

The schema matches the report's ``docs/report/tables/tab_arch.csv``: each row
documents a planned full-model component, its specification, output dim,
trainable-parameter estimate, and implementation status. Components flagged
``planned full model`` are listed in IMPL-02..05 of CODING_AGENT_HANDOFF.md.
"""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm


def main() -> None:
    parser = _common.base_parser("Architecture parameter table (App. D).")
    parser.parse_args()

    h = hm.DEFAULT_HPARAMS
    rows = [
        {
            "component": "ISAB layer 1",
            "specification": f"4 heads, m={h.inducing_points}, hidden {h.hidden_dim}",
            "output_dim": h.hidden_dim,
            "trainable_params": "~100K",
            "status": "planned full model",
        },
        {
            "component": "ISAB layer 2",
            "specification": f"4 heads, m={h.inducing_points}, hidden {h.hidden_dim}",
            "output_dim": h.hidden_dim,
            "trainable_params": "~100K",
            "status": "planned full model",
        },
        {
            "component": "PMA pool",
            "specification": "4 heads, k=1",
            "output_dim": h.hidden_dim,
            "trainable_params": "~50K",
            "status": "planned full model",
        },
        {
            "component": "Dilated CNN sequence encoder",
            "specification": "6 layers, receptive field 2 kb",
            "output_dim": h.hidden_dim,
            "trainable_params": "~200K",
            "status": "optional planned",
        },
        {
            "component": "HyenaDNA projection",
            "specification": "frozen 256 -> 128",
            "output_dim": h.hidden_dim,
            "trainable_params": "~30K",
            "status": "optional planned",
        },
        {
            "component": "Context fusion MLP",
            "specification": "fragment + sequence + variational means + coverage",
            "output_dim": h.hidden_dim,
            "trainable_params": "~70K",
            "status": "planned full model",
        },
        {
            "component": f"NSF block x K={h.flow_blocks}",
            "specification": f"rational-quadratic, {h.nsf_bins} bins",
            "output_dim": 1,
            "trainable_params": "~700K",
            "status": "planned full model",
        },
        {
            "component": "Beta-Binomial head",
            "specification": "sufficient-statistic head",
            "output_dim": 1,
            "trainable_params": "~10K",
            "status": "planned full model",
        },
        {
            "component": "End-motif head",
            "specification": "256-way categorical",
            "output_dim": 256,
            "trainable_params": "~30K",
            "status": "planned full model",
        },
        {
            "component": "Coverage NB head",
            "specification": "mean-dispersion parameterization",
            "output_dim": 1,
            "trainable_params": "~10K",
            "status": "planned full model",
        },
        {
            "component": "Dirichlet ToO head",
            "specification": f"T<={h.t_max_hdp} softplus concentration",
            "output_dim": h.t_max_hdp,
            "trainable_params": "~10K",
            "status": "planned full model",
        },
        {
            "component": "Population lambda_l",
            "specification": "m_l, v_l",
            "output_dim": "2L",
            "trainable_params": "data-scale",
            "status": "variational",
        },
        {
            "component": "Sample shift nu_s",
            "specification": "m_s^delta, v_s^delta",
            "output_dim": "2S",
            "trainable_params": "sample-scale",
            "status": "variational",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_arch.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
