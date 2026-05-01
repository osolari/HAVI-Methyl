"""Regenerate Appendix D Table~\\ref{tab:arch} — HAVI-Methyl architecture
parameter accounting at d_c=128, L_e=2 ISAB layers, K=6 NSF blocks, T=64.
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
            "component": f"ISAB layer 1 ({h.isab_layers} heads, m={h.inducing_points})",
            "output_dim": h.hidden_dim,
            "trainable_params": "~100K",
        },
        {
            "component": f"ISAB layer 2 ({h.isab_layers} heads, m={h.inducing_points})",
            "output_dim": h.hidden_dim,
            "trainable_params": "~100K",
        },
        {"component": "PMA pool (k=1)", "output_dim": h.hidden_dim, "trainable_params": "~50K"},
        {
            "component": "Dilated CNN (sequence encoder)",
            "output_dim": h.hidden_dim,
            "trainable_params": "~200K",
        },
        {
            "component": "HyenaDNA proj. (frozen 256 -> 128)",
            "output_dim": h.hidden_dim,
            "trainable_params": "~30K",
        },
        {
            "component": "Concat + MLP context fusion",
            "output_dim": h.hidden_dim,
            "trainable_params": "~70K",
        },
        {
            "component": f"NSF block x K={h.flow_blocks} (rational-quadratic, {h.nsf_bins} bins)",
            "output_dim": 1,
            "trainable_params": "~700K",
        },
        {
            "component": "Beta-Binomial sufficient-stat head",
            "output_dim": 1,
            "trainable_params": "~10K",
        },
        {"component": "End-motif categorical head", "output_dim": 256, "trainable_params": "~30K"},
        {"component": "Coverage NB head", "output_dim": 1, "trainable_params": "~10K"},
        {
            "component": f"Dirichlet head (T={h.t_max_hdp})",
            "output_dim": h.t_max_hdp,
            "trainable_params": "~10K",
        },
        {
            "component": "Population lambda_l (m_l, v_l)",
            "output_dim": "2L",
            "trainable_params": "60M (data-scale)",
        },
        {
            "component": "Sample shift nu_s (m_s^delta, v_s^delta)",
            "output_dim": "2S",
            "trainable_params": "~S",
        },
        {
            "component": "Total trainable encoder/heads",
            "output_dim": "—",
            "trainable_params": "~1.0M",
        },
        {
            "component": "Total variational parameters",
            "output_dim": "—",
            "trainable_params": "~60M",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_arch.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
