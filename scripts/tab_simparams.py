"""Regenerate Appendix E Table~\\ref{tab:simparams} — simulator parameter
table with sources from the literature."""

from __future__ import annotations

import _common  # type: ignore
import havi_methyl as hm


def main() -> None:
    parser = _common.base_parser("Simulator parameter table (App. E).")
    parser.parse_args()
    sim_params = hm.SimulatorParams()

    rows = [
        {
            "parameter": "Nucleosome spacing mean",
            "value": f"{sim_params.nuc_spacing_mean:.0f} bp",
            "source": "Empirical chromatin biology",
        },
        {
            "parameter": "Nucleosome spacing std",
            "value": f"{sim_params.nuc_spacing_std:.0f} bp",
            "source": "Empirical chromatin biology",
        },
        {"parameter": "Strauss interaction gamma", "value": "0.1", "source": "Strauss 1975"},
        {
            "parameter": "Cut-site rolloff sigma_nuc",
            "value": f"{sim_params.sigma_nuc:.0f} bp",
            "source": "Inferred from cfDNA",
        },
        {
            "parameter": "Periodicity amplitude a",
            "value": f"{sim_params.periodicity_amp:.1f}",
            "source": "Snyder et al. 2016",
        },
        {
            "parameter": "Periodicity period",
            "value": f"{sim_params.periodicity_period:.1f} bp",
            "source": "Helical pitch of B-DNA",
        },
        {
            "parameter": "Length mixture mode 1",
            "value": f"N({sim_params.length_means[0]:.0f}, {sim_params.length_stds[0]:.0f}^2)",
            "source": "Snyder et al. 2016",
        },
        {
            "parameter": "Length mixture mode 2",
            "value": f"N({sim_params.length_means[1]:.0f}, {sim_params.length_stds[1]:.0f}^2)",
            "source": "Snyder et al. 2016",
        },
        {
            "parameter": "Length mixture mode 3",
            "value": f"N({sim_params.length_means[2]:.0f}, {sim_params.length_stds[2]:.0f}^2)",
            "source": "Snyder et al. 2016",
        },
        {
            "parameter": "Per-base error rate",
            "value": f"{sim_params.error_rate * 100:.1f}%",
            "source": "Illumina NovaSeq spec",
        },
        {
            "parameter": "Read length",
            "value": "150 bp paired-end",
            "source": "Illumina NovaSeq spec",
        },
        {
            "parameter": "NB dispersion r",
            "value": f"{sim_params.nb_dispersion:.0f}",
            "source": "Snyder et al. 2016",
        },
    ]
    out = _common.write_csv("outputs/tables/tab_simparams.csv", rows)
    _common.copy_to_report_tables(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
