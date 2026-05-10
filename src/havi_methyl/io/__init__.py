"""Real-data loaders for the HAVI-Methyl benchmarks (Phase 5 / IMPL-06 dataset gate).

Three datasets are supported, matching the access the lab already has on disk:

  - Loyfer 2023 atlas (`load_loyfer_atlas_matrix`,
    `load_loyfer_pat_directory`).
  - Liu 2024 FinaleMe paired WGS + WGBS (`load_finaleme_dataset`).
  - Roadmap Bisulfite-Seq WGBS tracks (`load_roadmap_wgbs_atlas`).

Every loader returns objects that match the existing pipeline shape so
``evaluate_real_data_benchmark``, ``deconvolve_least_squares``, etc. work
unchanged. Each loader writes a small provenance entry documenting the
source path, sample count, and any filtering applied.
"""

from __future__ import annotations

from havi_methyl.io.finaleme import load_finaleme_dataset
from havi_methyl.io.loyfer import (
    load_loyfer_atlas_matrix,
    load_loyfer_pat_directory,
)
from havi_methyl.io.roadmap import load_roadmap_wgbs_atlas

__all__ = [
    "load_finaleme_dataset",
    "load_loyfer_atlas_matrix",
    "load_loyfer_pat_directory",
    "load_roadmap_wgbs_atlas",
]
