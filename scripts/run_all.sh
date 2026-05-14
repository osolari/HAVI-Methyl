#!/usr/bin/env bash
# Run every figure / table / benchmark script in order.
#
# Order matters: scripts/bench_synth_recovery.py is the canonical pipeline
# run that writes outputs/results.json + outputs/plot_data.npz; the figure
# and several table scripts read from these artifacts (IMPL-10 in
# docs/report/CODING_AGENT_HANDOFF.md). Running it first guarantees that
# every downstream artifact is internally consistent.
#
# Flags:
#   --fast        Use small S/L/iter for quick iteration (suffix --fast on
#                 every child script). Default uses paper-scale config.
#   --figures     Only run figures.
#   --tables      Only run tables.
#   --benchmarks  Only run benchmarks.
#   --all         Run figures + tables + benchmarks (default).
set -euo pipefail

cd "$(dirname "$0")/.."

FAST=""
RUN_FIG=1
RUN_TAB=1
RUN_BENCH=1
SELECT=""

for arg in "$@"; do
  case "$arg" in
    --fast) FAST="--fast" ;;
    --figures) SELECT="figures" ;;
    --tables) SELECT="tables" ;;
    --benchmarks) SELECT="benchmarks" ;;
    --all) SELECT="all" ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if [[ -n "$SELECT" ]]; then
  RUN_FIG=0; RUN_TAB=0; RUN_BENCH=0
  case "$SELECT" in
    figures)   RUN_FIG=1; RUN_BENCH=1 ;;   # figures need plot_data.npz
    tables)    RUN_TAB=1; RUN_BENCH=1 ;;   # several tables read results.json
    benchmarks) RUN_BENCH=1 ;;
    all)       RUN_FIG=1; RUN_TAB=1; RUN_BENCH=1 ;;
  esac
fi

PYTHON=${PYTHON:-python3}
mkdir -p outputs/figures outputs/tables docs/report/tables docs/report/figures

run_step() {
  echo
  echo "=== $1 ==="
  $PYTHON "$@" $FAST
}

# Auto-detect real-data assets for Phase 5 benches. If the committed panels +
# manifests are present (and, for the FinaleMe paired bench, the lab drive at
# /Volumes/Omid Solari is mounted), the real-data path runs. Otherwise the
# benches fall back to their synthetic-proxy defaults so a fresh checkout on a
# machine without the drive still reproduces a self-contained pipeline.
LOYFER_PANEL_HG38="data/loyfer_panel/Atlas.U25.l4.hg38.tsv"
LOYFER_PANEL_HG19="data/loyfer_panel/Atlas.U25.l4.hg19.tsv"
FINALEME_DRIVE="/Volumes/Omid Solari/finaleme"
FINALEME_MANIFEST="data/finaleme_manifest/sample_pairs.csv"
FINALEME_HV_PANEL="data/finaleme_manifest/high_variance_cpgs.hg19.bed"
BUFFY_COAT_BW="$FINALEME_DRIVE/wgbs_buffyCoat_jensen2015GB.methy.hg19.bw"

TISSUE_LOO_ARGS=()
if [[ -f "$LOYFER_PANEL_HG38" ]]; then
  TISSUE_LOO_ARGS+=(--atlas-tsv "$LOYFER_PANEL_HG38" --n-tissues 36 --loci 900 --samples 50)
fi

FINALEME_ARGS=()
if [[ -d "$FINALEME_DRIVE" && -f "$FINALEME_MANIFEST" && -f "$FINALEME_HV_PANEL" && -f "$BUFFY_COAT_BW" ]]; then
  FINALEME_ARGS+=(--data-dir "$FINALEME_DRIVE" --manifest "$FINALEME_MANIFEST"
                  --locus-panel "$FINALEME_HV_PANEL" --buffy-coat-bw "$BUFFY_COAT_BW")
fi

run_step_args() {
  local script=$1; shift
  echo
  echo "=== $script ==="
  $PYTHON "$script" "$@" $FAST
}

# Always run the canonical pipeline first if we need anything that depends on it.
if [[ $RUN_BENCH -eq 1 || $RUN_FIG -eq 1 || $RUN_TAB -eq 1 ]]; then
  echo "### Canonical pipeline (Sec. 11) ###"
  run_step scripts/bench_synth_recovery.py
fi

if [[ $RUN_BENCH -eq 1 ]]; then
  echo
  echo "### Other benchmarks (Sec. 12, App. H) ###"
  run_step_args scripts/bench_finaleme_realdata.py "${FINALEME_ARGS[@]+"${FINALEME_ARGS[@]}"}"
  run_step scripts/bench_simulator_validation.py
  run_step scripts/bench_compute_budget.py
  # bench_torch_svi.py requires the optional `torch` extras (`make install-torch`).
  # Skip cleanly when torch is not importable so a default install still produces
  # the rest of the canonical pipeline byte-for-byte.
  if $PYTHON -c "import torch" >/dev/null 2>&1; then
    run_step scripts/bench_torch_svi.py
  else
    echo
    echo "=== scripts/bench_torch_svi.py === (skipped: torch not installed)"
  fi
  # bench_ablation_matrix.py also depends on torch via TorchSVIConfig.
  if $PYTHON -c "import torch" >/dev/null 2>&1; then
    run_step scripts/bench_ablation_matrix.py
  else
    echo
    echo "=== scripts/bench_ablation_matrix.py === (skipped: torch not installed)"
  fi
  run_step_args scripts/bench_tissue_loo.py "${TISSUE_LOO_ARGS[@]+"${TISSUE_LOO_ARGS[@]}"}"
  run_step scripts/bench_multiseed_recovery.py
fi

if [[ $RUN_TAB -eq 1 ]]; then
  echo
  echo "### Tables (Sec. 4.6, 5.6, 9, 11, 12, App. D-F) ###"
  run_step scripts/tab_vfamily.py
  run_step scripts/tab_rescale.py
  run_step scripts/tab_datasets.py
  run_step scripts/tab_ablations.py
  run_step scripts/tab_arch.py
  run_step scripts/tab_simparams.py
  run_step scripts/tab_hparams.py
  run_step scripts/tab_recovery.py
  run_step scripts/tab_extended_recovery.py
  run_step scripts/tab_identifiability.py
  run_step scripts/tab_tissue.py
fi

if [[ $RUN_FIG -eq 1 ]]; then
  echo
  echo "### Figures (Sec. 11) ###"
  run_step scripts/fig_recovery_scatter.py
  run_step scripts/fig_calibration.py
  run_step scripts/fig_elbo_trajectory.py
fi

echo
echo "All done. Outputs in outputs/, copies mirrored to docs/report/{figures,tables}."
