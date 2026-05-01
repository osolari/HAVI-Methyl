#!/usr/bin/env bash
# Run every figure / table / benchmark script in order.
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
    figures)   RUN_FIG=1 ;;
    tables)    RUN_TAB=1 ;;
    benchmarks) RUN_BENCH=1 ;;
    all)       RUN_FIG=1; RUN_TAB=1; RUN_BENCH=1 ;;
  esac
fi

PYTHON=${PYTHON:-python3}
mkdir -p outputs/figures outputs/tables docs/report/tables

run_step() {
  echo
  echo "=== $1 ==="
  $PYTHON "$@" $FAST
}

if [[ $RUN_TAB -eq 1 ]]; then
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

if [[ $RUN_BENCH -eq 1 ]]; then
  echo
  echo "### Benchmarks (Sec. 11, 12, App. H) ###"
  run_step scripts/bench_synth_recovery.py
  run_step scripts/bench_finaleme_realdata.py
  run_step scripts/bench_simulator_validation.py
  run_step scripts/bench_compute_budget.py
fi

echo
echo "All done. Outputs in outputs/, copies mirrored to docs/report/{figures,tables}."
