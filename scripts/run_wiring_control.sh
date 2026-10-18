#!/usr/bin/env bash
# usage: bash scripts/run_wiring_control.sh N_WORKERS
set -e
N=${1:-4}
git diff --quiet HEAD || { echo "tracked files modified: commit first"; exit 1; }
git -C .. diff --quiet HEAD || { echo "fly-brain (simulator) has modified tracked files"; exit 1; }
for S in 0 1 2; do
  python -m scripts.controls.make_shuffled_circuit --seed $S
  export FLY_CIRCUIT_DIR="$(pwd)/assets/circuits/shuffled_s$S"
  export FLY_RUNS_DIR="$(pwd)/runs_shuffled_s$S"
  python -m scripts.bandit.run_bandit --workers $N
  python -m scripts.comparison.run_comparison --workers $N
  python -m scripts.addition.run_addition --workers $N
  python -m scripts.addition.run_addition_diagnostic --workers $N
done
echo "WIRING CONTROL FINISHED"
