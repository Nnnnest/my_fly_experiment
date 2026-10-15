#!/usr/bin/env bash
# usage: bash scripts/run_final.sh N_WORKERS
set -e
N=${1:-4}
git diff --quiet HEAD || { echo "tracked files modified: commit first"; exit 1; }
git -C .. diff --quiet HEAD || { echo "fly-brain (simulator) has modified tracked files"; exit 1; }
python -m scripts.bandit.run_bandit --workers $N
python -m scripts.comparison.run_comparison --workers $N
python -m scripts.addition.run_addition --workers $N
python -m scripts.addition.run_addition_diagnostic --workers $N
python -m scripts.tictactoe.run_tictactoe --workers $N
python -m scripts.gridworld.run_gridworld --workers $N
echo "FINAL BATCHES FINISHED"
