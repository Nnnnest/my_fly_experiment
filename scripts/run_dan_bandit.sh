#!/usr/bin/env bash
# usage: bash scripts/run_dan_bandit.sh N_WORKERS
set -e
N=${1:-4}
git diff --quiet HEAD || { echo "tracked files modified: commit first"; exit 1; }
git -C .. diff --quiet HEAD || { echo "fly-brain (simulator) has modified tracked files"; exit 1; }
python -m scripts.addition.run_addition_dan_kc --workers $N
python -m scripts.bandit.run_bandit_punish --workers $N
echo "DAN-KC AND BANDIT-PUNISH BATCHES FINISHED"
