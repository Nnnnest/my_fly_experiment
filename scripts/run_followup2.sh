#!/usr/bin/env bash
# usage: bash scripts/run_followup2.sh N_WORKERS
# Before the first call: run scripts/check_followup2.sh, run the N0 tuning (README) and set N0_DECAY, commit everything.
set -e
N=${1:-4}
git diff --quiet HEAD || { echo "tracked files modified: commit first"; exit 1; }
git -C .. diff --quiet HEAD || { echo "fly-brain (simulator) has modified tracked files"; exit 1; }
git ls-files --error-unmatch scripts/kc_diag/run_kc_diag.py scripts/readouts/run_readouts_constrained.py scripts/comparison/run_comparison_capacity.py agents/shared/kc_native.py > /dev/null || { echo "new scripts not committed"; exit 1; }
python -m scripts.kc_diag.run_kc_diag --workers $N
python -m scripts.comparison.run_comparison_capacity --workers $N
python -m scripts.readouts.run_readouts_constrained --workers $N
echo "FOLLOWUP2 BATCHES FINISHED"
