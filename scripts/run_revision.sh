#!/usr/bin/env bash
# usage: bash scripts/run_revision.sh N_WORKERS
# Before the first call: run the smoke tests and the equivalence checks, and set ALPHA_DELTA from the readouts tuning run.
set -e
N=${1:-4}
git diff --quiet HEAD || { echo "tracked files modified: commit first"; exit 1; }
git -C .. diff --quiet HEAD || { echo "fly-brain (simulator) has modified tracked files"; exit 1; }
git ls-files --error-unmatch scripts/readouts/run_readouts.py scripts/plasticity/run_plasticity.py scripts/gridworld/run_gridworld_layouts.py agents/shared/readouts.py > /dev/null || { echo "new scripts not committed"; exit 1; }
python -m scripts.readouts.run_readouts --workers $N
python -m scripts.plasticity.run_plasticity --workers $N
python -m scripts.gridworld.run_gridworld_layouts --workers $N
echo "REVISION BATCHES FINISHED"
