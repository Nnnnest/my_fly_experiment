#!/usr/bin/env bash
# usage: bash scripts/run_disjoint.sh N_WORKERS
set -e
N=${1:-4}
git diff --quiet HEAD || { echo "tracked files modified: commit first"; exit 1; }
git -C .. diff --quiet HEAD || { echo "fly-brain (simulator) has modified tracked files"; exit 1; }
python -m scripts.diagnostics.verify_disjoint_split
python -m scripts.comparison.run_comparison_disjoint --workers $N
echo "DISJOINT RUNS FINISHED"
