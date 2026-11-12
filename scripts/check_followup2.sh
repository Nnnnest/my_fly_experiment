#!/usr/bin/env bash
# usage: bash scripts/check_followup2.sh     (equivalence checks, then a smoke run of each new experiment)
set -e
python -m scripts.diagnostics.verify_constrained_readouts
python -m scripts.diagnostics.verify_capacity_encoder
python -m scripts.diagnostics.verify_kc_native
python -m scripts.kc_diag.run_kc_diag --smoke
python -m scripts.readouts.run_readouts_constrained --smoke
python -m scripts.comparison.run_comparison_capacity --smoke
rm -rf runs/_smoke
echo "FOLLOWUP2 CHECKS PASSED"
