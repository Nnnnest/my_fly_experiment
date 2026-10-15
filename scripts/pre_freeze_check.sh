#!/usr/bin/env bash
set -e
python -m scripts.diagnostics.verify_brain_reset
python -m scripts.diagnostics.verify_value_fn
python -m scripts.diagnostics.verify_addition_agent
python -m scripts.gridworld.verify_gridworld_v2_matches_v1
python -m scripts.tictactoe.verify_v2_matches_v1
python -m scripts.diagnostics.verify_symmetry_values
python -m scripts.bandit.run_bandit --smoke
python -m scripts.comparison.run_comparison --smoke
python -m scripts.gridworld.run_gridworld --smoke
python -m scripts.tictactoe.run_tictactoe --smoke
python -m scripts.addition.run_addition --smoke
python -m scripts.addition.run_addition_diagnostic --smoke
rm -rf runs/_smoke
echo "ALL CHECKS PASSED"
