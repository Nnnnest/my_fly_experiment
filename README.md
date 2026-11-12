# What a *Drosophila* mushroom-body connectome model can and cannot learn

Code, run records and analysis for a controlled comparison of the `mb` mode of the
[fly-brain](https://github.com/onnxaa/fly-brain) simulator (FlyWire mushroom-body circuit, 6,289 neurons) against
tabular Q-learning, a linear learner and a small MLP on bandit, number-comparison, maze, tic-tac-toe and addition tasks,
with diagnostics of the Kenyon-cell code, of the plasticity rule and of the ALPN-to-KC wiring.

- Paper: *(preprint link to be added after publication)*
- Code archive (Zenodo, v2.1): https://doi.org/10.5281/zenodo.23919513
- Run records and experiment splits (Zenodo): https://doi.org/10.5281/zenodo.23501884

The Zenodo dataset contains the raw experimental run records and associated experiment splits. The analysis code,
tables and figures are maintained in this GitHub repository.

## Main findings

Numbers, confidence intervals and caveats are in the paper and in `paper/tables/`.

- **Bandit.** At `min_pulls = 10`, mean cumulative pseudo-regret matched sample-average Q-learning (9.44 for both). From 8 pulls upward, observed mean differences were at most 0.2, but no equivalence margin was specified; below 8 pulls the connectome was worse and noisier. Occasional wrong-arm lock-in remained. With four arms and arm order controlled it was worse than Q-learning. An apparent
  advantage in the first four-armed run (`bandit_multi`) was a positional artifact, exposed by the arm-order control
  (`bandit_perm`).
- **Number comparison.** Competitive when a number is one intensity, including when evaluation pairs are strictly
  excluded from training; it degrades with positional digit encodings relative to tuned baselines.
- **Readouts on the same KC codes.** A delta rule or ridge regression reading the same KC codes beats the native
  plasticity rule on every task tested, so the native rule leaves task-relevant information unused. The contribution of
  the KC code itself is task-dependent (on digit comparison with 5 and 9 digits, raw inputs work better than the KC code).
- **Maze.** The ungated connectome agent degrades from 72 states. Training gates bring it within 0.02 of Q-learning at
  72 states and on the 113-state fixed start, also when the baselines receive the same binary feedback, but not on
  independent layouts of about 113 states or at 316 states. An MLP given the same binary feedback reaches 0.891
  success on the 316-state maze, showing that the connectome's failure is not an inherent impossibility of the task.
- **Tic-tac-toe.** Far below Q-learning and the MLP on minimax coverage in every tier. The odd `gate_visit` result against
  the heuristic opponent is a role effect (draws as first mover, loses as second mover).
- **Addition.** Single-digit: above raw-input baselines, below baselines given oracle features. Two-digit: no learning by
  the native rule, although a ridge readout on the KC code reaches 0.77 on held-out operand pairs.
- **Update strength and frequency.** Shrinking the native step to a tenth raises ungated maze success from 0.08 to 0.81
  (5 seeds) but does not remove collapses.
- **Wiring control.** Degree-preserving shuffles lower accuracy on number comparison and single-digit addition; on
  two-digit addition, the real circuit performs worse than the shuffled circuits.

## Requirements

- Linux and Python 3.12.3. Package versions are pinned in `requirements.txt`; run manifests record the interpreter, package versions and execution environment.
- The fly-brain simulator at commit `0ef5d843431f0a4bec8a199d6aca0a5a02f6e449`. The data files it ships
  (`mb_circuit.npz`, `mb_groups*.npz`) are used directly.
- All results were produced on an Intel Core i9-9900KF (8 cores / 16 threads), 62 GB RAM, no GPU.

## Installation

```bash
git clone https://github.com/onnxaa/fly-brain.git
cd fly-brain
git checkout 0ef5d843431f0a4bec8a199d6aca0a5a02f6e449
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/Nnnnest/my_fly_experiment.git
cd my_fly_experiment
git checkout v2.1-release
pip install -r requirements.txt
ls ../mb_circuit.npz ../mb_groups*.npz      # must list the simulator data files
```

This repository must live **inside** the fly-brain directory: `core/paths.py` takes the parent directory as the simulator
root.

## Quick check (minutes)

```bash
bash scripts/pre_freeze_check.sh
```

Runs the exactness and equivalence checks (value cache against `brain.step`, pooled-brain reset, symmetry wrapper, v2
against v1 agents) and a smoke run of every main experiment.

## Reproducing the results

Working rules:

- One run is one folder `runs/<experiment>/<task>__<agent>__<tier>__s<SEED>/` with `manifest.json` (configuration, seed,
  code and simulator commits, dirty flag, versions, hardware, status, runtime), `metrics.csv` (long format: step, split,
  metric, value) and `summary.json`.
- Final seeds are 0 to 9, tuning seeds 1000 to 1004 (disjoint). The layout experiments use layout seeds 100 to 109 with
  agent seeds 0 to 2.
- The runner caches by configuration hash and does **not** see code changes. After changing code that affects results,
  delete the affected `runs/<experiment>/` folder.
- Do not edit tracked files while a batch runs: runs record the git commit and a `dirty` flag, and the launcher scripts
  refuse to start on a dirty tree. All reported runs have `dirty = false`.

### Main experiments and controls

```bash
# hyperparameter tuning (separate seeds; stored in runs/_tuning)
python -m scripts.bandit.run_bandit --tuning --workers 14
python -m scripts.comparison.run_comparison --tuning --workers 14
python -m scripts.gridworld.run_gridworld --tuning --workers 14
python -m scripts.tictactoe.run_tictactoe --tuning --workers 14
python -m scripts.addition.run_addition --tuning --workers 14

# main experiments: 1,700 runs, about 10 h with 14 workers on the machine above
bash scripts/run_final.sh 14

# controls
bash scripts/run_wiring_control.sh 14                  # 3 shuffled ALPN->KC circuits -> runs_shuffled_s{0,1,2}/
python -m scripts.bandit.run_bandit_perm --workers 14  # arm-order control (2 and 4 arms)
python -m scripts.bandit.run_bandit_multi --workers 14 # first 4-arm run, superseded by run_bandit_perm (kept for transparency)
```

Wiring controls use environment variables handled in `core/paths.py`: `FLY_CIRCUIT_DIR` (directory with a modified
`mb_circuit.npz`) and `FLY_RUNS_DIR` (output directory). Each shuffled circuit has its own live-ALPN file in `assets/`.

### Follow-up experiments

```bash
# readouts (delta rule, ridge) on the same KC codes
python -m scripts.readouts.run_readouts --tuning --workers 4      # choose ALPHA_DELTA, edit it, commit
python -m scripts.readouts.run_readouts --workers 14

# plasticity sweep (update strength g and frequency f)
python -m scripts.plasticity.run_plasticity

# addition with held-out operand pairs
python -m scripts.addition.run_addition_disjoint

# KC-code overlap diagnostic (addition_diag; standalone rerun)
python -m scripts.addition.run_addition_diagnostic

# independent maze layouts (add --ungated for the slow ungated agent)
python -m scripts.gridworld.run_gridworld_layouts --workers 14

# number comparison with strictly disjoint evaluation pairs
bash scripts/run_disjoint.sh 14

# maze baselines with the connectome's binary feedback (tune first, then final)
python -m scripts.gridworld.run_gridworld_binary --tuning --workers 14
python -m analysis.binary_maze_tables --tuning                    # set ALPHA_Q_BIN / LR_MLP_BIN, commit
python -m scripts.gridworld.run_gridworld_binary --workers 14

# the same baselines on the independent layouts
python -m scripts.gridworld.run_gridworld_layouts_binary --workers 14

# role breakdown of tic-tac-toe play against the heuristic opponent
python -m scripts.tictactoe.run_tictactoe_roles --workers 14

bash scripts/check_followup2.sh                                           # equivalence checks + smoke runs
python -m scripts.readouts.run_readouts_constrained --tuning --workers 14 # choose N0_DECAY
python -m analysis.constrained_tables --tuning                            # edit N0_DECAY in the script, commit
bash scripts/run_followup2.sh 14                                          # kc_diag, comparison_capacity, readouts_constrained
python -m analysis.kc_diag_tables
python -m analysis.constrained_tables
python -m analysis.capacity_tables
```

### Analysis (run after the batches)

```bash
python -m analysis.validate            # run inventory, seeds, status, commits; writes compute_cost
python -m analysis.tables              # paper/tables/*.csv and *.tex
python -m analysis.wiring_tables       # real versus shuffled wiring
python -m analysis.perm_tables         # arm-order control
python -m analysis.extra_tables        # first 4-arm run
python -m analysis.readout_tables      # readouts on KC codes
python -m analysis.eval_overlap        # training/evaluation overlap in the main experiments
python -m analysis.disjoint_cmp_tables # disjoint number comparison, with a reproduction check against the readouts
python -m analysis.binary_maze_tables  # binary-feedback maze baselines
python -m analysis.layout_tables       # maze layout tables
python -m analysis.layouts_binary_tables
python -m analysis.plasticity_tables   # plasticity sweep
python -m analysis.roles_tables        # role breakdown (with a reproduction check against the main runs)
python -m analysis.figures             # paper/figures/*.pdf and *.png (needs the tables above)
python -m analysis.cost <runs_dir>              # per-task timing from any runs directory
python -m analysis.compare_runs <old_runs_dir>  # determinism check against an earlier run of the same ids
```

## Repository layout

```
agents/       connectome and baseline agents, per task; shared caches, readouts, encoders, tic-tac-toe solver, symmetry wrapper
encoders/     task -> ALPN-vector encoders (comparison, addition, tic-tac-toe)
envs/         bandit, grid world, maze generator, tic-tac-toe
opponents/    random and heuristic tic-tac-toe opponents
core/         paths, config (seeds), pooled brain instances, runner (manifests, caching, parallelism), loaders
scripts/      one runner per experiment, controls/, diagnostics/ (equivalence and exactness checks), launcher .sh files
analysis/     validation, tables, figures, controls, timing, determinism comparison
assets/       live ALPN indices (real circuit and the three shuffled circuits)
paper/        tables (csv, tex) and figures (pdf, png) used in the paper
```

## Provenance of the reported runs

| group | runs | code commit |
|---|---|---|
| bandit (2 arms), comparison, gridworld, tictactoe, addition, addition_diag | 1,700 | `2cfae74` (tag `v2-final`) |
| wiring control (3 shuffled circuits, 920 runs each) | 2,760 | `245dd05` (tag `v2-controls`) |
| first 4-arm bandit (`bandit_multi`, superseded) | 180 | `245dd05` |
| arm-order control `bandit_perm` | 720 | `6d532fd` (480), `5201997` (240) |
| readouts, plasticity sweep, independent layouts, addition with held-out operand pairs | 60, 505, 360, 160 | `271247d` |
| number comparison, disjoint evaluation (`comparison_disjoint`) | 360 | `8431750` |
| maze baselines with binary feedback (`gridworld_binary`), tic-tac-toe role replay (`tictactoe_roles`) | 100, 50 | `c1eb7db` |
| layouts with binary-feedback baselines (`gridworld_layouts_binary`) | 120 | `11d704a` |

A complete second execution of the 1,700 main runs was bit-identical to the first (`analysis.compare_runs`, maximum
difference 0). The role replays reproduced the corresponding main runs exactly. Wall-clock times of the batch runs are
inflated by parallel workers (14 workers on 8 physical cores); costs in the paper come from sequential single-process
runs (`analysis.cost`) and from the number of `train()` calls.

## Known limitations

See the Limitations section of the paper. In short: one maze instance per size in the main experiment; the 316-state maze
was not used for tuning; the connectome receives assists the baselines cannot use (calibration, k-hot encoding, training
gates); evaluation items overlap with training in the main number-comparison and addition experiments (quantified by
`analysis.eval_overlap`; disjoint controls exist for the key conditions); the wiring control covers only three shuffles of
the ALPN-to-KC projection; all experiments use `hops=1` and toy tasks. Vision experiments and early exploratory
diagnostics are not part of this pipeline (tag `v1-article`).

## Licenses

- **Code:** MIT (`LICENSE`).
- **Run records and associated research data:** CC BY-NC 4.0 (`DATA_LICENSE.md`), because they derive from the
  simulator and FlyWire data, which are CC BY-NC 4.0 (non-commercial use only).
- **Tables and figures:** available in `paper/`; see `DATA_LICENSE.md` for the applicable terms.

## Acknowledgements

FlyWire connectome (Dorkenwald et al., *Nature* 2024) and the `fly-brain` simulator (`onnxaa/fly-brain`). Parts of the code,
the analysis planning and the drafting of the manuscript were done with the assistance of Claude (Anthropic). ChatGPT
(OpenAI; GPT-6) was used for critical manuscript review and editorial revision. The author is responsible for all content.

## Citation

See `CITATION.cff`.


