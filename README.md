# What a *Drosophila* mushroom-body connectome model can and cannot learn

Code, run records and analysis for a controlled comparison of the `mb` mode of the
[fly-brain](https://github.com/onnxaa/fly-brain) simulator (FlyWire mushroom-body circuit, 6,289 neurons) against
tabular Q-learning, a linear learner and a small MLP on a bandit, number comparison, maze navigation,
tic-tac-toe and addition tasks.

Paper: *(preprint link and DOI: to be added)*  ·  Archive of all run records: *(Zenodo DOI: to be added)*

## Main findings

- **Bandit.** With two arms the connectome is statistically indistinguishable from Q-learning once at least 8 forced pulls
  per arm are used (with an occasional wrong-arm lock-in). With four arms and arm order controlled it is worse than
  Q-learning. An apparent advantage in the first 4-arm run (`bandit_multi`) was a positional artifact, exposed by the
  arm-order control (`bandit_perm`).
- **Number comparison.** Competitive when a number is one intensity; it degrades with positional digit encodings
  relative to tuned baselines.
- **Wiring control.** Degree-preserving shuffles of the ALPN→KC wiring lower accuracy on number comparison and
  single-digit addition. They do not lower it on two-digit addition, where the real circuit does not learn.
- **Maze.** The untouched connectome agent degrades from 72 states. External training gates recover most of the gap up
  to 113 states. All connectome variants fail at 316 states.
- **Tic-tac-toe.** Far below Q-learning and the MLP on minimax coverage in every tier.
- **Addition.** Single-digit: above raw-input baselines, below baselines given oracle features. Two-digit: no learning.

Numbers, confidence intervals and caveats are in the paper and in `paper/tables/`.

## Requirements

- Linux, Python 3 (exact interpreter and package versions are in `requirements.txt` and in every run's `manifest.json`).
- The fly-brain simulator at commit `0ef5d843431f0a4bec8a199d6aca0a5a02f6e449`. The data files it ships
  (`mb_circuit.npz`, `mb_groups*.npz`) are used directly.
- All results were produced on an Intel i9-9900KF (8 cores / 16 threads), 62 GB RAM, no GPU.

## Installation

```bash
git clone https://github.com/onnxaa/fly-brain.git
cd fly-brain
git checkout 0ef5d843431f0a4bec8a199d6aca0a5a02f6e449
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/Nnnnest/my_fly_experiment.git
cd my_fly_experiment
pip install -r requirements.txt
ls ../mb_circuit.npz ../mb_groups*.npz      # must list the simulator data files
```

This repository must live **inside** the fly-brain directory: `core/paths.py` takes the parent directory as the
simulator root.

## Quick check (minutes)

```bash
bash scripts/pre_freeze_check.sh
```

Runs the exactness and equivalence checks (value cache vs `brain.step`, pooled-brain reset, symmetry wrapper, v2 vs v1
agents) and a smoke run of every experiment.

## Reproducing the results

Working rules:

- One run = one folder `runs/<experiment>/<task>__<agent>__<tier>__s<SEED>/` with `manifest.json` (config, seed, commits,
  versions, hardware, status, runtime), `metrics.csv` (long format: step, split, metric, value) and `summary.json`.
- Final seeds are 0–9, tuning seeds 1000–1004 (disjoint).
- The runner caches by config hash and does **not** see code changes. After changing code that affects results, delete
  the affected `runs/<experiment>/` folder.
- Do not edit tracked files while a batch runs: runs record the git commit and a `dirty` flag, and `run_final.sh`
  refuses to start on a dirty tree.

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
python -m scripts.bandit.run_bandit_perm --workers 14  # arm-order control of the bandit (2 and 4 arms)
python -m scripts.bandit.run_bandit_multi --workers 14 # first 4-arm run, superseded by run_bandit_perm (kept for transparency)
```

Wiring controls use environment variables handled in `core/paths.py`: `FLY_CIRCUIT_DIR` (directory with a modified
`mb_circuit.npz`) and `FLY_RUNS_DIR` (output directory). Each shuffled circuit has its own live-ALPN file in `assets/`.

### Analysis (run in this order)

```bash
python -m analysis.validate         # run inventory, seeds, status, commits; writes compute_cost
python -m analysis.tables           # paper/tables/*.csv and *.tex, printed to stdout
python -m analysis.wiring_tables    # real vs shuffled wiring
python -m analysis.perm_tables      # arm-order control
python -m analysis.extra_tables     # first 4-arm run
python -m analysis.figures          # paper/figures/*.pdf and *.png (needs the tables above)
python -m analysis.cost <runs_dir>              # per-task timing from any runs directory
python -m analysis.compare_runs <old_runs_dir>  # determinism check against an earlier run of the same ids
```

## Repository layout

```
agents/       connectome and baseline agents, per task; shared caches, encoders, tic-tac-toe solver, symmetry wrapper
encoders/     task -> ALPN-vector encoders (comparison, addition, tic-tac-toe)
envs/         bandit, grid world, maze generator, tic-tac-toe
opponents/    random and heuristic tic-tac-toe opponents
core/         paths, config (seeds), pooled brain instances, runner (manifests, caching, parallelism), loaders
scripts/      one runner per experiment, controls/, diagnostics/ (equivalence and exactness checks), run_final.sh
analysis/     validation, tables, figures, controls, timing, determinism comparison
assets/       live ALPN indices (real circuit and the three shuffled circuits)
```

## Provenance of the reported runs

| group | runs | code |
|---|---|---|
| bandit (2 arms), comparison, gridworld, tictactoe, addition, addition_diag | 1,700 | `2cfae74` (tag `v2-final`) |
| wiring control (shuffled circuits) and first 4-arm bandit | 3 × 920 + 180 | `245dd05` (tag `v2-controls`) |
| arm-order control `bandit_perm` | 720 | `6d532fd` (480 runs), `5201997` (240 runs) |

A full second run of all 1,700 main runs was bit-identical to the first (`analysis.compare_runs`; the only differences are
floating-point noise up to 4e-4 in `addition_diag`). Wall-clock times of the batch runs are inflated 2–7x by
14 parallel workers on 8 physical cores. Costs in the paper come from sequential single-thread runs (`analysis.cost`),
and for the maze experiments from the number of `train()` calls.

## Known limitations

- One maze instance per size (`generate_maze(..., seed=0)`). The 316-state maze was tuned only for the delta agents.
- Tic-tac-toe Q-learning step size 0.1 was used; 0.3 (grid edge) was marginally better in tuning.
- Best-cell tuning optimism: baselines were tuned on the same task families they are evaluated on.
- The wiring control shuffles only the ALPN→KC connections (3 shuffles).
- `connectome [gate_visit]` against the heuristic tic-tac-toe opponent shows draw 0.526 / loss 0.474,
  which means one role always draws and the other always loses. This affects one cell of one task.
- Exploratory diagnostics and vision experiments of the first analysis are not part of this pipeline
  (see tag `v1-article`).

## Licenses

- **Code:** MIT (`LICENSE`).
- **Run records, tables and figures:** CC BY-NC 4.0 (`DATA_LICENSE.md`), because they derive from the simulator and
  FlyWire data, which are CC BY-NC 4.0 (non-commercial use only).

## Acknowledgements

FlyWire connectome (Dorkenwald et al., *Nature* 2024) and the `fly-brain` simulator (`onnxaa/fly-brain`). Parts of the code
and the analysis planning were developed with the assistance of an AI model (Claude, Anthropic); the author is
responsible for all content.

## Citation

See `CITATION.cff`.
