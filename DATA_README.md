# What a Drosophila mushroom-body connectome model can and cannot learn in a simulator: a benchmark against Q-learning and small neural networks

This archive contains the raw experimental run records and saved experiment splits supporting the paper. The
experiments use the `mb` mode of the [fly-brain simulator](https://github.com/onnxaa/fly-brain), built from FlyWire
mushroom-body connectivity, and compare it with conventional learning approaches.

**Archive contents:** 8,075 retained run records across the main experiments, follow-up experiments and controls. Hyperparameter-tuning runs, smoke tests and obsolete tuning records are excluded.

Code repository: https://github.com/Nnnnest/my_fly_experiment
Code version: `v2.3-release`

## Structure

Each run is stored as:

```text
runs/<experiment>/<task>__<agent>__<tier>__s<SEED>/
    manifest.json   configuration and hash, seed, code and simulator commits, dirty flag,
                    software versions, hardware, status, runtime and CPU time
    metrics.csv     long format: step, split, metric, value
    summary.json    final run metrics
```

The archive also contains:

- `runs/_splits/`: saved training, calibration and evaluation pairs for disjoint number-comparison experiments.
- `runs_shuffled_s0/`, `runs_shuffled_s1/`, `runs_shuffled_s2/`: wiring-control records from three degree-preserving
  shuffles of the ALPN-to-KC projection.
- `DATA_README.md` and `DATA_LICENSE.md`.

## Experiments

| Folder / group | Runs | Code commit | Content |
|---|---:|---|---|
| `bandit` | 210 | `2cfae74` | Two-armed bandit, forced-exploration sweep |
| `comparison` | 450 | `2cfae74` | Number comparison (intensity and digit encodings) |
| `gridworld` | 440 | `2cfae74` | Mazes of 26, 72, 113 and 316 states |
| `tictactoe` | 340 | `2cfae74` | Reinforcement learning, imitation and self-play |
| `addition` | 240 | `2cfae74` | Single- and two-digit addition |
| `addition_diag` | 20 | `2cfae74` | KC-code overlap diagnostic |
| `bandit_multi` | 180 | `245dd05` | Initial four-armed bandit; superseded because of a positional artifact, retained for transparency |
| `bandit_perm` | 720 | `6d532fd` (480), `5201997` (240) | Arm-order control for two- and four-armed bandits |
| `runs_shuffled_s0/1/2` | 2,760 | `245dd05` | Three degree-preserving ALPN-to-KC wiring shuffles; 920 runs per shuffle |
| `readouts` | 60 | `271247d` | Delta-rule and ridge readouts on identical input-only KC codes |
| `plasticity` | 505 | `271247d` | Update-strength and update-frequency sweep |
| `gridworld_layouts` | 360 | `271247d` | Independent maze layouts |
| `addition_disjoint` | 160 | `271247d` | Addition with held-out operand pairs |
| `comparison_disjoint` | 360 | `8431750` | Number comparison with evaluation pairs excluded from training |
| `gridworld_binary` | 100 | `c1eb7db` | Q-learning and MLP baselines using the connectome's binary feedback |
| `tictactoe_roles` | 50 | `c1eb7db` | Role breakdown against the heuristic opponent |
| `gridworld_layouts_binary` | 120 | `11d704a` | Binary-feedback baselines on independent layouts |
| `nocalib` | 180 | `c064e1b` | Calibration ablation: full OLS, constant-offset correction and no correction |
| `comparison_capacity` | 420 | `baa8a3d` | Digit comparison under alternative ALPN-capacity allocations |
| `kc_diag` | 60 | `baa8a3d` | KC-selection diagnostics, including training/evaluation mask frequencies and exclusion tiers |
| `readouts_constrained` | 40 | `baa8a3d` | Signed, non-negative and anatomically constrained readout comparisons |
| `addition_dan_kc` | 60 | `9cba3cb` | DAN-modulated versus input-only KC activity and input-only-mask ablation |
| `bandit_punish` | 240 | `9cba3cb` | Four-armed bandit: reward-only versus reward-plus-punishment training |
| **Total** | **8,075** |  | **Retained experimental run records** |

All runs use the fly-brain simulator at commit
`0ef5d843431f0a4bec8a199d6aca0a5a02f6e449`.

## Calibration ablation

The `runs/nocalib/` experiment compares three readout-calibration conditions:

- `full`: standard label-blind ordinary least-squares (OLS) calibration.
- `const`: constant-offset correction only, using the mean untrained readout over the calibration sample.
- `raw`: no calibration correction.

It contains 180 runs across six task/configuration groups, with ten seeds per condition and group. The groups cover
two-, five- and nine-digit number comparison, number comparison with `n_max = 49`, and single- and two-digit addition
with held-out operand pairs. The 60 `full` runs reproduce the corresponding calibrated results exactly, with a maximum
absolute difference of zero.

## Reproduction notes

- Final seeds are 0–9; tuning seeds 1000–1004 are excluded from this archive. Maze layout experiments use layout seeds
  100–109 and three agent seeds per layout.
- All archived run manifests should have `status = ok` and `dirty = false`.
- The 1,700 main runs were independently repeated and reproduced bit-identically. Tic-tac-toe role replays also
  reproduced their corresponding main runs exactly.
- The first four-arm `bandit_multi` experiment is superseded by the arm-order control and is not used as a main result.
- Wall-clock times in parallel-run manifests are inflated by worker contention; see the paper for the cost measures used.
- Machine: Intel Core i9-9900KF (8 cores, 16 threads), 62 GB RAM, no GPU; Python 3.12.3.

## Citation

Please cite the associated paper:

Zudov, N. *What a Drosophila mushroom-body connectome model can and cannot learn in a simulator: a benchmark against Q-learning and small neural networks*.

Also cite the FlyWire connectome (Dorkenwald et al., *Nature*, 2024) and the `fly-brain` simulator
(https://github.com/onnxaa/fly-brain).

## License

CC BY-NC 4.0. The run records derive from the fly-brain simulator and FlyWire data; consult the upstream licences
and the included `DATA_LICENSE.md` before reuse.
