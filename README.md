# My Fly Experiments

A collection of experiments exploring the learning capabilities and limitations of the *Drosophila melanogaster* mushroom body connectome using the [fly-brain](https://github.com/onnxaa/fly-brain) simulator.

The goal of this project is not to build the strongest possible reinforcement learning agent, but to investigate which kinds of tasks can be learned by a biologically inspired learning system and where its limitations begin to appear.

Most experiments compare connectome-based agents against conventional machine learning baselines such as MLPs and Q-learning.

## Experiments

Current experiments include:

* Multi-armed bandits
* Number comparison
* Maze / gridworld navigation
* Tic-tac-toe
* Visual classification tasks
* Arithmetic tasks
* Various diagnostic and analysis tools

The repository contains both the experiment implementations and the code used to generate the results, plots, and tables presented in the accompanying article.

## Installation

This repository depends on the fly-brain project and is intended to be placed inside the fly-brain directory.

Clone and set up fly-brain:

```bash
git clone https://github.com/onnxaa/fly-brain.git
cd fly-brain

python3 -m venv .venv
source .venv/bin/activate
```

Clone this repository inside the fly-brain directory:

```bash
git clone https://github.com/Nnnnest/my_fly_experiment.git
```

The resulting structure should look similar to:

```text
fly-brain/
├── api.py
├── ...
└── my_fly_experiment/
    ├── agents/
    ├── encoders/
    ├── envs/
    ├── models/
    ├── results/
    └── scripts/
```

## Running Experiments

Run experiments from the `my_fly_experiment` root directory while the virtual environment is activated.

### Multi-Armed Bandit

```bash
python scripts/bandit/run_bandit.py
```

### Number Comparison

```bash
python scripts/comparison/run_comparison.py
```

### Gridworld / Maze Navigation

```bash
python scripts/gridworld/run_gridworld.py
```

### Tic-Tac-Toe

```bash
python scripts/tictactoe/play_tictactoe.py
```

### Visual Classification

```bash
python scripts/shapes/run_shapes.py
```

Many experiments support additional command-line arguments. See the individual script files for available options.

## Repository Structure

```text
my_fly_experiment/
├── agents/        # Connectome and baseline agents
├── encoders/      # State encoding implementations
├── envs/          # Task environments
├── models/        # Saved and trained models
├── results/       # Experimental outputs, logs, and plots
├── scripts/       # Runnable experiments
├── tests/         # Validation and test code
└── utils/         # Shared utilities
```

## Research Focus

The experiments are designed to answer questions such as:

* How competitive is connectome-based learning compared to standard methods?
* How well does the mushroom body scale to larger problems?
* Which limitations arise from the biological architecture itself?
* How sensitive is performance to the choice of representation?
* Can a connectome learn simple symbolic or arithmetic tasks?

