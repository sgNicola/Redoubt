# REDOUBT

This repository contains the code for **REDOUBT: Duo Safety Validation for Autonomous Vehicle Motion Planning**.

## Overview

`REDOUBT` builds on the nuPlan ecosystem and integrates multiple planners and evaluation pipelines for autonomous driving research.

Current repository highlights:
- Scope/Pluto-style training and simulation entry points in the project root.
- `GameFormer-Planner` integration for data processing, caching, and planner experiments.
- nuPlan-based closed-loop/open-loop simulation scripts.

## Repository Structure

- `src/`: core training modules, models, and custom training logic.
- `config/`: hydra configs for training/simulation in the root pipeline.
- `GameFormer-Planner/`: GameFormer-related processing, training, simulation, and configs.
- `run_training.py`: main training entry.
- `run_simulation.py`: main simulation entry.
- `train_scope.sh`, `sim_scope.sh`: convenient scripts for common experiments.

## Getting Started

### 1) Prerequisites

- Python/Conda environment compatible with nuPlan devkit.
- nuPlan devkit installed (tested with `v1.2.2`):  
  [nuPlan installation guide](https://nuplan-devkit.readthedocs.io/en/latest/installation.html)
- nuPlan dataset downloaded and prepared:  
  [nuPlan dataset setup](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

### 2) Installation

```bash
git clone https://github.com/sgNicola/Redoubt.git
cd Redoubt
conda activate nuplan
pip install -r requirements.txt
```

If your root workflow needs additional packages beyond the nuPlan environment, install them as needed.

### 3) Environment Variables

Set environment variables in your shell config (`~/.bashrc` as an example):

```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
export PYTHONPATH="$PYTHONPATH:/path/to/Redoubt"
```

Then reload:

```bash
source ~/.bashrc
```

## Data Processing

### Root pipeline preprocessing

Use the `GameFormer-Planner` preprocessing script:

```bash
cd GameFormer-Planner
python data_process.py \
  --data_path "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini" \
  --map_path "$NUPLAN_MAPS_ROOT" \
  --save_path "$NUPLAN_EXP_ROOT/processed_data"
```

Required arguments:
- `--data_path`: nuPlan scenario database path.
- `--map_path`: nuPlan maps path.
- `--save_path`: output directory for processed data.

Optional controls such as `--scenarios_per_type` and `--total_scenarios` can be used to limit processing size.

### GameFormer cache example

```bash
cd GameFormer-Planner
sh test_cache.sh
```

`test_cache.sh` calls `cache_process.py` with `config/scenario_filter/train_InD.yaml` and writes logs to `cache.log`.

## Training

### Train Scope (root pipeline)

```bash
cd /path/to/Redoubt
sh train_scope.sh
```

This script includes a short sanity run and a full training section. Please update GPU IDs and workspace paths in the script before launching.

## Simulation

### Run Scope simulation

```bash
cd /path/to/Redoubt
sh sim_scope.sh
```

Before simulation:
- Put checkpoints under `checkpoints/` (or update `CKPT_ROOT` in script).
- Ensure planner model settings used in simulation match those used during training.
- Choose proper scenario builder/filter in the script (`nuplan_mini`, `mini_demo_scenario`, etc.).

## To Do

- [ ] Improve documentation.
- [ ] Release more complete training recipes.
- [ ] Release feature builder details.
- [ ] Finalize paper and reproducibility package.

## Acknowledgements

Many thanks to the open-source community. Related projects:
- [planTF](https://github.com/jchengai/planTF)
- [GameFormer-Planner](https://github.com/MCZhi/GameFormer-Planner)
- [Pluto](https://github.com/jchengai/pluto)
- [PlanScope](https://github.com/Rex-sys-hk/PlanScope)

## Contact

If you have any questions or suggestions, please open an issue or contact:
- shuguangwang6@gmail.com

## Citation

If this repository is useful for your research, please consider giving it a star.
The citation entry will be updated after publication details are finalized.
