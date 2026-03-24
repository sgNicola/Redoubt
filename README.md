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
### GameFormer cache example

```bash
cd GameFormer-Planner
sh test_cache.sh
```

`test_cache.sh` calls `cache_process.py` with `config/scenario_filter/train_InD.yaml` and writes logs to `cache.log`.

### Pluto cache example
1. cache the training data
```bash
script/cache_train_pluto.sh
```
2. cache the val data and put the val data under the training dataset
```bash
sh script/cache_val_pluto.sh
cd $NUPLAN_EXP_ROOT/pluto_val
mv $NUPLAN_EXP_ROOT/pluto_val/* $NUPLAN_EXP_ROOT/pluto_train
```

## Training
1. make checkpoints files for saving the trained planner checkpoints.
2. train planners:
```bash
sh script/train_pluto.sh
```
3. rename the best saved planner checkpoints and move to your checkpoints files
```bash
mkdir checkpoints
```
4. make folder for saving the pluto_dataset (latent features)
```bash
mkdir pluto_dataset
```

5. run inference scripts to get the latent features
```bash
#   +stage='train' 
sh script/inference_train_pluto.sh
```
change the stage in script to 'val'
```bash
#   +stage='val' 
sh script/inference_train_pluto.sh
```

This script includes a short sanity run and a full training section. Please update GPU IDs and workspace paths in the script before launching.

## Simulation

### Run pluto/scope/plantf simulation

example:
```bash
bash script/inference_sim_pluto.sh
```
Folder where all simulation results are stored: $NUPLAN_EXP_ROOT/exp/exp/simulation/open_loop_boxes/scenario_group_0/inference_pluto_planner
 

Then we can see that in pluto_datase, we get three folders: simulation_results, train_results, val_results.

Before simulation:
- Put checkpoints under `checkpoints/` (or update `CKPT_ROOT` in script).
- Ensure planner model settings used in simulation match those used during training.
- 
then we can see in the pluto_dataset. it contains 

## Train flow matching model on the latent features
change the path in train_cflow.sh
--data_dir /path/to/Redoubt/pluto_dataset --model_name pluto
'''
sh train_cflow.sh
'''
## Inference flow matching
change the data_dir and checkpoint path in script/flow_inference.sh
```bash
sh script/flow_inference.sh
```
## To Do

- [ ] Improve documentation.


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
