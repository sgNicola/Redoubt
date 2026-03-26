以下是为您整理好的 Markdown 源码。您可以直接将其复制并保存为 `README.md` 文件：

````markdown
# REDOUBT: Duo Safety Validation for Autonomous Vehicle Motion Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://openreview.net/forum?id=lEsvczuPVj)

REDOUBT builds upon the **nuPlan** ecosystem, integrating multiple planners and evaluation pipelines to enhance safety validation for autonomous vehicle motion planning.

---

## 📂 Repository Structure

* `src/`: Core training modules, models, and custom training logic.
* `config/`: Hydra configurations for training and simulation pipelines.
* `GameFormer-Planner/`: GameFormer-specific processing, training, and simulation.
* `script/`: Utility scripts for common experiments and automation.
* `cflow/`: Training implementation for **Flow Matching**.
* `ade/`: Training code for **Uncertainty Estimation**.
* `utils/`: Tools for analyzing simulation reports.

---

## 🚀 Getting Started

### 1. Prerequisites
REDOUBT requires two separate Conda environments due to conflicting PyTorch versions between planners (e.g., Pluto, PlanScope) and Flow Matching modules.

#### **Environment A: Planners**
Compatible with Pluto and PlanScope.
```bash
conda create -n planner python=3.9
conda activate planner

# Install nuplan-devkit
git clone [https://github.com/motional/nuplan-devkit.git](https://github.com/motional/nuplan-devkit.git)
cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt
cd ..
````

#### **Environment B: Flow Matching**

```bash
conda create -n flow python=3.9
conda activate flow
pip install -r ./flow_requirements.txt
```

### 2\. Installation

```bash
git clone [https://github.com/sgNicola/Redoubt.git](https://github.com/sgNicola/Redoubt.git)
cd Redoubt
conda activate planner
# Setup environment and install remaining dependencies
sh ./script/setup_env.sh
pip install -r ./requirements.txt
```

> [\!IMPORTANT]
> **Common Issue:** If you encounter nuPlan dataset setup problems, please refer to [nuplan-devkit issue \#379](https://github.com/motional/nuplan-devkit/issues/379).
> **Advised NATTEN Version:** `0.14.6+torch1121cu116`

### 3\. Environment Variables

Add the following to your shell config (e.g., `~/.bashrc`):

```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
export PYTHONPATH="$PYTHONPATH:/path/to/Redoubt"
```

-----

## 🛠️ Pipeline Workflow

### Phase 1: Data Processing (Pluto Example)

```bash
# Cache training and validation data
sh script/cache_train_pluto.sh
sh script/cache_val_pluto.sh

# Move val data under the training directory
cd $NUPLAN_EXP_ROOT/pluto_val
mv $NUPLAN_EXP_ROOT/pluto_val/* $NUPLAN_EXP_ROOT/pluto_train
```

### Phase 2: Training Planners

```bash
# Train the planner
sh script/train_pluto.sh

# Organize checkpoints
mkdir checkpoints
# (Manual Action) Rename the best saved checkpoint and move it to ./checkpoints
```

### Phase 3: Latent Feature Extraction

```bash
mkdir pluto_dataset

# Run inference to generate latent features for train and val stages
# For train:
sh script/inference_train_pluto.sh # Ensure +stage='train' is set in script
# For val:
sh script/inference_train_pluto.sh # Ensure +stage='val' is set in script
```

### Phase 4: Simulation

Modify `CHALLENGE` in `inference_sim_$planner$.sh` to select the type:

  * `closed_loop_nonreactive_agents`
  * `closed_loop_reactive_agents`
  * `open_loop_boxes`

<!-- end list -->

```bash
bash script/inference_sim_pluto.sh
```

Results will be stored in: `$NUPLAN_EXP_ROOT/exp/exp/simulation/...`

-----

## 🌊 Flow Matching & Evaluation

### 1\. Training Flow Matching

Train on the latent features generated in Phase 3.

```bash
conda activate flow
# Update paths in train_cflow.sh: --data_dir /path/to/Redoubt/pluto_dataset
sh train_cflow.sh
```

### 2\. Inference

```bash
# Update data_dir and checkpoint paths in script/flow_inference.sh
sh script/flow_inference.sh
```

### 3\. Safety Evaluation (OOD)

Evaluate the planner based on the density files generated.

```bash
python evaluation_ood.py \
  --planner pluto \
  --benchmark closed_loop_nonreactive_agents \
  --density-file pluto_dataset/prediction/log_density.parquet
```

-----

## 📝 Roadmap

  - [ ] Improve documentation.
  - [ ] Add support for more baseline planners.

## 🤝 Acknowledgements

This project stands on the shoulders of the open-source community:

  * [planTF](https://www.google.com/search?q=https://github.com/nuPlan-ranking/planTF) | [GameFormer-Planner](https://www.google.com/search?q=https://github.com/ZhiyuHuang/GameFormer-Planner) | [Pluto](https://www.google.com/search?q=https://github.com/vueren/pluto) | [PlanScope](https://github.com/Rex-sys-hk/PlanScope)

## 📧 Contact

For questions or suggestions, please open an **Issue** or contact:
**Shuguang Wang**: [shuguangwang6@gmail.com](mailto:shuguangwang6@gmail.com)

## 🎓 Citation

If you find this repository useful, please consider giving it a star ⭐ and citing our work:

```bibtex
@inproceedings{
wang2025redoubt,
title={{REDOUBT}: Duo Safety Validation for Autonomous Vehicle Motion Planning},
author={Shuguang Wang and Qian Zhou and Kui Wu and Dapeng Wu and Wei-Bin Lee and Jianping Wang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=lEsvczuPVj}
}
```

 