# REDOUBT
This repository contains the code for  
**REDOUBT: Duo Safety Validation for Autonomous Vehicle Motion Planning**
<br> [City University of Hong Kong]

## Overview

## Getting started
### 1. Installation
To begin, please follow these steps:
- Download the [nuPlan dataset](https://www.nuscenes.org/nuplan#download) and set it up as described [here](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html). 
- Install the nuPlan devkit [here](https://nuplan-devkit.readthedocs.io/en/latest/installation.html) (version tested: v1.2.2). 
- Clone this repository and navigate into the folder:
```
git clone  && cd  
```
- Activate the environment created when installing the nuPlan-devkit:
```
conda activate nuplan
```
- Install the required packages:
```
pip install -r requirements.txt
```
- Add the following environment variable to your `~/.bashrc` (you can customize it):
```
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
```

### 2. Data process
Before training the GameFormer model, you need to preprocess the raw data using:
```
python data_process.py \
--data_path nuplan/dataset/nuplan-v1.1/splits/mini \
--map_path nuplan/dataset/maps \
--save_path nuplan/processed_data
```
Three arguments are necessary: ```--data_path``` to specify the path to the stored nuPlan dataset, ```--map_path``` to specify the path to the nuPlan map data, and ```--save_path``` to specify the path to save the processed data. 

Optional arguments like ```--scenarios_per_type``` and ```--total_scenarios``` can also be used to specify the amount of data to process.

### 3. Training
### 4. Simulation
**Make sure the model parameters in ```planner.py``` in ```_initialize_model``` match those used in training.**

###  To Do

###  To Do
The code is under cleaning and will be released gradually.
-[] improve docs
-[] training code
-[] feature builder 
-[] initial repo & paper

## Contact
If you have any questions or suggestions, please feel free to open an issue or contact us .

## Citation
If you find this repository useful for your research, please consider giving us a star; and citing our paper.

``` 

```
