#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SIM_DATA="$(pwd)/gameformer_dataset"
cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=inference_gameformer_planner
BUILDER=nuplan_challenge
FILTERS=(
    "scenario_group_0"
    "scenario_group_1"
    "scenario_group_2" 
    "scenario_group_3"
    "scenario_group_4"
)
# CHALLENGE="open_loop_boxes"
CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"

CUDA_VISIBLE_DEVICES=''
for FILTER in "${FILTERS[@]}"; do
    python run_simulation.py \
        +simulation=$CHALLENGE \
        planner=$PLANNER \
        scenario_builder=$BUILDER \
        scenario_filter=$FILTER \
        worker.threads_per_node=80 \
        verbose=true \
        experiment_uid="$FILTER/$PLANNER" \
        planner.$PLANNER.model_path="$CKPT_ROOT/game0406.pth" 
done