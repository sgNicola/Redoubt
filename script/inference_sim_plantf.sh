#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SIM_DATA="$(pwd)/plantf_dataset"
cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=inference_plantf_planner

CKPT=plantf.ckpt
BUILDER=nuplan_challenge

FILTERS=(
    "scenario_group_0"
    "scenario_group_1"
    "scenario_group_2" 
    "scenario_group_3"
    "scenario_group_4"
)

VIDEO_SAVE_DIR=$cwd/videos/$PLANNER.$CKPT_N/$FILTER

# CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
CHALLENGE="open_loop_boxes"

CUDA_VISIBLE_DEVICES=''
for FILTER in "${FILTERS[@]}"; do
    python run_simulation_inference.py \
        +simulation=$CHALLENGE \
        planner=$PLANNER \
        scenario_builder=$BUILDER \
        scenario_filter=$FILTER \
        worker.threads_per_node=80 \
        verbose=true \
        experiment_uid="$FILTER/$PLANNER" \
        planner.$PLANNER.planner_ckpt="$CKPT_ROOT/plantf/plantf.ckpt" 
done