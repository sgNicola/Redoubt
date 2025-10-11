#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SIM_DATA="$(pwd)/planscope_dataset"
cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=inference_scope_planner

CKPT=last.ckpt
BUILDER=nuplan_challenge

# FILTERS=(
#     "scenario_group_0"
#     "scenario_group_1"
#     "scenario_group_2" 
#     "scenario_group_3"
#     "scenario_group_4"
# )

FILTERS=(
    "test14-random"
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
        planner.$PLANNER.render=false \
        planner.$PLANNER.planner_ckpt="$CKPT_ROOT/frac1/$CKPT" \
        +planner.$PLANNER.save_dir=$VIDEO_SAVE_DIR/$CHALLENGE.norule \
        planner.$PLANNER.rule_based_evaluator=false \
        planner.$PLANNER.planner.use_hidden_proj=false \
        planner.$PLANNER.planner.cat_x=true \
        planner.$PLANNER.planner.ref_free_traj=true \
        planner.$PLANNER.planner.num_modes=6 \
        planner.$PLANNER.planner.future_steps=80 \
        planner.$PLANNER.planner.recursive_decoder=true \
        +planner.$PLANNER.planner.multihead_decoder=false \
        +planner.$PLANNER.planner.wtd_with_history=false
done