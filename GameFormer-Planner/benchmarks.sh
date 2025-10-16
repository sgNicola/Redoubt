export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=6,7

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER="Inference"
SPLIT=$1
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"
        # worker.threads_per_node=20 \
# CHALLENGES="closed_loop_nonreactive_agents"
export CUDA_VISIBLE_DEVICES=6,7
for challenge in $CHALLENGES; do
    CUDA_LAUNCH_BLOCKING=1
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        experiment_uid=$SPLIT/$PLANNER \
        worker.threads_per_node=80 \
        verbose=true \
        planner.planner.model_path="$CKPT_ROOT/gameInD.pth"
done