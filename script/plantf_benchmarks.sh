cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"
SPLIT='test14-random'
# CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"
CHALLENGES="open_loop_boxes"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PLANTF=$(pwd)
for challenge in $CHALLENGES; do
    CUDA_LAUNCH_BLOCKING=1
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=80 \
        experiment_uid=$SPLIT/$PLANNER \
        verbose=true \
        planner.imitation_planner.planner_ckpt="$CKPT_ROOT/plantf/last.ckpt"
done