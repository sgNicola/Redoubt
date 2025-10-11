export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=$NUPLAN_EXP_ROOT/pluto_resample \
    cache.cleanup_cache=true \
    scenario_filter=resample \
    worker.threads_per_node=80