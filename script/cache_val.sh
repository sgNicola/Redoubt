export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=$NUPLAN_EXP_ROOT/pluto_val \
    cache.cleanup_cache=true \
    scenario_filter=val_InD \
    worker.threads_per_node=80