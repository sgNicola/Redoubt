export PYTHONPATH=$PYTHONPATH:$(pwd)
export PLANSCOPE=$(pwd)
export CUDA_VISIBLE_DEVICES=6,7
# python run_training.py \
#   py_func=cache +training=train_pluto \
#   scenario_builder=nuplan_mini \
#   cache.cache_path=$WS/exp/sanity_check \
#   cache.cleanup_cache=true \
#   scenario_filter=training_scenarios_tiny \
#   worker=sequential &&
   
# python run_training.py \
#   py_func=cache +training=train_pluto \
#   scenario_builder=nuplan \
#   cache.cache_path=$WS/exp/cache_pluto_1M \
#   cache.cleanup_cache=true \
#   scenario_filter=training_scenarios_1M \
#   worker.threads_per_node=40

echo "====Start Sanity Check====" &&

CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  epochs=2 warmup_epochs=1 \
  scenario_builder=nuplan cache.cache_path=$NUPLAN_EXP_ROOT/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1 \
  model.use_hidden_proj=false +custom_trainer.use_contrast_loss=false \
  model.cat_x=true model.ref_free_traj=true \
  model.num_modes=6 \
  &&

  echo "====End training===="
