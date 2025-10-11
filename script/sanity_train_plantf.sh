export PYTHONPATH=$PYTHONPATH:$(pwd)
export PLANTF=$(pwd)
export CUDA_VISIBLE_DEVICES=7
python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=$NUPLAN_EXP_ROOT/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=16 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=32 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 