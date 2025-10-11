export PYTHONPATH=$PYTHONPATH:$(pwd)
export PLUTO=$(pwd)
export CUDA_VISIBLE_DEVICES=7

python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=$NUPLAN_EXP_ROOT/pluto_train cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  +custom_trainer.use_contrast_loss=true model.use_hidden_proj=true