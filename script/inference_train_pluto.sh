export PYTHONPATH=$PYTHONPATH:$(pwd)
export PLUTO="$(pwd)/pluto_dataset"
export CUDA_VISIBLE_DEVICES=2,3,5,7
cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
echo "====Start training====" &&

python run_pluto_latent.py \
  py_func=train +training=train_pluto \
  checkpoint="$CKPT_ROOT/pluto_cat.ckpt" \
  epochs=1 warmup_epochs=1 \
  scenario_builder=nuplan cache.cache_path=$NUPLAN_EXP_ROOT/pluto_train cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=1 data_loader.params.num_workers=1 \
  worker=single_machine_thread_pool worker.max_workers=32 \
  lr=1e-3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  data_loader.datamodule.train_fraction=1.0 \
  data_loader.datamodule.val_fraction=1.0 \
  data_loader.datamodule.test_fraction=1.0 \
  model.use_hidden_proj=true +custom_trainer.use_contrast_loss=false \
  model.cat_x=true model.ref_free_traj=true \
  model.num_modes=12 \
  +stage='val' \
  &&

  echo "====Training End===="