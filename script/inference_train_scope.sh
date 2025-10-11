export PYTHONPATH=$PYTHONPATH:$(pwd)
export PLANSCOPE="$(pwd)/sanity_scope"
export CUDA_VISIBLE_DEVICES=2,3,5,7
cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
echo "====Start training====" &&

python run_scope_latent.py \
  py_func=train +training=train_scope \
  checkpoint="$CKPT_ROOT/frac1/last.ckpt" \
  epochs=1 warmup_epochs=1 \
  scenario_builder=nuplan cache.cache_path=$NUPLAN_EXP_ROOT/pluto_train cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=1 data_loader.params.num_workers=1 \
  worker=single_machine_thread_pool worker.max_workers=32 \
  lr=1e-3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  data_loader.datamodule.train_fraction=1.0 \
  data_loader.datamodule.val_fraction=1.0 \
  data_loader.datamodule.test_fraction=1.0 \
  model.cat_x=true model.ref_free_traj=true \
  +custom_trainer.use_contrast_loss=false model.use_hidden_proj=false \
  +custom_trainer.mul_ade_loss=[] \
  +custom_trainer.max_horizon=20 \
  +custom_trainer.dynamic_weight=false \
  +model.wtd_with_history=false +custom_trainer.wtd_with_history=false \
  model.recursive_decoder=true +model.multihead_decoder=false \
  model.future_steps=80 \
  +custom_trainer.learning_output='velocity' \
  +custom_trainer.init_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
  +custom_trainer.wavelet=['cgau1','constant','haar','constant'] \
  +stage='train' \
  && 
  echo "====Training End===="