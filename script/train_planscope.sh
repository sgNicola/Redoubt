export PYTHONPATH=$PYTHONPATH:$(pwd)
export PLANSCOPE=$(pwd)
export CUDA_VISIBLE_DEVICES=2,3,5,7

echo "====Start training====" &&

CUDA_VISIBLE_DEVICES=2,3,5,7 python run_training.py \
  py_func=train +training=train_scope \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.use_cache_without_dataset=true \
  cache.cache_path=$NUPLAN_EXP_ROOT/pluto_train \
  data_loader.params.batch_size=32 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=35 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  data_loader.datamodule.train_fraction=1.0 \
  data_loader.datamodule.val_fraction=1.0 \
  data_loader.datamodule.test_fraction=1.0 \
  model.cat_x=true model.ref_free_traj=true \
  +custom_trainer.use_contrast_loss=true model.use_hidden_proj=true \
  +custom_trainer.mul_ade_loss=[] \
  +custom_trainer.max_horizon=20 \
  +custom_trainer.dynamic_weight=false \
  +model.wtd_with_history=false +custom_trainer.wtd_with_history=false \
  model.recursive_decoder=true +model.multihead_decoder=false \
  model.future_steps=80 \
  +custom_trainer.learning_output='velocity' \
  +custom_trainer.init_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
  +custom_trainer.wavelet=['cgau1','constant','haar','constant'] \
  &&

  echo "====Training End===="