export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=2,3,5,7
# python cflow/inference.py --data_dir /path/to/Redoubt/pluto_dataset --flow_checkpoint_path /path/to/Redoubt/checkpoints/pluto_cflow.ckpt 
python cflow/inference.py --data_dir pluto_dataset --flow_checkpoint_path plantf_flow-epoch=349-val_loss=0.0933.ckpt