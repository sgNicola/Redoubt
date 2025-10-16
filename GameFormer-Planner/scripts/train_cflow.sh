export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=2,3,5,7
python cflow/train_cflow.py --data_dir "$(pwd)/gameformer_dataset" --model_name gameformer