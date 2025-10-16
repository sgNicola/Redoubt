export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=6,7
python cflow/train_ablition.py --data_dir "$(pwd)/gameformer_dataset" --model_name gameformer
