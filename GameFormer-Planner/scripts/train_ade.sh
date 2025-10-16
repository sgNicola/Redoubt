export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=2,3,5,7

python ade/train_ade.py --data_dir "$(pwd)/gameformer_dataset" --model_name gameformer